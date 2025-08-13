'''
Based on https://github.com/microsoft/vscode/issues/125993 
baseline
python -u train_BASE.py 20 full fitzpatrick BASE
python -u train_BASE.py 15 full ddi BASE
'''
from __future__ import print_function, division
from sklearn.decomposition import TruncatedSVD
import torch
from torchvision import transforms, models
import pandas as pd
import numpy as np
import os
import skimage
import cv2
from skimage import io
import warnings
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import WeightedRandomSampler
from torch.optim import lr_scheduler
import time
import copy
import sys
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
# get model
from Models.models_losses import Network , Supervised_Contrastive_Loss
from transformers import AutoTokenizer, AutoModel
from GOT import cost_matrix_batch_torch, GW_distance_uniform, IPOT_distance_torch_batch_uniform


warnings.filterwarnings("ignore")

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert_model = AutoModel.from_pretrained('bert-base-uncased')
bert_model.to(torch.device('cuda')) #added


def got_loss(p,q,lamb):
    cos_distance = cost_matrix_batch_torch(p.transpose(2,1), q.transpose(2,1))
    cos_distance = torch.tensor(cos_distance).transpose(1,2)
    beta=0.1
    min_score = cos_distance.min()
    max_score = cos_distance.max()
    threshold = min_score + beta * (max_score - min_score)
    cos_dist = torch.nn.functional.relu(cos_distance - threshold)
    wd = IPOT_distance_torch_batch_uniform(cos_dist,p.size(0),p.size(1),q.size(1), 50)
    gwd = GW_distance_uniform(p.transpose(2,1), q.transpose(2,1))
    twd = lamb*torch.mean(gwd) + (1-lamb)*torch.mean(wd)
    return twd

def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])


def train_model(label, dataloaders, device, dataset_sizes, model,
                criterion, optimizer, scheduler, num_epochs=2):
    since = time.time()
    training_results = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 0.0
    train_step = 0 # for tensorboard
    leading_epoch = 0  # record best model epoch
    text_embeddings = np.load('embeddings.npy')#added
    text_embeddings = np.array(text_embeddings,dtype=np.double) #added
    
    """
    # model layer freezing stuff
    for param in model.parameters():
        param.requires_grad = False
    for param in list(model.module.feature_extractor.parameters())[-3:]:
        param.requires_grad = True
    for param in model.module.classifier.parameters():
        param.requires_grad = True
    """
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                scheduler.step()
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            running_balanced_acc_sum = 0.0
            # running_total = 0
            print(phase)
            # Iterate over data.
            for n_iter, batch in enumerate(dataloaders[phase]):
                
                inputs = batch["image"].to(device)
                labels = batch[label]
                bs = len(batch['image'])
                textual_embeddings = torch.cat(tuple([torch.tensor(text_embeddings).unsqueeze(0)]*bs)).cuda().double()
                labels = torch.from_numpy(np.asarray(labels)).to(device)
                #partition = torch.tensor(batch["partition"]).to(device)
                '''
                partition = batch["partition"]
                bert_model.to(device)
                encoding = tokenizer(partition, padding='max_length', truncation=True, max_length=256, return_tensors='pt')

                input_ids =  torch.tensor(encoding['input_ids'].squeeze()).to(device)
                attention_mask =  torch.tensor(encoding['attention_mask'].squeeze()).to(device)

                bert_output = bert_model(input_ids, attention_mask)    
                '''        
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    inputs = inputs.float()  # ADDED AS A FIX
                    outputs = model(inputs)

                    #l_got = got_loss(outputs[1], bert_output.last_hidden_state, lamb = 0.2)
                    l_got = got_loss(outputs[1], textual_embeddings, lamb = 0.2)


                
                    _, preds = torch.max(outputs[0], 1)
                    loss0 = criterion[0](outputs[0], labels)
                    loss = loss0  +  l_got
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    # statistics
                # tensorboard
                if phase == 'train':
                    writer.add_scalar('Loss/'+phase, loss.item(), train_step)
                    # writer.add_scalar('Loss/'+phase+'contrast_loss', loss1.item(), train_step)
                    writer.add_scalar('Accuracy/'+phase, (torch.sum(preds == labels.data)).item()/inputs.size(0), train_step)
                    writer.add_scalar('Balanced-Accuracy/'+phase, balanced_accuracy_score(labels.data.cpu(), preds.cpu()), train_step)
                    train_step += 1
                # -------------------------
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_balanced_acc_sum += balanced_accuracy_score(labels.data.cpu(), preds.cpu())*inputs.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            epoch_balanced_acc = running_balanced_acc_sum / dataset_sizes[phase]
            # print("Loss: {}/{}".format(running_loss, dataset_sizes[phase]))
            print("Accuracy: {}/{}".format(running_corrects,
                                           dataset_sizes[phase]))
            print('{} Loss: {:.4f} Acc: {:.4f} Balanced-Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_balanced_acc))
            # tensorboard 
            writer.add_scalar('lr/'+phase, scheduler.get_last_lr()[0], epoch)
            if phase == 'val':
                writer.add_scalar('Loss/'+phase, epoch_loss, epoch)
                writer.add_scalar('Accuracy/'+phase, epoch_acc, epoch)
                writer.add_scalar('Balanced-Accuracy/'+phase, epoch_balanced_acc, epoch)
            # ---------------------    
            training_results.append([phase, epoch, epoch_loss, epoch_acc.item(), epoch_balanced_acc])
            if epoch > 0:
                if phase == 'val' and epoch_acc > best_acc:
                    print("New leading accuracy: {}".format(epoch_acc))
                    best_acc = epoch_acc
                    leading_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
                # use balanced acc
                # if phase == 'val' and epoch_balanced_acc > best_acc:
                #     print("New leading balanced accuracy: {}".format(epoch_balanced_acc))
                #     best_acc = epoch_balanced_acc
                #     leading_epoch = epoch
                #     best_model_wts = copy.deepcopy(model.state_dict())      
                # if phase == 'val' and epoch_loss < best_loss:
                #     print("New leading accuracy: {}".format(epoch_acc))
                #     best_acc = epoch_acc
                #     best_loss = epoch_loss
                #     leading_epoch = epoch
                #     best_model_wts = copy.deepcopy(model.state_dict())
            elif phase == 'val':
                best_acc = epoch_acc
                # best_loss = epoch_loss
                # use balanced acc
                # best_acc = epoch_balanced_acc
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best model epoch:', leading_epoch)
    model.load_state_dict(best_model_wts)
    training_results = pd.DataFrame(training_results)
    training_results.columns = ["phase", "epoch", "loss", "accuracy", "balanced-accuracy"]
    return model, training_results


import logging
class MIMICDATASET(torch.utils.data.Dataset):
    def __init__(self, dataset_name, root_dir, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.dataset_name = dataset_name
        #logging.basicConfig(filename='skipped_images.log', level=logging.INFO)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print(f"Accessing index: {idx}")
        if idx >= len(self.df):
            raise IndexError(f"Index {idx} is out of bounds for DataFrame with shape {self.df.shape}.")

        img_name = os.path.join(self.root_dir, str(self.df.loc[self.df.index[idx], 'Path']))
        image = io.imread(img_name)

        if len(image.shape) < 3:
            image = skimage.color.gray2rgb(image)
            

        

        Path = self.df.loc[self.df.index[idx], 'Path']
        fitzpatrick = self.df.loc[self.df.index[idx], 'fitzpatrick']
        label = int(self.df.loc[self.df.index[idx], 'No Finding'])  # Convert to integer

        if self.transform:
            image = self.transform(image)

        sample = {
            'image': image,
            'Path': Path,
            'fitzpatrick': fitzpatrick,
            'No Finding': label
        }

        return sample
def custom_load(
        batch_size=32,
        num_workers=0,
        train_dir='',
        val_dir='',
        label='No Finding',
        dataset_name='chexpert',
        image_dir='/DATA2/',  # chexpert
):
    # Load the training and validation data
    val = pd.read_csv(val_dir)
    train = pd.read_csv(train_dir)

    # Ensure train[label] contains integer values (0 and 1)
    train[label] = train[label].astype(int)  # Convert to int

    # Compute class sample count (counts the occurrence of each label)
    class_sample_count = np.array(train[label].value_counts().sort_index())

    # Calculate weights for each class (1 / class sample count)
    weight = 1. / class_sample_count
    print("Weight for each class:", weight) 

    # Create the sample weights based on the labels in the train dataset
    samples_weight = np.array([weight[t] for t in train[label]])  # Now train[label] is integer
    samples_weight = torch.from_numpy(samples_weight)

    # Create a WeightedRandomSampler for training data
    sampler = WeightedRandomSampler(
        samples_weight.type('torch.DoubleTensor'),
        len(samples_weight),
        replacement=True
    )

    # Define dataset sizes
    dataset_sizes = {"train": train.shape[0], "val": val.shape[0]}

    # Initialize the training dataset with transformations
    transformed_train = MIMICDATASET(
        dataset_name=dataset_name,
        root_dir=image_dir,
        csv_file=train_dir,
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),  # ImageNet standards
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    )

    # Initialize the validation dataset with transformations
    transformed_test = MIMICDATASET(
        dataset_name=dataset_name,
        root_dir=image_dir,
        csv_file=val_dir,
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    )

    # Initialize the validation dataset
    transformed_test = MIMICDATASET(
        dataset_name=dataset_name,
        root_dir=image_dir,  # Set the image directory
        csv_file=val_dir,
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    )
  
    # Create DataLoader for training and validation sets
    dataloaders = {
        "train": DataLoader(
            transformed_train,
            batch_size=batch_size,
            #shuffle=True,
            sampler=sampler,
            num_workers=num_workers,
            drop_last=False,
            #collate_fn=custom_collate
        ),
        "val": DataLoader(
            transformed_test,
            batch_size=batch_size,
            shuffle=False,
            drop_last= False,
            num_workers=num_workers,
            #collate_fn=custom_collate
        )
    }

    return dataloaders, dataset_sizes
'''
def custom_load(
        batch_size=32,
        num_workers=0,
        train_dir='',
        val_dir='',
        label='No Finding',
        dataset_name='mimic',
        image_dir='/DATA2/DATA1/physionet.org/files/mimic-cxr-jpg/2.1.0/',
       # image_dir='/home/user/physionet.org/physionet.org/files/mimic-cxr-jpg/2.1.0/',
        #image_dir='/home/darakshan/Documents/physionet.org/files/mimic-cxr-jpg/2.1.0/',
):
    val = pd.read_csv(val_dir)
    train = pd.read_csv(train_dir)
    #print("train.index: ",train.index)
   # print("val.index: ",val.index)

    class_sample_count = np.array(train[label].value_counts().sort_index())
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in train[label]])
    samples_weight = torch.from_numpy(samples_weight)

    sampler = WeightedRandomSampler(
        samples_weight.type('torch.DoubleTensor'),
        len(samples_weight),
        replacement=True
    )

    dataset_sizes = {"train": train.shape[0], "val": val.shape[0]}

    # Initialize the training dataset
    transformed_train = MIMICDATASET(
        dataset_name=dataset_name,
        root_dir=image_dir,  # Set the image directory
        csv_file=train_dir,
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),  # Image net standards
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    )

    # Initialize the validation dataset
    transformed_test = MIMICDATASET(
        dataset_name=dataset_name,
        root_dir=image_dir,  # Set the image directory
        csv_file=val_dir,
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    )
  
    # Create DataLoader for training and validation sets
    dataloaders = {
        "train": DataLoader(
            transformed_train,
            batch_size=batch_size,
            #shuffle=True,
            sampler=sampler,
            num_workers=num_workers,
            drop_last=False,
            #collate_fn=custom_collate
        ),
        "val": DataLoader(
            transformed_test,
            batch_size=batch_size,
            shuffle=False,
            drop_last= False,
            num_workers=num_workers,
            #collate_fn=custom_collate
        )
    }

    return dataloaders, dataset_sizes
'''
# tensorboard writer
# writer = SummaryWriter()

if __name__ == '__main__':
    # In the custom_load() function, make sure to specify the path to the images
    print("\nPlease specify number of epochs and 'dev' mode or not... e.g. python train.py 10 full \n")
    n_epochs = int(sys.argv[1])
    dev_mode = sys.argv[2]
    dataset_name = sys.argv[3]
    model_name = sys.argv[4]
    domain=0
    print("CUDA is available: {} \n".format(torch.cuda.is_available()))
    print("Starting... \n")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if dev_mode == "dev":
        df = pd.read_csv('/home/user/physionet.org/physionet.org/xray_training_dataset_patch.csv').sample(100)  # Change the file name as needed
        #df = pd.read_csv('/home/darakshan/Documents/physionet.org/xray_training_dataset_patch_AP.csv').sample(20000)
    else:
        #df = pd.read_csv("/DATA1/physionet.org/xray_training_dataset_patch_AP.csv")  # Use the full dataset
       # df = pd.read_csv('/home/user/physionet.org/physionet.org/xray_training_dataset_patch.csv')
        #df = pd.read_csv('/home/darakshan/Documents/physionet.org/xray_training_dataset_patch.csv')
        #df = pd.read_csv('/home/server6000/DATA2/DATA1/physionet.org/xray_training_dataset_patch.csv')
       df =pd.read_csv("/DATA2/existing_images_only.csv")
             
       
    domain = ["random_holdout", "a12", "a34", "a56"][domain]
    print(f'DOMAIN: {domain}')
    
    for holdout_set in [domain]: 
        if holdout_set == "random_holdout":
            # Split data for training and testing
            train, test, y_train, y_test = train_test_split(
                df,
                df['No Finding'],  # Update this column based on your dataset
                test_size=0.2,
                random_state=12140420,
                stratify=df['No Finding']  # Ensure stratified sampling based on labels
            )
            print(f"Training set length: {len(train)}")
            print(f"Validation/Test set length: {len(test)}")
            
        elif holdout_set == "a12":
            train = df[(df['some_feature'] == 1) | (df['some_feature'] == 2)]  # Replace with actual feature names
            test = df[(df['some_feature'] != 1) & (df['some_feature'] != 2)]
        elif holdout_set == "a34":
            train = df[(df['some_feature'] == 3) | (df['some_feature'] == 4)]
            test = df[(df['some_feature'] != 3) & (df['some_feature'] != 4)]
        elif holdout_set == "a56":
            train = df[(df['some_feature'] == 5) | (df['some_feature'] == 6)]
            test = df[(df['some_feature'] != 5) & (df['some_feature'] != 6)]
        level = 'No Finding'
        

        
        train_path = "temp_train_{}.csv".format(model_name)
        test_path = "temp_test_{}.csv".format(model_name)
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)

        for indexer, label in enumerate([level]):
            # tensorboard
            writer = SummaryWriter(comment="logs_{}_{}_{}_{}.pth".format(model_name, n_epochs, label, holdout_set))
            print(label)
            weights = np.array(max(train[label].value_counts())/train[label].value_counts().sort_index())
            label_codes = sorted(list(train[label].unique()))
            dataloaders, dataset_sizes = custom_load(
                128,
                10,
                "{}".format(train_path),
                "{}".format(test_path),
                label = label,
                dataset_name=dataset_name)
            print(dataset_sizes)
            # ------------------

            model_ft = Network('vit', len(label_codes), pretrained=True)

            total_params = sum(p.numel() for p in model_ft.parameters())
            print('{} total parameters'.format(total_params))
            total_trainable_params = sum(
                p.numel() for p in model_ft.parameters() if p.requires_grad)
            print('{} total trainable parameters'.format(total_trainable_params))
            
             
            total_params = sum(p.numel() for p in model_ft.parameters())
            print('{} total parameters'.format(total_params))
            total_trainable_params = sum(
                p.numel() for p in model_ft.parameters() if p.requires_grad)
            print('{} total trainable parameters'.format(total_trainable_params))
            model_ft = model_ft.to(device)
            model_ft = nn.DataParallel(model_ft)
            class_weights = torch.FloatTensor(weights).cuda()
            # criterion = nn.NLLLoss()
            criterion = [nn.CrossEntropyLoss(), Supervised_Contrastive_Loss(0.1, device)]
            optimizer_ft = optim.Adam(model_ft.parameters(), 0.0001)
            # optimizer_ft = optim.AdamW(model_ft.parameters(),0.0001,weight_decay=0.05)
            exp_lr_scheduler = lr_scheduler.StepLR(
                optimizer_ft,
                step_size=2,
                gamma=0.9)
            # exp_lr_scheduler = lr_scheduler.StepLR(
            # optimizer_ft,
            # step_size=2,
            # gamma=0.9)

            print("\nTraining classifier for {}........ \n".format(label))
            print("....... processing ........ \n")
            model_ft, training_results = train_model(
                label,
                dataloaders, device,
                dataset_sizes, model_ft,
                criterion, optimizer_ft,
                exp_lr_scheduler, n_epochs)
            print("Training Complete")

            torch.save(model_ft.state_dict(), "model_GOT_path_{}_{}_{}_{}.pth".format(model_name, n_epochs, label, holdout_set))
            torch.save(model_ft, "model_GOT_path_{}_{}_{}_{}.pt".format(model_name, n_epochs, label, holdout_set))
            print("gold")
            training_results.to_csv("training_GOT_{}_{}_{}_{}.csv".format(model_name, n_epochs, label, holdout_set))

            model = model_ft.eval()
            loader = dataloaders["val"]
            prediction_list = []
            fitzpatrick_list = []
            Path_list = []
            labels_list = []
            p_list = []
            topk_p = []
            topk_n = []
            d1 = []
            d2 = []
            d3 = []
            p1 = []
            p2 = []
            p3 = []
            with torch.no_grad():
                running_corrects = 0
                running_balanced_acc_sum  = 0
                total = 0
                for i, batch in enumerate(dataloaders['val']):
                    inputs = batch["image"].to(device)
                    classes = batch[label].to(device)
                    fitzpatrick = batch["fitzpatrick"]  # skin type
                    Path = batch["Path"]
                    outputs = model(inputs.float())  # (batchsize, classes num)
                    probability = torch.nn.functional.softmax(outputs[0], dim=1)
                    ppp, preds = torch.topk(probability, 1) #topk values, topk indices
                    if label == "low":
                        _, preds5 = torch.topk(probability, 3)  # topk values, topk indices
                        # topk_p.append(np.exp(_.cpu()).tolist())
                        topk_p.append((_.cpu()).tolist())
                        topk_n.append(preds5.cpu().tolist())
                    running_corrects += torch.sum(preds.reshape(-1) == classes.data)
                    running_balanced_acc_sum += balanced_accuracy_score(classes.data.cpu(), preds.reshape(-1).cpu()) * inputs.shape[0]
                    p_list.append(ppp.cpu().tolist())
                    prediction_list.append(preds.cpu().tolist())
                    labels_list.append(classes.tolist())
                    fitzpatrick_list.append(fitzpatrick.tolist())
                    Path_list.append(Path)
                    total += inputs.shape[0]
                acc = float(running_corrects)/float(dataset_sizes['val'])
                balanced_acc = float(running_balanced_acc_sum)/float(dataset_sizes['val'])
            if label == "low":
                for j in topk_n: # each sample
                    for i in j:  # in k
                        d1.append(i[0])
                        d2.append(i[1])
                        d3.append(i[2])
                for j in topk_p:
                    for i in j:
                        # print(i)
                        p1.append(i[0])
                        p2.append(i[1])
                        p3.append(i[2])
                df_x=pd.DataFrame({
                                    "Path": flatten(Path_list),
                                    "label": flatten(labels_list),
                                    "fitzpatrick": flatten(fitzpatrick_list),
                                    "prediction_probability": flatten(p_list),
                                    "prediction": flatten(prediction_list),
                                    "d1": d1,
                                    "d2": d2,
                                    "d3": d3,
                                    "p1": p1,
                                    "p2": p2,
                                    "p3": p3})
            else:
                # print(len(flatten(Path_list)))
                # print(len(flatten(labels_list)))
                # print(len(flatten(fitzpatrick_list)))
                # print(len(flatten(p_list)))
                # print(len(flatten(prediction_list)))
                df_x=pd.DataFrame({
                                    "Path": flatten(Path_list),
                                    "label": flatten(labels_list),
                                    "fitzpatrick": flatten(fitzpatrick_list),
                                    "prediction_probability": flatten(p_list),
                                    "prediction": flatten(prediction_list)})
            df_x.to_csv("results_GOT_{}_{}_{}_{}.csv".format(model_name, n_epochs, label, holdout_set),
                            index=False)
            print("\n Accuracy: {}  Balanced Accuracy: {} \n".format(acc, balanced_acc))
        print("done")
        # writer.close()
