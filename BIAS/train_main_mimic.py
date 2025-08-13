'''
Based on https://github.com/microsoft/vscode/issues/125993 
only use known skin type
train dissentangle network
python -u train_DisCo.py 20 full fitzpatrick DisCo
python -u train_DisCo.py 15 full ddi DisCo
'''
# Future imports for compatibility
from __future__ import print_function, division

# Standard Libraries
import os
import time
import copy
import random
import sys
import warnings

# Data Manipulation
import numpy as np
import pandas as pd
import skimage
# Image Processing and Computer Vision
from skimage import io, color

import cv2
from skimage import io
from PIL import Image

# PyTorch Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim import lr_scheduler
from torchvision import transforms, models
from torch.utils.tensorboard import SummaryWriter
#from torchsummary import summary

# Machine Learning Libraries
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import balanced_accuracy_score

# Progress Bar
from tqdm import tqdm

# Transformer Model
from transformers import AutoTokenizer, AutoModel

# Custom Modules
from Models.got_losses import Network, Confusion_Loss, Supervised_Contrastive_Loss, BinaryMatrixGenerator
from Masked_GOT_NewSinkhorn import cost_matrix_batch_torch, GW_distance_uniform, IPOT_distance_torch_batch_uniform

#torch.autograd.set_detect_anomaly(True) # GHC added
# Reproducibility

warnings.filterwarnings("ignore")

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert_model = AutoModel.from_pretrained('bert-base-uncased')
bert_model.to(torch.device('cuda'))

labels_top = [
    'No Finding',
    'Enlarged Cardiomediastinum',
    'Cardiomegaly',
    'Lung Lesion',
    'Lung Opacity',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other',
    'Fracture',
    'Support Devices'
]
def save_checkpoint(model, optimizer, epoch, label, model_name, n_epochs, holdout_set):
    checkpoint_dir = "checkpoints"  
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{model_name}_{n_epochs}_{label}_{holdout_set}_epoch_{epoch}.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

def calculate_probabilities(string_list):
    n = len(string_list)
    counts = {}
    probabilities = []
    
    # Count occurrences of each string
    for string in string_list:
        counts[string] = counts.get(string, 0) + 1

    # Calculate probabilities
    summation = 0
    for string in string_list:
        probability = counts[string] / n
        probabilities.append(probability)
        summation+=probability
    probabilities = [i/summation for i in probabilities]

    return probabilities

def got_loss(p,q, Mask, lamb):
    #print('NAN values in p:', torch.sum(torch.isnan(p)).item())
    cos_distance = cost_matrix_batch_torch(p.transpose(2,1), q.transpose(2,1)).transpose(1,2)
    
    beta=0.1
    min_score = cos_distance.min()
    max_score = cos_distance.max()
    threshold = min_score + beta * (max_score - min_score)
    cos_dist = torch.nn.functional.relu(cos_distance - threshold)
    wd, T = IPOT_distance_torch_batch_uniform(cos_dist, Mask, p.size(0),p.size(1),q.size(1),30)
    #print(f"A shape: {A.shape}, T shape: {T.shape}")
    gwd = GW_distance_uniform(p.transpose(2,1), q.transpose(2,1), Mask)
    twd = lamb * torch.mean(gwd) + (1 - lamb) * torch.mean(wd)
    return twd


def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])

def find_largest_parameter(model): # GHC Added
    largest_param = None
    largest_value = float('-inf')

    for name, param in model.named_parameters():
        if param.numel() > 0:  # Check if the parameter has elements
            param_max = torch.max(param)
            if param_max > largest_value:
                largest_param = name
                largest_value = param_max.item()

    return largest_param, largest_value

def train_model(label, dataloaders, device, dataset_sizes, model,
                criterion, optimizer, scheduler, num_epochs=20):
    #print('hyper-parameters alpha: {}  beta: {}'.format(alpha, beta))
    print("starting train_model")
    since = time.time()
    training_results = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 0.0
    train_step = 0  # for tensorboard
    leading_epoch = 0  # record best model epoch
    text_embeddings = np.load('embeddings.npy')  # replace the embedding file
    text_embeddings = np.array(text_embeddings, dtype=np.double)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            running_balanced_acc_sum = 0.0
            print(f'print either {phase} phase: ', phase)
            loop = tqdm(dataloaders[phase], leave=True, desc=f" {phase}-ing Epoch {epoch + 1}/{num_epochs}")
            print(f'number of batches in {phase} loop: ', len(loop))

            for n_iter, batch in enumerate(loop):
                if n_iter >= len(dataloaders[phase].dataset): 
                    print(f"Index {n_iter} out of bounds for {phase} dataset")
                    break
                bs = len(batch['fitzpatrick'])
                textual_embeddings = torch.cat(tuple([torch.tensor(text_embeddings).unsqueeze(0)] * bs)).cuda().double()
                inputs = batch["image"].to(device)
                #print(f"batch['No Finding']: {batch['No Finding']}")
                label_c = torch.tensor(batch['No Finding']).to(device) 
                label_c, label_t = batch[label], batch['fitzpatrick'] - 1  # label_condition, label_type
                label_c = torch.tensor(label_c).to(device)  # Convert to tensor and move to device
                label_t = torch.from_numpy(np.asarray(label_t)).to(device)  # Assuming label_t is already numeric

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    inputs = inputs.float()
                    output = model(inputs)
                    l_got = got_loss(output[-1], textual_embeddings, output[3], lamb=0.9)
                    _, preds = torch.max(output[0], 1)
                    loss0 = criterion[0](output[0], label_c)
                    loss1 = criterion[1](output[1], label_t) # branch 2 confusion 
                    loss2 = criterion[2](output[2], label_t) # branch 2 ce loss
                    #loss3 = torch.tensor(0)#criterion[3](output[3], label_c)  # supervised contrastive loss initial
                    #loss3 = criterion[3](output[2], label_c)  # supervised contrastive loss # changed to this
                    #loss = loss0 + loss1 * 0.5 + loss2 + 1 * l_got#+loss3*beta #initially
                    # L2 regularization
                    #l2_reg = sum(torch.norm(param) ** 2 for param in model.parameters())
                    #l2_reg_loss = lambda_reg * l2_reg
                    #loss = loss0 + loss1 * 0.7 + loss2 + 1 * l_got + loss3 * 0.7 # changed to this
                    loss = loss0 + loss1 * 0.7 + loss2 + 1 * l_got #+ l2_reg_loss

                    if phase == 'train':
                        loss.backward(retain_graph=True)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                if phase == 'train':
                    writer.add_scalar('Loss/' + phase, loss.item(), train_step)
                    writer.add_scalar('Loss/' + phase + 'loss0', loss0.item(), train_step)
                    writer.add_scalar('Loss/' + phase + 'loss1_conf', loss1.item(), train_step)
                    writer.add_scalar('Loss/' + phase + 'loss2', loss2.item(), train_step)
                    #writer.add_scalar('Loss/' + phase + 'contrast_loss', loss3.item(), train_step)
                    writer.add_scalar('Accuracy/' + phase, (torch.sum(preds == label_c.data)).item() / inputs.size(0), train_step)
                    writer.add_scalar('Balanced-Accuracy/' + phase, balanced_accuracy_score(label_c.data.cpu(), preds.cpu()), train_step)
                    train_step += 1

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == label_c.data)
                running_balanced_acc_sum += balanced_accuracy_score(label_c.data.cpu(), preds.cpu()) * inputs.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            epoch_balanced_acc = running_balanced_acc_sum / dataset_sizes[phase]

            print("Accuracy: {}/{}".format(running_corrects, dataset_sizes[phase]))
            print('{} Loss: {:.4f} Acc: {:.4f} Balanced-Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_balanced_acc))
            writer.add_scalar('lr/' + phase, scheduler.get_last_lr()[0], epoch)
            if phase == 'val':
                writer.add_scalar('Loss/' + phase, epoch_loss, epoch)
                writer.add_scalar('Accuracy/' + phase, epoch_acc, epoch)
                writer.add_scalar('Balanced-Accuracy/' + phase, epoch_balanced_acc, epoch)
            training_results.append([phase, epoch, epoch_loss, epoch_acc.item(), epoch_balanced_acc])
            # Check for best model
            if epoch > 0:
                if phase == 'val' and epoch_acc > best_acc:
                    print("New leading accuracy: {}".format(epoch_acc))
                    best_acc = epoch_acc
                    leading_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
            elif phase == 'val':
                best_acc = epoch_acc
        save_checkpoint(model_ft, optimizer_ft, epoch, label, model_name, n_epochs, holdout_set)
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

        img_name = os.path.join(self.root_dir, str(self.df.loc[self.df.index[idx], 'image_location']))
        image = io.imread(img_name)

        if len(image.shape) < 3:
            image = skimage.color.gray2rgb(image)
            

        

        image_location = self.df.loc[self.df.index[idx], 'image_location']
        fitzpatrick = self.df.loc[self.df.index[idx], 'fitzpatrick']
        label = int(self.df.loc[self.df.index[idx], 'No Finding'])  # Convert to integer

        if self.transform:
            image = self.transform(image)

        sample = {
            'image': image,
            'image_location': image_location,
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
        dataset_name='mimic',
        image_dir='/DATA2//DATA1/physionet.org/files/mimic-cxr-jpg/2.1.0/'
        #image_dir='/DATA2', #chexpert
       # image_dir='/home/user/physionet.org/physionet.org/files/mimic-cxr-jpg/2.1.0/',
        #image_dir='/home/darakshan/Documents/physionet.org/files/mimic-cxr-jpg/2.1.0/',
):
    val = pd.read_csv(val_dir)
    train = pd.read_csv(train_dir)
    #print("train.index: ",train.index)
   # print("val.index: ",val.index)

    class_sample_count = np.array(train[label].value_counts().sort_index())
    print("train[label]",train[label])  # Inspect the label values
       # Inspect the weight array

    weight = 1. / class_sample_count
    print("weight",weight) 
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


if __name__ == '__main__':
    # In the custom_load() function, make sure to specify the path to the images
    print("\nPlease specify number of epochs and 'dev' mode or not... e.g. python train.py 10 full \n")
    n_epochs = int(sys.argv[1])
    dev_mode = sys.argv[2]
    dataset_name = sys.argv[3]
    model_name = sys.argv[4]
    domain = 0  # in domain

    torch.manual_seed(200704)
    random.seed(200704)
    np.random.seed(200704)
    print("CUDA is available: {} \n".format(torch.cuda.is_available()))
    print("Starting... \n")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load the MIMIC dataset based on the specified mode
    if dev_mode == "dev":
        #df = pd.read_csv('/home/user/physionet.org/physionet.org/xray_training_dataset_patch.csv').sample(100)  # Change the file name as needed
       # df = pd.read_csv('/home/darakshan/Documents/physionet.org/xray_training_dataset_patch_AP.csv').sample(20000)
        #df=pd.read_csv("/DATA1/physionet.org/xray_training_dataset_patch_AP.csv").sample(20000)
        df=pd.read_csv("/DATA2/DATA1/physionet.org/xray_training_dataset_patch.csv").sample(20000)
    else:
        #df = pd.read_csv("/DATA1/physionet.org/xray_training_dataset_patch.csv")  # Use the full dataset
       # df = pd.read_csv('/home/user/physionet.org/physionet.org/xray_training_dataset_patch.csv')
        #df = pd.read_csv('/home/darakshan/Documents/physionet.org/xray_training_dataset_patch.csv')
        #df = pd.read_csv("/DATA1/physionet.org/xray_training_dataset_patch.csv")
        df=pd.read_csv("/DATA2/DATA1/physionet.org/xray_training_dataset_patch_AP.csv")
        #df=pd.read_csv("/DATA2/existing_images_only.csv")
       
    domain = ["random_holdout", "a12", "a34", "a56"][domain]
    #print("DOMAIN:", domain)
    
    # Update holdout set logic based on the MIMIC dataset structure
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

        level = "No Finding"  # 9-label

        train_path = "temp_train_{}.csv".format(model_name)
        test_path = "temp_test_{}.csv".format(model_name)
        train.to_csv(train_path, index=True)  # false initially 
        test.to_csv(test_path, index=True)

        for indexer, label in enumerate([level]):
            writer = SummaryWriter(comment="logs_{}_{}_{}_{}.pth".format(model_name, n_epochs, label, holdout_set))
            #print("Processing label:", label)

            # Calculate weights based on class distribution
            weights = np.array(max(train[label].value_counts()) / train[label].value_counts().sort_index())
            label_codes = sorted(list(train[label].unique()))
               
            # Load custom dataset
            dataloaders, dataset_sizes = custom_load(
                128,
                0,
                "{}".format(train_path),
                "{}".format(test_path),
                label=label,
                dataset_name=dataset_name
            )
            print("Dataset sizes from custom_load:", dataset_sizes)
            print("length of the label codes", label_codes)
            model_ft = Network('sparse', [len(label_codes), 6], pretrained=True)

            total_params = sum(p.numel() for p in model_ft.feature_extractor.parameters())
            print('{} total parameters'.format(total_params))

            i = 0
            for p in model_ft.feature_extractor.parameters():
                p.requires_grad = i >= 50
                i += 1

            total_trainable_params = sum(p.numel() for p in model_ft.feature_extractor.parameters() if p.requires_grad)
            print('{} total trainable parameters'.format(total_trainable_params))
           # summary(model_ft)

            model_ft = model_ft.to(device)
            model_ft = nn.DataParallel(model_ft)
            class_weights = torch.FloatTensor(weights).cuda()

            criterion = [
                nn.CrossEntropyLoss(),
                Confusion_Loss(),
                nn.CrossEntropyLoss(),
                Supervised_Contrastive_Loss(0.1, device)
            ]
            optimizer_ft = optim.Adam(list(model_ft.parameters()), 0.0001)
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=2, gamma=0.8)
            print("\nTraining classifier for {}........ \n".format(label))
            model_ft, training_results = train_model( #yha se 
                label,
                dataloaders,
                device,
                dataset_sizes,
                model_ft,
                criterion,
                optimizer_ft,
                exp_lr_scheduler,
                n_epochs
            )
            #save_checkpoint(model_ft, optimizer_ft, epoch, label, model_name, n_epochs, holdout_set)
            #torch.save(model_ft.state_dict(), "model_path_{}_{}_{}_{}.pth".format(model_name, n_epochs, label, holdout_set))
            training_results.to_csv("training_{}_{}_{}_{}.csv".format(model_name, n_epochs, label, holdout_set))
            print("Training Complete")

            #torch.save(model_ft.state_dict(), "model_path_{}_{}_{}_{}.pth".format(model_name, n_epochs, label, holdout_set))
            #training_results.to_csv("training_{}_{}_{}_{}.csv".format(model_name, n_epochs, label, holdout_set))

            # Evaluation phase
            model = model_ft.eval()
            loader = dataloaders["val"]
            prediction_list = []
            fitzpatrick_list = []
            image_location_list = []
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
                    image_location = batch["image_location"]
                    outputs = model(inputs.float())  # (batchsize, classes num)
                    probability = torch.nn.functional.softmax(outputs[0], dim=1)
                    ppp, preds = torch.topk(probability, 1) #topk values, topk indices
                    '''
                    if label == "low":
                        _, preds5 = torch.topk(probability, 3)  # topk values, topk indices
                        # topk_p.append(np.exp(_.cpu()).tolist())
                        topk_p.append((_.cpu()).tolist())
                        topk_n.append(preds5.cpu().tolist())
                        '''
                    running_corrects += torch.sum(preds.reshape(-1) == classes.data)
                    running_balanced_acc_sum += balanced_accuracy_score(classes.data.cpu(), preds.reshape(-1).cpu()) * inputs.shape[0]
                    p_list.append(ppp.cpu().tolist())
                    prediction_list.append(preds.cpu().tolist())
                    labels_list.append(classes.tolist())
                    fitzpatrick_list.append(fitzpatrick.tolist())
                    image_location_list.append(image_location)
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
                                    "image_location": flatten(image_location_list),
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
                # print(len(flatten(image_location_list)))
                # print(len(flatten(labels_list)))
                # print(len(flatten(fitzpatrick_list)))
                # print(len(flatten(p_list)))
                # print(len(flatten(prediction_list)))
                df_x=pd.DataFrame({
                                    "image_location": flatten(image_location_list),
                                    "label": flatten(labels_list),
                                    "fitzpatrick": flatten(fitzpatrick_list),
                                    "prediction_probability": flatten(p_list),
                                    "prediction": flatten(prediction_list)})
            df_x.to_csv("results_{}_{}_{}_{}.csv".format(model_name, n_epochs, label, holdout_set),
                            index=False)
            print("\n Accuracy: {}  Balanced Accuracy: {} \n".format(acc, balanced_acc))
        print("done")
        # writer.close()
'''
                for i, batch in enumerate(loader):
                    print(f"Validating batch index: {i}, total validation samples: {len(loader.dataset)}")
                    if i >= len(loader.dataset):
                        print(f"Index {i} out of bounds for validation dataset")
                        break

                    inputs = batch["image"].to(device)
                    classes = batch[label].to(device)
                    classes = torch.tensor(batch[label]).to(device)
                    outputs = model(inputs.float())
                    preds = torch.argmax(outputs[0], dim=1)

                    running_corrects += torch.sum(preds == classes.data)
                    prediction_list.extend(preds.cpu().tolist())
                    labels_list.extend(classes.tolist())
                    total += inputs.shape[0]

                    print(f"Validation batch {i}: inputs shape {inputs.shape}, classes shape {classes.shape}, preds shape {preds.shape}")

                acc = float(running_corrects) / total
                print(f'Accuracy: {acc:.4f}')
                results_df = pd.DataFrame({"True Label": labels_list,"Prediction": prediction_list})
                csv_filename = "evaluation_results.csv"
                results_df.to_csv(csv_filename, index=False)
                print(f"Results saved to {csv_filename}")
'''
