
#Set up OpenAI API key

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# Load the CSV file
df = pd.read_csv('/home/user/physionet.org/physionet.org/xray_training_dataset_patch.csv')
df.columns = df.columns.str.strip()

# Define the specific labels
label_columns = [
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

# Create concatenated labels for each row
#concatenated_labels = []
#for _, row in df.iterrows():
#    row_labels = [col for col in label_columns if row[col] == 1]
#    concatenated_labels.append(" ".join(row_labels))
#print("concatenated labels: ", len(concatenated_labels))

# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("feature-extraction", model="dmis-lab/biobert-v1.1")
# Load model directly
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")



# Function to get embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Pooling by taking the mean of the last hidden state
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embeddings.numpy()

# Collect embeddings for each concatenated label string
res1 = []
for label_text in label_columns:
    embedding = get_embedding(label_text)
    res1.append(embedding)

# Convert embeddings to numpy array and save to .npy file
res1_array = np.array(res1)
np.save("embeddings.npy", res1_array)
print("size of embeddings", np.shape(res1_array))

print("Embeddings generated and saved as 'embeddings.npy'")
