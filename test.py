import cv2
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import string
from torch.utils.data import random_split, DataLoader, Dataset
import torchvision
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
sns.set_style('whitegrid')
from sklearn.metrics import confusion_matrix , classification_report
from torch.utils.data import random_split, DataLoader
from PIL import Image


label_map = {char: index for index, char in enumerate(string.ascii_uppercase, start=0)}
label_map['del'] = 26
label_map['space'] = 27
label_map['nothing'] = 28
# print(label_map)

criterion = nn.CrossEntropyLoss()

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        modules = list(resnet50(weights=ResNet50_Weights.DEFAULT).children())[:-1]
        
        self.resnet50_feature_extractor = nn.Sequential(*modules)
        for param in self.resnet50_feature_extractor.parameters():
            param.requires_grad = True
        # Add a fully-connected layer for CIFAR-10 classification
        
        self.fc_layer = nn.Linear(2048, 29)


    def forward(self, images, labels):
        loss = None
        x = self.resnet50_feature_extractor(images)
        x = x.view(x.shape[0], -1)
        x = self.fc_layer(x)
        loss = criterion(x, labels)
        return loss

    def predict(self, images):
        labels_pred = None
        x = self.resnet50_feature_extractor(images)
        x = x.view(x.shape[0], -1)
        x = self.fc_layer(x)
        _, labels_pred = torch.max(x, dim=1)
        return labels_pred


model = Model()

model.load_state_dict(torch.load('/Users/sowmiyanarayanselvam/Documents/AirCloud/Rutgers/Spring24/ML2/Project/ML/model.pth', map_location=torch.device('cpu')))

def eval_acc(model, data_loader):

  with torch.no_grad():
    model.eval()

    total = 0
    correct = 0
    val_loss = 0
    for batch, (images, labels) in enumerate(data_loader):

        labels_pred = model.predict(images)
        label_tensor = torch.tensor([label_map[char] for char in labels])
        label_tensor = label_tensor

        total += len(label_tensor)
        
        correct += (labels_pred == label_tensor).sum().item()
         

    acc = 100 * correct / total
    
  return acc


test_df = pd.read_pickle('/Users/sowmiyanarayanselvam/Documents/AirCloud/Rutgers/Spring24/ML2/Project/ML/test.pkl')
class CustomDataset(Dataset):
    def __init__(self, data_df, transform=None):
        self.data_df = data_df
        self.transform = transform

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        img_path = self.data_df.iloc[idx, 0]  # Assuming the first column contains file paths
        label = self.data_df.iloc[idx, 1]  # Assuming the second column contains labels
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Define transformations
transform = ResNet50_Weights.DEFAULT.transforms()
test_dataset = CustomDataset(test_df, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=50, shuffle=True, pin_memory=True)



test_acc_finetune = eval_acc(model, test_loader)
print('Test Accuracy:', test_acc_finetune)