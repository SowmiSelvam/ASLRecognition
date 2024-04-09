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

#Warnings
import warnings
warnings.filterwarnings('ignore')


#Warnings
import warnings
warnings.filterwarnings('ignore')

print('python: {}, torch: {}, torchvision: {}'.format(sys.version, torch.__version__, torchvision.__version__))

print(torch.cuda.is_available())

import os
data='/dataset/asl_alphabet_train/asl_alphabet_train'

df = pd.read_pickle('/common/home/ss4370/Documents/ML/data.pkl')

test_df = pd.read_pickle('/common/home/ss4370/Documents/ML/test.pkl')

train_set, val_set = random_split(df, [0.8, 0.2], generator=torch.Generator().manual_seed(42))

train_df = df.iloc[train_set.indices]
val_df = df.iloc[val_set.indices]

label_map = {char: index for index, char in enumerate(string.ascii_uppercase, start=0)}
label_map['del'] = 26
label_map['space'] = 27
label_map['nothing'] = 28
print(label_map)

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

# Assuming df contains image file paths and labels
train_dataset = CustomDataset(train_df, transform=transform)
valid_dataset = CustomDataset(val_df, transform=transform)
test_dataset = CustomDataset(test_df, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True, pin_memory=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=50, shuffle=True, pin_memory=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=50, shuffle=True, pin_memory=True, num_workers=4)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        modules = list(resnet50(weights=ResNet50_Weights.DEFAULT).children())[:-1]
        
        self.resnet50_feature_extractor = nn.Sequential(*modules)
        for param in self.resnet50_feature_extractor.parameters():
            param.requires_grad = True
        
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

model = Model().cuda()
# END OF YOUR CODE

torch.manual_seed(0)

num_epochs = 5

#########################################################################
# TODO: choose an optimizer                                             #
#########################################################################
# Replace "pass" statement with your code
optimizer = None
optimizer = optim.Adamax(model.parameters(),lr=0.0001, weight_decay=1e-4)
# END OF YOUR CODE

loss_history = []

criterion = nn.CrossEntropyLoss()

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)


# Example of how you can use accuracy as a metric
def eval_acc(model, data_loader):

  with torch.no_grad():
    model.eval()

    total = 0
    correct = 0
    val_loss = 0
    for batch, (images, labels) in enumerate(data_loader):
        
        images = images.cuda()

        labels_pred = model.predict(images)
        label_tensor = torch.tensor([label_map[char] for char in labels])
        label_tensor = label_tensor.cuda()

        total += len(label_tensor)
        
        correct += (labels_pred == label_tensor).sum().item()
         

    acc = 100 * correct / total
    
  return acc

loss_history=[]
val_loss_history=[]
val_acc_history=[]
num_epochs = 5
for epoch in range(num_epochs):
    val_acc = eval_acc(model, valid_loader)
    val_acc_history.append(val_acc)

    model.train()
    running_loss = 0.0
    
    for batch, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        label_tensor = torch.tensor([label_map[char] for char in labels])
        label_tensor = label_tensor.cuda()
        
        optimizer.zero_grad()
        loss = model(images, label_tensor)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            loss_history.append(loss.item())
            if batch == 0:
                print('Train Epoch: {:3} \t Loss: {:F} \t Val Acc: {:F}'.format(
                  epoch, loss.item(), val_acc))
                

    epoch_loss = running_loss / len(train_loader.dataset)


loss_df = pd.DataFrame(loss_history)
loss_df.to_pickle('/common/home/ss4370/Documents/ML/loss.pkl')

val_acc_df = pd.DataFrame(val_acc_history)
val_acc_df.to_pickle('/common/home/ss4370/Documents/ML/val_acc_loss.pkl')

torch.save(model.state_dict(), '/common/home/ss4370/Documents/ML/model.pth')



with torch.no_grad():
    plt.plot(loss_history, 'o')
    plt.xlabel('Iteration number')
    plt.ylabel('Loss value')
    plt.show()

