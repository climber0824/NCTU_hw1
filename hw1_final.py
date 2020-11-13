import pandas as pd 
import torch
import torchvision
import torch.nn  as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from skimage import io
from torchvision import transforms, utils, datasets, models
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image 
import numpy as np
from pathlib import Path


import time
import json
import copy
import os
import glob

#data_dir
train_dir = './training_data'
test_dir = './testing_data'
label_dir = 'training_labels.csv'

label_df = pd.read_csv('./DeepCars_dataset/names.csv', names=["label"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),        
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([       
        transforms.CenterCrop(224),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),                                  
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ]),
}


class MyDataset(Dataset):
    def __init__(self, ippnl, labels, labels_dict, transform):
        self.imgs_path = ippnl
        self.label_list = labels
        self.labels_dict = labels_dict
        self.transform = transform

    def __getitem__(self, index):
        imgpath = self.imgs_path[index]
        label = self.label_list[index]
        label = self.labels_dict[label]
        img = Image.open(imgpath).convert('RGB')
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs_path)


df = pd.read_csv('./training_labels.csv')
img_path_list = []
labels = []
labels_dict = {}
count = 0
for w in df['id']:
    zeros = 6 - len(str(w))
    zeros_tmp = []
    for i in range(zeros):
        zeros_tmp.append('0')
    zeros_tmp = ''.join(zeros_tmp)
    w = zeros_tmp + str(w)
    img_path_list.append(f"./training_data/training_data/{w}.jpg")

count = 0
for w in df['label']:
    labels.append(w)
    if labels_dict.get(w) is None:
        labels_dict[w] = count
        count+=1


batch_size=32
train_set = trainingSet(img_path_list, labels,labels_dict, data_transforms["train"])

valid_size  = int(0.1 * len(train_set))
train_size = len(train_set) - valid_size
dataset_sizes = {'train': train_size, 'valid': valid_size}


train_dataset, valid_dataset = torch.utils.data.random_split(train_set, [train_size, valid_size])


dataloaders = {'train': DataLoader(train_dataset, batch_size = batch_size, shuffle = True),
              'valid': DataLoader(valid_dataset, batch_size = batch_size, shuffle = False)}

images, labels = next(iter(dataloaders['train']))

#NNnetwork
model = models.vgg19(pretrained=True)
num_in_features = 25088

for param in model.parameters():
  param.require_grad = False

def build_classifier(num_in_features, hidden_layers, num_out_features):
   
    classifier = nn.Sequential()
    if hidden_layers == None:
      
        classifier.add_module('fc0', nn.Linear(num_in_features, 196))
        
    else:
      
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        classifier.add_module('fc0', nn.Linear(num_in_features, hidden_layers[0]))
        classifier.add_module('relu0', nn.ReLU())
        classifier.add_module('drop0', nn.Dropout(.6))
        
        
        for i, (h1, h2) in enumerate(layer_sizes):
            classifier.add_module('fc'+str(i+1), nn.Linear(h1, h2))
            classifier.add_module('relu'+str(i+1), nn.ReLU())
            classifier.add_module('drop'+str(i+1), nn.Dropout(.5))

        classifier.add_module('output', nn.Linear(hidden_layers[-1], num_out_features))
        
    return classifier


hidden_layers = None
classifier = build_classifier(num_in_features, hidden_layers, 196)


model.classifier = classifier
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.0001)
sched = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)



def train_model(model, criterion, optimizer, sched, num_epochs=50, device='cuda'):
    start = time.time()
    train_results = []
    valid_results = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':            
              model.train() 
            else:
              model.eval()   

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

        
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    
                    if phase == 'train':
                        
                        loss.backward()
                        
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]


            if(phase == 'train'):
              train_results.append([epoch_loss,epoch_acc])
            if(phase == 'valid'):
              
              valid_results.append([epoch_loss,epoch_acc])


            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())       
                
                model_save_name = "hw1.pt"
                path = F"/home/kenchang/Documents/NCTU/hw1/{model_save_name}"
                torch.save(model.state_dict(), path)        

        print()

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    
    model.load_state_dict(best_model_wts)
    
    return model,train_results,valid_results

epochs = 50
model.to(device)
model,train_results,valid_results = train_model(model, criterion, optimizer, sched, epochs)




with torch.no_grad():
  model.eval()
  dataset = datasets.ImageFolder(test_dir, transform=data_transforms['test'])
  testloader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                             shuffle=False, num_workers=2)
 
  image_names = []
  pred = []
  for index in testloader.dataset.imgs:
    
    image_names.append(Path(index[0]).stem)

  results = []
  file_names = []
  predicted_car = []
  predicted_class =[]

  for inputs, labels in testloader:
    inputs = inputs.to(device)
    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)
    
    for i in range(len(inputs)):
        file_names.append(image_names[i])
        predicted_car.append(int(predicted[i] + 1))
results.append((file_names, predicted_car))        


tempt = []
tempt_2 = []
for w in os.listdir('./HW1_dataset_test/testing_data'):
    if 'C-V' in w :
        tempt.append('Ram C/V Cargo Van Minivan 2012')
    else :
        tempt.append(w)
for w in os.listdir('./HW1_dataset_test/training_data'):
    if 'C-V' in w :
        tempt_2.append('Ram C/V Cargo Van Minivan 2012')
    else :
        tempt_2.append(w)
print(tempt[4])
print(tempt_2[4])


print("Predictions on Test Set:")

df = pd.DataFrame({'Id': image_names, 'label': results[0][1]})
label_df = pd.read_csv(label_dir, names=["label"])
tempt_results = []
for i in range(len(results[0][1])):
    tempt_results.append(ltempt[results[0][1][i]-1])
df = pd.DataFrame({'Id': image_names, 'label': tempt_results})
pd.set_option('display.max_colwidth', None)
df.to_csv('/home/kenchang/Documents/NCTU/hw1/predictions1.csv')




