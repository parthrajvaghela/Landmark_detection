import os
import tensorflow as tf
import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler 
import matplotlib.pyplot as plt
import random
import torch.optim as optim
import torch.nn as nn
from PIL import Image

batch_size= 20 # how many samples the CNN sees and learn from at a time
valid_size = 0.2
# define training and test data directories
data_dir = 'landmark_images/'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
data_transform = transforms.Compose([transforms.RandomResizedCrop(256),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_data = datasets.ImageFolder(train_dir, transform=data_transform)
test_data = datasets.ImageFolder(test_dir, transform=data_transform)

# print out some data stats
print('Num training images: ', len(train_data))
print('Num test images: ', len(test_data))

num_train = len(train_data)
indices = list(range(num_train)) # indices of the enire dataset
np.random.shuffle(indices) 
split = int(np.floor(valid_size * num_train))  # take 20% of training set size
train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,sampler=train_sampler, num_workers=0)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,sampler=valid_sampler, num_workers=0)
test_loader =  torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=0)

# allow us to iterate data once batch at a time
loaders_scratch = {'train':train_loader ,'valid': valid_loader, 'test':test_loader }

#print(train_data.classes)
classes = [classes_name.split(".")[1] for classes_name in train_data.classes]
#print(classes[47])     

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))  # convert from Tensor image
    return img

fig = plt.figure(figsize=(20,2*8))
for index in range(12):
    ax = fig.add_subplot(4, 4, index+1, xticks=[], yticks=[])
    rand_img = random.randint(0, len(train_data))
    img = imshow(train_data[rand_img][0]) # unnormalize
    class_name = classes[train_data[rand_img][1]]
    ax.set_title(class_name)

use_cuda = tf.config.experimental.list_physical_devices('GPU')
for gpu in use_cuda: 
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.list_physical_devices('GPU')



criterion_scratch = nn.CrossEntropyLoss()
def get_optimizer_scratch(model):
    ## TODO: select and return an optimizer
    return optim.SGD(model.parameters(), lr=0.01)
    
import torch.nn.functional as F
class Net(nn.Module):
    ## TODO: choose an architecture, and complete the class
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 32 * 32 , 256)
        self.fc2 = nn.Linear(256, 50)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        ## Define forward behavior
        x = self.pool(F.relu(self.conv1(x))) # size 128
        x = self.pool(F.relu(self.conv2(x))) # size 64
        x = self.pool(F.relu(self.conv3(x))) # size 32
        x = x.view(-1, 64 * 32 * 32 )
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
model_scratch = Net()
if use_cuda:
    model_scratch.cuda()

def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        # set the module to training mode
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda: # load them in parallel
                data, target = data.cuda(), target.cuda() 
            ## TODO: find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data.item() - train_loss))
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward() # calculate gradient
            optimizer.step() # update wieghts
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data.item() - train_loss))

        ######################    
        # validate the model #
        ######################
        # set the model to evaluation mode
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## TODO: update average validation loss 
            output = model(data)
            loss = criterion(output, target)
            valid_loss =valid_loss + ((1 / (batch_idx + 1)) * (loss.data.item() - valid_loss))

            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch,train_loss,valid_loss))

        ## TODO: if the validation loss has decreased, save the model at the filepath stored in save_path
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss       
   
        
        
    return model

def custom_weight_init(m):
    ## TODO: implement a weight initialization strategy
    
    classname = m.__class__.__name__
    # for the two Linear layers
    if classname.find('Linear') != -1:
        num_inputs = m.in_features
        y= 1.0/np.sqrt(num_inputs) # general rule
        m.weight.data.uniform_(-y , y) 
        m.bias.data.fill_(0)

def default_weight_init(m):
    reset_parameters = getattr(m, 'reset_parameters', None)
    if callable(reset_parameters):
        m.reset_parameters()

model_scratch.apply(default_weight_init)

def test(loaders, model, criterion, use_cuda):

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    # set the module to evaluation mode
    model.eval()

    # for batch_idx, (data, target) in enumerate(loaders['test']):
    #     # move to GPU
    #     if use_cuda:
    #         data, target = data.cuda(), target.cuda()
    #     # forward pass: compute predicted outputs by passing inputs to the model
    #     output = model(data)
    #     # calculate the loss
    #     loss = criterion(output, target)
    #     # update average test loss 
    #     test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data.item() - test_loss))
    #     # convert output probabilities to predicted class
    #     pred = output.data.max(1, keepdim=True)[1]
    #     # compare predictions to true label
    #     correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
    #     total += data.size(0)
            
    # print('Test Loss: {:.6f}\n'.format(test_loss))

    # print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
    #     100. * correct / total, correct, total))
    
model_scratch.load_state_dict(torch.load('model_scratch.pt'))
test(loaders_scratch, model_scratch, criterion_scratch, use_cuda)

batch_size= 20 # how many samples the CNN sees and learn from at a time
valid_size=0.2

data_dir = 'landmark_images/'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

data_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.ToTensor()])
train_data = datasets.ImageFolder(train_dir, transform=data_transform)
test_data = datasets.ImageFolder(test_dir, transform=data_transform)

print('Num training images: ', len(train_data))
print('Num test images: ', len(test_data))
num_train = len(train_data)
indices = list(range(num_train)) # indices of the enire dataset
np.random.shuffle(indices) 
split = int(np.floor(valid_size * num_train))  # take 20% of training set size
train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,sampler=train_sampler, num_workers=0)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,sampler=valid_sampler, num_workers=0)
test_loader =  torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=0)


# allow us to iterate data once batch at a time
loaders_transfer = {'train':train_loader ,'valid': valid_loader, 'test':test_loader }


#print(train_data.classes)
classes = [classes_name.split(".")[1] for classes_name in train_data.classes]
#print(classes[49])     

  #----------------------------------------------------------------------------------------------------------------------------------

# ### (IMPLEMENTATION) Specify Loss Function and Optimizer
#
# Use the next code cell to specify a [loss function](http://pytorch.org/docs/stable/nn.html#loss-functions) and [optimizer](http://pytorch.org/docs/stable/optim.html).  Save the chosen loss function as `criterion_transfer`, and fill in the function `get_optimizer_transfer` below.


## TODO: select loss function
import torch.optim as optim
import torch.nn as nn

criterion_transfer = nn.CrossEntropyLoss()

def get_optimizer_transfer(model):
    ## TODO: select and return optimizer
    return optim.SGD(model.classifier.parameters(), lr=0.01)


    #----------------------------------------------------------------------------------------------------------------------------------

# ### (IMPLEMENTATION) Model Architecture
#
# Use transfer learning to create a CNN to classify images of landmarks.  Use the code cell below, and save your initialized model as the variable `model_transfer`.


## TODO: Specify model architecture
import torch.nn as nn
from torchvision import models

model_transfer = models.vgg16(weights=True)

#freezing features- weights
for param in model_transfer.features.parameters():
    param.require_grad =False

# replace last layer    
model_transfer.classifier[6] = nn.Linear( model_transfer.classifier[6].in_features , len(classes) )

print(model_transfer)

#-#-# Do NOT modify the code below this line. #-#-#
if use_cuda:
    model_transfer = model_transfer.cuda()

  #----------------------------------------------------------------------------------------------------------------------------------

# import signal

from contextlib import contextmanager
model_transfer.load_state_dict(torch.load('model_transfer.pt'))


    #----------------------------------------------------------------------------------------------------------------------------------

# ### (IMPLEMENTATION) Test the Model
#
# Try out your model on the test dataset of landmark images. Use the code cell below to calculate and print the test loss and accuracy.  Ensure that your test accuracy is greater than 60%.


# test(loaders_transfer, model_transfer, criterion_transfer, use_cuda)

  #----------------------------------------------------------------------------------------------------------------------------------

# ## Step 3: Write Your Landmark Prediction Algorithm
  #----------------------------------------------------------------------------------------------------------------------------------
# 
# Great job creating your CNN models! Now that you have put in all the hard work of creating accurate classifiers, let's define some functions to make it easy for others to use your classifiers.
# 
# ### (IMPLEMENTATION) Write Your Algorithm, Part 1
# 
# Implement the function `predict_landmarks`, which accepts a file path to an image and an integer k, and then predicts the **top k most likely landmarks**. You are **required** to use your transfer learned CNN from Step 2 to predict the landmarks.



## the class names can be accessed at the `classes` attribute
## of your dataset object (e.g., `train_dataset.classes`)
def predict_landmarks(img_path, k):
    ## TODO: return the names of the top k landmarks predicted by the transfer learned CNN
    image = Image.open(img_path)
    
    transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.ToTensor()])
                                    
    image= transform(image)
    image.unsqueeze_(0)
  
    if use_cuda:
        image = image.cuda()
        
    model_transfer.eval()  
                                    
    output = model_transfer(image)
    values, indices = output.topk(k)
    
    top_k_classes = []
    
    for i in indices[0].tolist():
        top_k_classes.append(classes[i])

    model_transfer.train()
    
    # return top_k_classes
    return ', '.join(top_k_classes)

import matplotlib.pyplot as plt

def suggest_locations(img_path):
    # get landmark predictions
    predicted_landmarks = predict_landmarks(img_path, 3)
    ## TODO: display image and display landmark predictions
    
    img = Image.open(img_path)
    plt.imshow(img)
    plt.show()
    print('Is this picture of the',predicted_landmarks[0])