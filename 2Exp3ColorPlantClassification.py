#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from glob import glob

from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from PIL import Image

# see all charts inline in the notebook and allow dynamic graph updates
import pylab as pl
from IPython import display


import torch
import torch.nn as nn
import torch.utils as utils
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms

from torch.utils.tensorboard import SummaryWriter


# In[2]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[3]:


#pwd   =  Parent working directory
experiment ='Exp3'
modality = 'Color'      #modalities = ['Boson', '577nm', '732nm', '970nm','692nm','polar','Color']
pattern   = '*'+modality+'*.jpg'
# pattern   = '*'+modality+'*.tiff'
start_dir = '/home/pnn/experiments/'+experiment+'/'


# In[4]:


#finding files in subdirectories: 

files = []
for dir,_,_ in os.walk(start_dir):
    files.extend(glob(os.path.join(dir,pattern)))
    #if len(files)==10:  # <--- defining number of images limiter
    #    break

files=sorted(files)


# In[5]:


# for the time being till i know how to read this :
 
color_pos=(
            (1091, 1103),(1501, 1093),(1875, 1093),(2227, 1109),(2585, 1113),(2909, 1089),
            (1107, 1459),(1489, 1451),(1909, 1447),(2255, 1465),(2587, 1457),(2915, 1467),
            (1109, 1759),(1529, 1771),(1915, 1767),(2259, 1795),(2601, 1791),(2921, 1807),
            (1107, 2059),(1533, 2061),(1917, 2077),(2289, 2073),(2581, 2039),(2945, 2067),
            (1103, 2343),(1551, 2315),(1957, 2359),(2299, 2341),(2599, 2327),(2957, 2341))


# In[6]:


def get_image(plant_position,img_len = 255,time_idx = 0):
        left = plant_position[0] - img_len // 2
        right = plant_position[0] + img_len // 2
        top = plant_position[1] - img_len // 2
        bottom = plant_position[1] + img_len // 2
        
        image_path = files[time_idx]

        image = Image.open(image_path)
        image = image.crop((left, top, right, bottom))

        to_tensor = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        return to_tensor(image).float()


# In[7]:


# Original labels
'''Labels = (
        3, 11, 6, 5,  4,
        3, 11, 6, 5,  4, 
        3,  4, 6, 5, 11,  
        3,  4, 5, 6, 11, 
        3,  4, 5, 6, 11, 
        3,  4, 5, 6, 11
    )'''

Labels = (
        0,  4, 3, 2,  1,
        0,  4, 3, 2,  1, 
        0,  1, 3, 2,  4,  
        0,  1, 2, 3,  4, 
        0,  1, 2, 3,  4, 
        0,  1, 2, 3,  4
    )


# In[8]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # (in_channels, out_channels, kernel_size)
        self.pool = nn.MaxPool2d(2, 2)  # was 2X2 
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 60 * 60, 3600)
        self.fc2 = nn.Linear(3600, 84)
        self.fc3 = nn.Linear(84, 5) # The last number of nodes should match the number of classes, number of tomato gynotypes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 60 * 60) # linear representation of x (reshape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net().to(device)


# In[9]:


criterion = nn.CrossEntropyLoss()

lr = 0.001
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.8)


# # NOW ALL COLOR IMAGES FROM THE FOLDERS OF EXP3

# In[10]:


# All Plant Images
PlantImages = []
for j,image_path in enumerate(files):
    # Getting All Plants Images
    for i,label in enumerate(Labels):
        PlantImages.append(get_image(color_pos[i],time_idx = j))


# In[11]:


# Genom Labels for all plant images
BLabels = []
for i in range(len(PlantImages)):
    BLabels.append(Labels[i%len(Labels)])


# In[12]:


PlantIdxes = [idx for idx in range(len(PlantImages))]


# In[13]:


x_data = torch.stack(PlantImages)
y_data = torch.Tensor(BLabels)  #   <---- Labels Tensor

dataset = utils.data.TensorDataset(x_data, y_data)
dataloader = DataLoader(dataset)


# In[14]:


plants = [plant for plant in range(len(Labels))]


# In[15]:


print(plants)


# In[16]:


AllFramePlants = []
for i in range(len(PlantImages)):
    AllFramePlants.append(plants[i%len(Labels)])

print(AllFramePlants)


# In[17]:


test_plants = [1, 2, 7, 29]
BPlants =[]
for plant in AllFramePlants:
    if plant in test_plants:
        BPlants.append(1)
    else:
        BPlants.append(0)


# In[18]:


#for i,allp in enumerate(AllFramePlants):
#    print(PlantIdxes[i],allp,BPlants[i],BLabels[i])


# In[19]:


test_idx = []
train_idx = []
for idx,plant in enumerate(PlantIdxes):
    test_num = plant*BPlants[idx]
    if test_num !=0:
        test_idx.append(idx)
    else:
        train_idx.append(idx)
            

print('test_idx  size = ',len(test_idx),'train_idx size = ',len(train_idx))
print('test_idx = ')
print(test_idx)


# In[20]:


BATCH_SIZE = 4
# CLASSES = (3, 4, 5, 6, 11)
CLASSES = (0, 1, 2, 3, 4)


# In[21]:


train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(train_idx))
test_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(test_idx))


# In[22]:


print('Train Loader Weight Size:', iter(train_loader).next()[0].size())
print('Train Loader Bias   Size:', iter(train_loader).next()[1].size())

print('Train Size:', len(train_idx))
print('Test Size:', len(test_idx))


# In[23]:


Tot_Loss = []
Accuracy = []
sizeTrain = len(train_loader)
MaxEpochs = 30

for epoch in range(MaxEpochs):  # loop over the dataset multiple times
    
    epoch_loss = 0
    running_loss = 0.0
    
    # Dynamic Learning Rate
    # adjust_learning_rate(optimizer, epoch)
    
    print('epoch = ', epoch, end = ' ')
    for i, data in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # print('inputs.shape = ',inputs.shape )
        labels = torch.Tensor.long(labels).to(device)
        inputs = inputs.to(device)
        # print('labels  = ',labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        # print(' LOSS   = ',loss)
        loss.backward()
        optimizer.step()

        
        # print statistics
        running_loss += loss.item()
        
        if i % 100 == 0:
            print('.', end ='')

    epoch_loss = running_loss/sizeTrain
    Tot_Loss.append(epoch_loss)
    
    # Computing Accuracy on Test Data:
    
    correct = 0
    total = 0
    #print('test_loader len = ',len(test_loader))
    with torch.no_grad():
        # TODO: DeBug - why IndexError: index 300 is out of bounds for dimension 0 with size 300
        # the bug is in the line below
        for index, val_data in enumerate(test_loader):  # index runs up to 8 and jumps to 300 thereafter
            #print('index  = ', index, end='  ')
            val_images, val_labels = val_data
            val_labels = torch.Tensor.long(val_labels).to(device)
            val_images = val_images.to(device)
            #print('val_labels  ', val_labels, end='  ')
            val_outputs = net(val_images)
            stam, predicted = torch.max(val_outputs.data, 1)
            #print('softmax result =',stam)
            #print('PlantIdxes   = ',val_idx, end='  ')
            #print(' predicted =',predicted)
            total += val_labels.size(0)
            temp = (predicted == val_labels)
            correct +=  1*temp[0].item()+1*temp[1].item()+1*temp[2].item()+1*temp[3].item()
            
    Accuracy.append(correct / total)
    
    # Dynamic plotting
    pl.clf()
    pl.plot(Tot_Loss,'.r', label='Loss curve')
    pl.plot(Accuracy,'.b', label='Accuracy')
    pl.xlabel('Epochs (100ds)')
    pl.ylabel('Loss')
    pl.ylim([0.0, 1.0])
    pl.xlim([-2.0, MaxEpochs+5.0])
    pl.legend(loc="upper right")
    display.clear_output(wait=False)
    display.display(pl.gcf())
        
print('Finished Training')


# In[24]:


print('Accuracy type is: ',type(Accuracy[0]))
print('correct type is: ',type(correct))
print('total type is: ',type(total))


# In[25]:


print(correct[1].item())


# In[ ]:


print(val_labels)


# In[ ]:


print(predicted == val_labels)


# In[ ]:


aaa= (predicted == val_labels)


# In[ ]:


aaa


# In[ ]:


aaa[0].item()*aaa[1].item()*aaa[2].item()*aaa[3].item()*1


# In[ ]:




