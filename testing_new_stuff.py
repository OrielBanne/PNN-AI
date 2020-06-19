import os
from glob import glob
from typing import Any

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_image(plant_position, img_len=255, time_idx=0):
    left = plant_position[0] - img_len // 2
    right = plant_position[0] + img_len // 2
    top = plant_position[1] - img_len // 2
    bottom = plant_position[1] + img_len // 2

    image_path = files[time_idx]

    image = Image.open(image_path)
    image = image.crop((left, top, right, bottom))

    to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    return to_tensor(image).float()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # (in_channels, out_channels, kernel_size)
        self.pool = nn.MaxPool2d(2, 2)  # was 2X2
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 60 * 60, 3600)
        self.fc2 = nn.Linear(3600, 84)
        self.fc3 = nn.Linear(84,
                             5)  # The last number of nodes should match the number of classes, number of tomato gynotypes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 60 * 60)  # linear representation of x (reshape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# pwd   =  Parent working directory
experiment = 'Exp3'
modality = 'Color'  # modalities = ['Boson', '577nm', '732nm', '970nm','692nm','polar','Color']
pattern = '*' + modality + '*.jpg'
# pattern   = '*'+modality+'*.tiff'
start_dir = '/home/pnn/experiments/' + experiment + '/'

# finding files in subdirectories:

files_full = []
for dir, _, _ in os.walk(start_dir):
    files_full.extend(glob(os.path.join(dir, pattern)))

files = files_full[0:10]

# arrange files by date
files.sort()
print(files[0:5])

# for the time being till i know how to read this :

color_pos = (
    (1091, 1103), (1501, 1093), (1875, 1093), (2227, 1109), (2585, 1113), (2909, 1089),
    (1107, 1459), (1489, 1451), (1909, 1447), (2255, 1465), (2587, 1457), (2915, 1467),
    (1109, 1759), (1529, 1771), (1915, 1767), (2259, 1795), (2601, 1791), (2921, 1807),
    (1107, 2059), (1533, 2061), (1917, 2077), (2289, 2073), (2581, 2039), (2945, 2067),
    (1103, 2343), (1551, 2315), (1957, 2359), (2299, 2341), (2599, 2327), (2957, 2341))

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
    0, 4, 3, 2, 1,
    0, 4, 3, 2, 1,
    0, 1, 3, 2, 4,
    0, 1, 2, 3, 4,
    0, 1, 2, 3, 4,
    0, 1, 2, 3, 4
)

net = Net()
criterion = nn.CrossEntropyLoss()
lr = 0.001
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.8)

PlantImages = []
for j, image_path in enumerate(files):
    # Getting All Plants Images
    for i, label in enumerate(Labels):
        PlantImages.append(get_image(color_pos[i], time_idx=j))

# Here I must make labels as large as the dataset
BLabels = []
for i in range(len(PlantImages)):
    BLabels.append(Labels[i % len(Labels)])

plants = [plant + 1 for plant in range(len(Labels))]

BPlants = []
for i in range(len(PlantImages)):
    BPlants.append(plants[i % len(Labels)])

x_data = torch.stack(PlantImages)
y_data = torch.Tensor(BLabels)  # <---- this is the line that creates the labels as a tensor

dataset = utils.data.TensorDataset(x_data, y_data)
dataloader = DataLoader(dataset)

test_plants = [1, 3, 8, 30]
for idx, plant in enumerate(BPlants):
    if plant in test_plants:
        BPlants[idx] = 1
    else:
        BPlants[idx] = 0

print(BPlants)

aa = BPlants  # <----- BPlants
bb = [b + 1 for b, _ in enumerate(BPlants)]  # <----- rolling index over all plants all images
cc = [a * bb[i] for i, a in enumerate(aa)]

BATCH_SIZE = 4
# CLASSES = (3, 4, 5, 6, 11)
CLASSES = (0, 1, 2, 3, 4)

test_idx = []
train_idx = []
for idx, c in enumerate(cc):
    if c != 0:
        test_idx.append(c)
    else:
        train_idx.append(bb[idx])

print('test set indices = ', test_idx)
print('train set indices = ', train_idx)

# TODO: Add num_workers
train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(train_idx))
test_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(test_idx))

print('Train Loader Weight Size:', iter(train_loader).next()[0].size())
print('Train Loader Bias   Size:', iter(train_loader).next()[1].size())

print('Train Size:', len(train_idx))
print('Test Size:', len(test_idx))

Tot_Loss = []
Accuracy = []
sizeTrain = len(train_loader)
MaxEpochs = 50

for epoch in range(MaxEpochs):  # loop over the dataset multiple times

    epoch_loss = 0
    running_loss = 0.0

    # TODO: Dynamic Learning Rate
    # adjust_learning_rate(optimizer, epoch)

    print('epoch = ', epoch)
    for i, data in enumerate(train_loader):
        inputs, labels = data
        labels = torch.Tensor.long(labels)

        # zero the gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        if i % 100 == 0:
            print('.', end='')

    epoch_loss = running_loss / sizeTrain
    Tot_Loss.append(epoch_loss)

    # Computing Accuracy on Test Data:
    correct = 0
    total = 0
    print('test_loader len = ', len(test_loader))
    with torch.no_grad():
        # TODO: DeBug - why IndexError: index 300 is out of bounds for dimension 0 with size 300
        # the bug is in the line below
        for index, val_data in enumerate(test_loader):  # index runs up to 8 and jumps to 300 thereafter
            print('index  = ', index, end='')
            val_images, val_labels = val_data
            val_outputs = net(val_images)
            stam, predicted = torch.max(val_outputs.data, 1)
            print('stam  = ',stam,' predicted  = ',predicted)
            total += val_labels.size(0)
            correct += (predicted == val_labels).sum().item()
            print('index  = ', index, end='')
    Accuracy.append(correct / total)

    # Dynamic plotting
    pl.clf()
    pl.plot(Tot_Loss, '.r', label='Loss curve')
    pl.plot(Accuracy, '.b', label='Accuracy')
    pl.xlabel('Epochs (100ds)')
    pl.ylabel('Loss')
    pl.ylim([0.0, 2.0])
    pl.xlim([-2.0, MaxEpochs + 5.0])
    # pl.title('Precision-Recall example: Max f1={0:0.2f}'.format(max_f1))
    pl.legend(loc="upper right")
    display.clear_output(wait=False)
    display.display(pl.gcf())

print('Finished Training')
