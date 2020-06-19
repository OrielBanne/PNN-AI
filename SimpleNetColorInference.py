import os
from glob import glob

from matplotlib import pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.utils as utils
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim

from torchvision.transforms import ToTensor
from torchvision import transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

experiment = 'Exp3'
modality = 'Color'  # modalities = ['Boson', '577nm', '732nm', '970nm','692nm','polar','Color']
pattern = '*' + modality + '*.jpg'
# pattern   = '*'+modality+'*.tiff'
start_dir = '/home/pnn/experiments/' + experiment + '/'

# finding files in subdirectories:

files = []
# start_dir = os.getcwd()

for dir, _, _ in os.walk(start_dir):
    files.extend(glob(os.path.join(dir, pattern)))

    # arrange files by date
files.sort()
print(files[0:5])

color_pos = (
    (1091, 1103), (1501, 1093), (1875, 1093), (2227, 1109), (2585, 1113), (2909, 1089),
    (1107, 1459), (1489, 1451), (1909, 1447), (2255, 1465), (2587, 1457), (2915, 1467),
    (1109, 1759), (1529, 1771), (1915, 1767), (2259, 1795), (2601, 1791), (2921, 1807),
    (1107, 2059), (1533, 2061), (1917, 2077), (2289, 2073), (2581, 2039), (2945, 2067),
    (1103, 2343), (1551, 2315), (1957, 2359), (2299, 2341), (2599, 2327), (2957, 2341))

print('color_pos  = ', color_pos)

# taking one color image , just one picture of the Hamama:

image_path = files[343]
image = Image.open(image_path)
image.show()


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


net = Net()

criterion = nn.CrossEntropyLoss()

lr = 0.001
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.8)

print(' number of files = ', len(files))

PlantImages = []
for img_idx, image_path in enumerate(files):
    # Getting All Plants Images
    for label_idx, label in enumerate(Labels):
        PlantImages.append(get_image(color_pos[label_idx], time_idx=img_idx))

# Here I must make labels as large as the dataset
BLabels = []
for i in range(len(PlantImages)):
    BLabels.append(Labels[i % len(Labels)])

x_data = torch.stack(PlantImages)
y_data = torch.Tensor(BLabels)  # <---- this is the line that creates the labels as a tensor

dataset = utils.data.TensorDataset(x_data, y_data)
dataloader = DataLoader(dataset)

torch.save(dataset)
torch.utils.data()

num = len(dataset)
print(' dataset length is ', num)
print('the number of hamama images is : ', len(PlantImages))

# TODO: parametrize the ratio of train/test sets
train_set, val_set = torch.utils.data.random_split(dataset, [int(num * 0.8), int(num * 0.2)])

BATCH_SIZE = 4
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

CLASSES = (0, 1, 2, 3, 4)  # original CLASSES = (3, 4, 5, 6, 11)

print('Train Loader Weight Size:', iter(train_loader).next()[0].size())
print('Train Loader Bias   Size:', iter(train_loader).next()[1].size())
print('Train Size:', len(train_set))
print('Test Size:', len(val_set))

####################################################################################
#       Train Loop                                                                 #
####################################################################################

plt.figure(1, figsize=(10, 20), facecolor='w')
Tot_Loss = []
Accuracy = []
MaxEpochs = 50
for epoch in range(MaxEpochs):  # loop over the dataset multiple times

    epoch_loss = 0
    running_loss = 0.0

    # TODO: Dynamic Learning Rate
    # adjust_learning_rate(optimizer, epoch)

    for i, data in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # print('inputs.shape = ',inputs.shape )
        labels = torch.Tensor.long(labels)
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

        if i % 2 == 0:
            print('.', end='')

    epoch_loss = running_loss / sizeTrain
    Tot_Loss.append(epoch_loss)

    # Computing Accuracy on Test Data:

    correct = 0
    total = 0
    with torch.no_grad():
        for testdata in test_loader:
            testimages, testlabels = testdata
            testoutputs = net(testimages)
            _, predicted = torch.max(testoutputs.testdata, 1)
            total += labels.size(0)
            correct += (predicted == testlabels).sum().item()
    Accuracy.append(correct / total)

    # Dynamic plotting
    plt.clf()
    plt.plot(Tot_Loss, '.r', label='Loss curve')
    plt.plot(Accuracy, '.b', label='Accuracy')
    plt.xlabel('Epochs (100ds)')
    plt.ylabel('Loss')
    plt.ylim([0.0, 2.0])
    plt.xlim([-2.0, MaxEpochs + 5.0])
    plt.title('Train Loss vs. Validation accuracy')
    plt.legend(loc="upper right")
    plt.show()

print('Finished Training')
