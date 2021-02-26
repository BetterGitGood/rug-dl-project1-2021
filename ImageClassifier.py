# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# A part of the code has been taken and edited from this guide.
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# running on CPU or GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# turn to false if you already have the data.
download_data = True

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=download_data, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=download_data, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Naive Student Network Setup
class Naive_Student(nn.Module):
    # defining network parts
    def __init__(self):
        super(Naive_Student, self).__init__()
        self.decode = nn.Linear(1024, 4096)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 13 * 13, 3072)
        self.conv3 = nn.Conv2d(3, 6, 5)
        self.conv4 = nn.Conv2d(6, 16, 5)
        self.fc2 = nn.Linear(16 * 5 * 5, 200)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, 10)

    # defining model calculations
    def forward(self, x):
        x = x.view(4, 3,1024)
        x = self.decode(x)
        x= x.view(4, 3, 64, 64)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 13 * 13)
        x = F.relu(self.fc1(x))
        x = x.view(4, 3, 32, 32)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# training function
def train_network(net, criterion, optimizer, name):
    loss_list = []
    acc_list = []
    epoch_list = []
    step_list = []
    
    one_back_acc = 0
    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # accuracy
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            # print statistics
            running_loss += loss.item()
            total += labels.size(0)
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                print('[%d, %5d] accuracy: %.3f' %
                    (epoch + 1, i + 1, 100 * correct / total))
                
                loss_list.append(running_loss / 2000)
                acc_list.append(100 * correct / total)
                epoch_list.append(epoch+1)
                step_list.append(i+1)
                running_loss = 0.0
                correct = 0
                total = 0

        # checking testing accuracy after every loop
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                test_in, labels = data[0].to(device), data[1].to(device)
                outputs = net(test_in)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_acc = (100 * correct / total)
        print('Accuracy of the network on the 10000 test images: %d %%' % test_acc)

        # early stopping
        if test_acc < one_back_acc:
            break
        one_back_acc = test_acc

    # saving the results
    running_results = {'Loss': loss_list, 'Accuracy': acc_list, 'Epoch': epoch_list, 'Step': step_list}
    df = pd.DataFrame(running_results, columns = ['Loss', 'Accuracy', 'Epoch', 'Step'])
    path = './results/running-results' + name
    df.to_csv(path)
    print('Finished Training')

# making a list of models
# All ResNet models are taken from the model library in pytorchvision 
model_list = []
model_list.append(models.resnet18().to(device))
model_list.append(models.resnet34().to(device))
model_list.append(models.resnet50().to(device))
model_list.append(models.resnet101().to(device))
model_list.append(Naive_Student().to(device))

# testing all models with AdamW
modelnr = 4
for model in model_list:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters())
    train_network(model, criterion, optimizer, (str(modelnr) + 'AdamW'))
    PATH = './cifar_adamW_' + str(modelnr) + '_net.pth'
    torch.save(model.state_dict(), PATH)
    modelnr = modelnr + 1
    
# testing all models with Adam
modelnr = 4
for model in model_list:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    train_network(model, criterion, optimizer, (str(modelnr) + 'Adam'))
    PATH = './cifar_adam_' + str(modelnr) + '_net.pth'
    torch.save(model.state_dict(), PATH)
    modelnr = modelnr + 1

print("End")