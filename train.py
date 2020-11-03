import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import datetime
from torch.utils.data import DataLoader
from torchvision import transforms

from data import ImageDataSet
from model import CNN

# Data
trainsform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
trainset = ImageDataSet('./data/train', 50000, train=True, transform=trainsform)
trainloader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=0)

# Cuda
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')

# Model
cnn = CNN()
cnn.to(device)

learning_rate = 0.002
momentum = 0.9

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=learning_rate, momentum=momentum)

#train
epochs = 10

for epoch in range(epochs):
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

       
        optimizer.zero_grad()
        outputs = cnn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
      
        running_loss += loss.item()

        if i % 1000 == 999:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0
print("Finished training")

cur_time = datetime.datetime.now()
time_str = datetime.datetime.strftime(cur_time,'%Y-%m-%d-%H-%M-%S')
torch.save(cnn.state_dict(), './run/' + time_str + '.pth')
