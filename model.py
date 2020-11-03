import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=5, padding=2)

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2)

        self.fc1 = nn.Linear(64 * 3 * 3, 120)
        self.fc2 = nn.Linear(120, 80)
        self.fc3 = nn.Linear(80, 10)


    def forward(self, x):
        x = self.max_pool(F.relu(self.conv1(x)))
        x = self.avg_pool(F.relu(self.conv2(x)))
        x = self.avg_pool(F.relu(self.conv3(x)))

        x = x.view(-1, 64 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
      
        return x