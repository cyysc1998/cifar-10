import torch
import pandas as pd
import json
from torch.utils.data import DataLoader
from data import ImageDataSet
from model import CNN
from torchvision import transforms

path = './data/test'

tranform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

batch_size = 16

testdata = ImageDataSet(path, 300000, train=False, transform=tranform)
testloader = DataLoader(testdata, batch_size=batch_size, shuffle=False, num_workers=0)

# Cuda
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')

# CNN
cnn = CNN()
cnn.load_state_dict(torch.load('./run/2020-11-02-19-07-35.pth'))
cnn.to(device)
cnn.eval()


# Map
with open('./class.json', 'r') as f:
    classes = json.load(f)
names = [0]*10
for key in classes.keys():
    names[classes[key]] = key

# Test
ans = []
index = 1
for i, data in enumerate(testloader, 0):
    data = data.to(device)
    output = cnn(data)
    labels = output.argmax(dim=1)
    for j in range(len(labels)):
        ans.append({'id': index, 'label': names[labels[j]]})
        index = index + 1
    if i % 1000 == 999:
        print('[%d] Finished' % (batch_size * (i + 1)))
    
    
result = pd.DataFrame(ans)
result.to_csv('./data/result.csv', index=False)