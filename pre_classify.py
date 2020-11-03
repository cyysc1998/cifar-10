import numpy as np
import pandas as pd
import json

path = './data/trainLabels.csv'

classes = {}

lables = pd.read_csv(path, delimiter=',')
total = lables.groupby('label').count()

i = 0
for item in total.iterrows():
    classes[item[0]] = i
    i = i + 1;

with open('./class.json', 'w') as f:
    json.dump(classes, f, indent=4)


