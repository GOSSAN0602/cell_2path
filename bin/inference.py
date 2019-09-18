import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data as D
from torchvision import models, transforms as T
from efficientnet_pytorch import EfficientNet

import sys
sys.path.append('./')
from libs.data import ImagesDS

import warnings
warnings.filterwarnings('ignore')

# some config
path_data = '../input'
SAVE_PATH = '/home/shuki_goto/log/'
device = 'cuda'
batch_size = 32

# define data loader
df_test = pd.read_csv(path_data+'/test.csv')
ds_test = ImagesDS(df_test, path_data, mode='test')
tloader = D.DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# make model
model = EfficientNet.from_pretrained('efficientnet-b5')
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, num_classes)
# add a new layer to combine outputs from two paths.
model.head = torch.nn.Linear(num_classes, num_classes)
trained_kernel = model.conv1.weight
# make our model work with 6 channels
new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
with torch.no_grad():
    new_conv.weight[:] = torch.stack([torch.mean(trained_kernel, 1)]*6, dim=1)
model.conv1 = new_conv
model = model.to(device)

# Load weight
model.load_weight(SAVE_PATH+'40.pth')

model.eval()
with torch.no_grad():
    preds = np.empty(0)
    for input, _ in tqdm(tloader):
        input1, input2 = input[:, :6].to(device), input[:, 6:].to(device)
        output = model.head(model(input1) - model(input2))
        idx = output.max(dim=-1)[1].cpu().numpy()
        preds = np.append(preds, idx, axis=0)

# Make submission
submission = pd.read_csv(path_data + '/test.csv')
submission['sirna'] = preds.astype(int)
submission.to_csv('submission.csv', index=False, columns=['id_code','sirna'])
