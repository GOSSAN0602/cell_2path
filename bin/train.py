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

import warnings
warnings.filterwarnings('ignore')

# some config
path_data = '../input'
SAVE_PATH = '/home/shuki_goto/log/'
device = 'cuda'
batch_size = 32
max_epochs = 40
img_size = 384
torch.manual_seed(0)
np.random.seed(0)
num_classes = 1108
num_workers = 4

# read tables
df = pd.read_csv(path_data+'/train.csv')
in_eval = df.experiment.isin(['HEPG2-07', 'HUVEC-16', 'RPE-07'])
df_train = df[~in_eval]
df_val = df[in_eval]
df_test = pd.read_csv(path_data+'/test.csv')

# Define dataset
ds = ImagesDS(df_train, path_data, mode='train')
ds_val = ImagesDS(df_val, path_data, mode='train')
ds_test = ImagesDS(df_test, path_data, mode='test')

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

# define data Loader
train_loader = D.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
eval_loader = D.DataLoader(ds_val, batch_size=batch_size, shuffle=True, num_workers=num_workers)
tloader = D.DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# define trainer
trainer = trainer(SAVE_PATH, max_epochs)

# train
for i in range(max_epochs):
    model, loss = trainer.train(model, train_loader, i)
    acc = trainer.evaluate(model, eval_loader)
    print('epoch %d loss %.2f acc %.2f%%' % (epoch, loss, acc))
    torch.save(model.state_dict(), SAVE_PATH+str(i)+'pth')
    import pdb; pdb.set_trace()
