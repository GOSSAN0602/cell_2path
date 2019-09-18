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
from libs.trainer import trainer
from libs.radam import RAdam

import warnings
warnings.filterwarnings('ignore')

# some config
path_data = '/home/shuki_goto/input'
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

# Define dataset
ds = ImagesDS(df_train, path_data, mode='train')
ds_val = ImagesDS(df_val, path_data, mode='train')

# make model
#model = EfficientNet.from_pretrained('efficientnet-b5')
model = models.resnet18(pretrained=True)
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
#model = torch.nn.DataParallel(model)

# define data Loader
train_loader = D.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
eval_loader = D.DataLoader(ds_val, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# define trainer
trainer = trainer(SAVE_PATH, max_epochs)
optimizer = RAdam(model.parameters())

# train
loss_array = np.zeros(max_epochs)
acc_array = np.zeros(max_epochs)
for i in range(max_epochs):
    model, loss = trainer.train(model, optimizer, train_loader, i)
    loss_array[i] = loss
    acc = trainer.evaluate(model, eval_loader)
    acc_array[i] = acc
    print('epoch %d loss %.2f acc %.2f%%' % (i, loss, acc))
    torch.save(model.state_dict(), SAVE_PATH+str(i)+'.pth')
    np.save(SAVE_PATH+'acc_array.npy', acc_array)
    np.save(SAVE_PATH+'loss_array.npy', loss_array)
