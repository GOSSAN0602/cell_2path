
import numpy as np
import pandas as pd
from PIL import Image
import sys
sys.path.append('./')

import torch
import torch.nn as nn
import torch.utils.data as D
import torch.nn.functional as F
import torchvision
from torchvision import transforms as T
from libs.radam import RAdam

from tqdm import tqdm

class trainer():
    def __init__(self, SAVE_PATH, num_epochs):
        self.SAVE_PATH = SAVE_PATH
        self.device='cuda'
        self.criterion = nn.CrossEntropyLoss()

    def train(self, model, optimizer, train_loader, epoch):
        model.train()
        if epoch == 0:
            # update only the last two FC layers
            for name, child in model.named_children():
                if (name != 'head') and (name != 'fc'):
                    for param in child.parameters():
                        param.requires_grad = False
        elif epoch == 3:
            # enable update on all layers
            for name, child in model.named_children():
                for param in child.parameters():
                    param.requires_grad = True

        loss_sum = 0
        for input, target in tqdm(train_loader):
            input1, input2 = input[:, :6].to(self.device), input[:, 6:].to(self.device)
            target = target.to(self.device)

            output = model.head(model(input1) - model(input2))
            loss = self.criterion(output, target)
            loss_sum += loss.data.cpu().numpy()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return model, (loss_sum / len(train_loader))

    def evaluate(self, model, eval_loader):
        model.eval()
        correct = 0
        with torch.no_grad():
            for input, target in tqdm(eval_loader):
                input1, input2 = input[:, :6].to(self.device), input[:, 6:].to(self.device)
                target = target.to(self.device)

                output = model.head(model(input1) - model(input2))
                preds = output.argmax(axis=1)
                correct += (target == preds).sum()

        return correct.cpu().numpy() * 100 / len(eval_loader.dataset)
