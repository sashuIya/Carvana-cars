import cv2
import numpy as np
import pandas as pd
import params
import torch
import train_util
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from model.u_net import UNet
from model.losses import Loss
from torch import optim
from image_transformations import *
from params import *
import random

import matplotlib
import matplotlib.pyplot as plt

class CarvanaTrainDataset(Dataset):
    def __init__(self, ids):
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def resize(self, img):
        return cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))

    def crop(self, img, crop_x, crop_y):
        crop = img[crop_x:crop_x + INPUT_SIZE, crop_y:crop_y + INPUT_SIZE]
        return crop

    def __getitem__(self, index):
        id = self.ids[index]
        img = cv2.imread('{}/{}.jpg'.format(TRAIN_DIR, id))
        mask = cv2.imread('{}/{}_mask.png'.format(MASKS_DIR, id), cv2.IMREAD_GRAYSCALE)
        crop_x = random.randint(0, ORIGINAL_HEIGHT - INPUT_SIZE)
        crop_y = random.randint(0, ORIGINAL_WIDTH - INPUT_SIZE)

        img = self.crop(img, crop_x, crop_y)
        mask = self.crop(mask, crop_x, crop_y)

        img = random_hue_saturation_value(
            img, hue_shift_limit=(-50, 50), sat_shift_limit=(-5, 5), val_shift_limit=(-15, 15))
        #  img, mask = random_shift_scale_rotate(
        #  img, mask, shift_limit=(-0.0625, 0.0625), scale_limit=(-0.1, 0.1), rotate_limit=(-0, 0))
        #  img, mask = random_horizontal_flip(img, mask)

        #  plt.imshow(img)
        #  plt.show()

        mask = np.expand_dims(mask, axis=2)
        img = np.array(img, np.float32) / 255
        mask = np.array(mask, np.float32) / 255
        return torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy(mask).permute(2, 0, 1)


def load_best_model(model: nn.Module) -> None:
    state = torch.load(str(SAVED_MODELS_ROOT / SAVED_MODEL))
    model.load_state_dict(state['state_dict'])
    print('Loaded model from epoch {epoch}'.format(**state))


def main():
    df_train = pd.read_csv(DATA_ROOT / 'train_masks.csv')
    ids_train = df_train['img'].map(lambda s: s.split('.')[0])

    ids_train_split, ids_valid_split = train_test_split(ids_train, test_size=0.2, random_state=42)

    print('Training on {} samples'.format(len(ids_train_split)))
    print('Validating on {} samples'.format(len(ids_valid_split)))

    train_dataset = CarvanaTrainDataset(ids_train_split.values)
    valid_dataset = CarvanaTrainDataset(ids_valid_split.values)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=TRAIN_BATCH_SIZE)
    valid_loader = DataLoader(valid_dataset, batch_size=TRAIN_BATCH_SIZE)

    model = UNet()
    model.cuda()
    if LOAD_MODEL:
        load_best_model(model)
    model.cuda()

    criterion = Loss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.0001)

    train_util.train(model, criterion, optimizer, 100, train_loader, valid_loader)


if __name__ == '__main__':
    main()
