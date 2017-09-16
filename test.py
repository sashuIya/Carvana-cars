from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from model.u_net import UNet
from params import *
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt


class CarvanaTestDataset(Dataset):
    def __init__(self, ids):
        self.ids = ids
        print(ids[:10])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        img = cv2.imread('{}/{}.jpg'.format(TEST_DIR, img_id))
        img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
        img = np.array(img, np.float32) / 255
        return torch.from_numpy(img).permute(2, 0, 1)


# https://www.kaggle.com/stainsby/fast-tested-rle
def run_length_encode(mask):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    inds = mask.flatten()
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle


def load_best_model(model: nn.Module) -> None:
    state = torch.load(str(SAVED_MODELS_ROOT / SAVED_MODEL))
    model.load_state_dict(state['state_dict'])
    print('Loaded model from epoch {epoch}'.format(**state))


def main():
    df_test = pd.read_csv(DATA_ROOT / 'sample_submission.csv')
    ids_test = df_test['img'].map(lambda s: s.split('.')[0])

    test_dataset = CarvanaTestDataset(ids_test.values)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=TEST_BATCH_SIZE)

    model = UNet()
    load_best_model(model)
    model.cuda()
    model.eval()

    rles4 = []
    rles5 = []
    rles6 = []
    for inputs in tqdm(test_loader):
        inputs = Variable(inputs.cuda())
        outputs = model(inputs).data.cpu().numpy()
        for k in range(outputs.shape[0]):
            mask = outputs[k][0]
            mask = cv2.resize(mask, (ORIGINAL_WIDTH, ORIGINAL_HEIGHT))
            mask4 = np.array(mask > 0.4, np.uint8)
            mask5 = np.array(mask > 0.5, np.uint8)
            mask6 = np.array(mask > 0.6, np.uint8)
            #  plt.imshow(mask)
            #  plt.show()
            rle4 = run_length_encode(mask4)
            rles4.append(rle4)
            rle5 = run_length_encode(mask5)
            rles5.append(rle5)
            rle6 = run_length_encode(mask6)
            rles6.append(rle6)

    print("Generating submission file...")
    names = []
    for img_id in ids_test:
        names.append('{}.jpg'.format(img_id))
    df4 = pd.DataFrame({'img': names, 'rle_mask': rles4})
    df4.to_csv('submit/submission_0_4.csv.gz', index=False, compression='gzip')

    df5 = pd.DataFrame({'img': names, 'rle_mask': rles4})
    df5.to_csv('submit/submission_0_5.csv.gz', index=False, compression='gzip')

    df6 = pd.DataFrame({'img': names, 'rle_mask': rles4})
    df6.to_csv('submit/submission_0_6.csv.gz', index=False, compression='gzip')


if __name__ == '__main__':
    main()
