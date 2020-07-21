from glob import glob
from sklearn.model_selection import GroupKFold
import cv2
from skimage import io
import torch
from torch import nn
import os
from datetime import datetime
import time
import random
import cv2
import pandas as pd
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
import sklearn
import jpegio as jpio


def JPEGdecompressYCbCr(jpegStruct):

    [col, row] = np.meshgrid(range(8), range(8))
    T = 0.5 * np.cos(np.pi * (2 * col + 1) * row / (2 * 8))
    T[0, :] = T[0, :] / np.sqrt(2)

    img_dims = np.array(jpegStruct.coef_arrays[0].shape)
    n_blocks = img_dims // 8
    broadcast_dims = (n_blocks[0], 8, n_blocks[1], 8)

    YCbCr = []
    for i, dct_coeffs, in enumerate(jpegStruct.coef_arrays):

        if i == 0:
            QM = jpegStruct.quant_tables[i]
        else:
            QM = jpegStruct.quant_tables[1]

        t = np.broadcast_to(T.reshape(1, 8, 1, 8), broadcast_dims)
        qm = np.broadcast_to(QM.reshape(1, 8, 1, 8), broadcast_dims)
        dct_coeffs = dct_coeffs.reshape(broadcast_dims)

        a = np.transpose(t, axes=(0, 2, 3, 1))
        b = (qm * dct_coeffs).transpose(0, 2, 1, 3)
        c = t.transpose(0, 2, 1, 3)

        z = a @ b @ c
        z = z.transpose(0, 2, 1, 3)
        YCbCr.append(z.reshape(img_dims))

    return np.stack(YCbCr, -1).astype(np.float32)

def get_valid_transforms():
    return A.Compose([
        A.Resize(height=512, width=512, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.0)


DATA_ROOT_PATH = '/home/shareData/ALASKA2'

from efficientnet_pytorch import EfficientNet

def get_net():
    net = EfficientNet.from_pretrained('efficientnet-b4')
    net._fc = nn.Linear(in_features=1792, out_features=4, bias=True)
    return net

net = get_net().cuda(1)

checkpoint = torch.load('/home/shareData/ys/alaska2/EB4_ycbcr/best-checkpoint-028epoch.bin',map_location={'cuda:0':'cuda:1'})
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()

class DatasetSubmissionRetriever(Dataset):

    def __init__(self, image_names, transforms):
        super().__init__()
        self.image_names = image_names
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_name = self.image_names[index]
        # image = cv2.imread(f'{DATA_ROOT_PATH}/Test/{image_name}', cv2.IMREAD_COLOR)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # image /= 255.0
        jpegStruct = jpio.read(f'{DATA_ROOT_PATH}/Test/{image_name}')
        imDecompressYCbCr = JPEGdecompressYCbCr(jpegStruct)

        image = imDecompressYCbCr/255.0
        sample = {'image': image}
        sample = self.transforms(**sample)
        image = sample['image']

        return image_name, image

    def __len__(self) -> int:
        return self.image_names.shape[0]


dataset = DatasetSubmissionRetriever(
    image_names=np.array([path.split('/')[-1] for path in glob('/home/shareData/ALASKA2/Test/*.jpg')]),
    transforms=get_valid_transforms(),

)

data_loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=False,
    num_workers=2,
    drop_last=False,
)

result = {'Id': [], 'Label': []}
for step, (image_names, images) in enumerate(data_loader):
    print(step, end='\r')

    y_pred = net(images.cuda(1))
    y_pred = 1 - nn.functional.softmax(y_pred, dim=1).data.cpu().numpy()[:, 0]

    result['Id'].extend(image_names)
    result['Label'].extend(y_pred)

submission = pd.DataFrame(result)
submission.to_csv('submission_b4.csv', index=False)
print(submission.head())