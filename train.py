#!/usr/bin/env python

import importlib
import os

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from monai.networks.nets.unet import UNet
from torch.utils.data import DataLoader, Dataset


def chunkify(li, n):
    # Chunkfy (unflatten) li into a 2D array with depth of n
    return [li[i : i + n] for i in range(0, len(li), n)]


class VeinDataset(Dataset):
    # sample_shape is the shape of eachsmaller chunk that we want to
    # get (since a big image can have many smaller views) narrow is
    # for cutting off fuzzy edges
    def __init__(self, image_file, path_file, sample_shape=None, narrow=slice(None)):
        self.image = torch.from_numpy(
            nib.load(image_file).get_fdata()[narrow]
        ).unsqueeze(0)

        self.shape = np.shape(self.image[0])  # shape of the image

        self.sample_shape = (
            sample_shape or self.shape
        )  # shape of each sample chunk of the image

        self.sample_space = (
            self.shape[0] - self.sample_shape[0] + 1,
            self.shape[1] - self.sample_shape[1] + 1,
            self.shape[2] - self.sample_shape[2] + 1,
        )  # shape of space of all possible chunks

        self.vein_paths = []
        with open(path_file) as f:
            for line in f:
                path = []
                for x, y, z in chunkify([int(i) for i in line.split()], 3):
                    if x < self.shape[0]:
                        path.append([x, y, z])
                    elif len(path) > 0 and path not in self.vein_paths:
                        self.vein_paths.append(path)
                        path = []
                if len(path) > 0 and path not in self.vein_paths:
                    self.vein_paths.append(path)
                    path = []

        self.veins_image = np.zeros(self.shape)
        for path in self.vein_paths:
            for x, y, z in path:
                self.veins_image[x, y, z] = 1
        self.veins_image = torch.from_numpy(self.veins_image).unsqueeze(0)

    def __len__(self):
        return np.prod(self.sample_space)

    def __getitem__(self, idx):
        x = idx // (self.sample_space[1] * self.sample_space[2])
        y = (idx // self.sample_space[2]) % self.sample_space[1]
        z = idx % self.sample_space[2]
        coords = (x, y, z)

        sample_slice = (slice(None),) + tuple(
            [
                slice(coords[dim], coords[dim] + self.sample_shape[dim])
                for dim in range(3)
            ]
        )

        # xr = range(x, x + self.sample_shape[0])
        # yr = range(y, y + self.sample_shape[1])
        # zr = range(z, z + self.sample_shape[2])

        # sveins = []
        # for vein in self.veins:
        #     svein = []
        #     for xv, yv, zv in vein:
        #         if (xv in xr) and (yv in yr) and (zv in zr):
        #             svein.append([xv, yv, zv])
        #         elif len(svein) > 0 and svein not in sveins:
        #             sveins.append(svein)
        #             svein = []
        #     if len(svein) > 0 and svein not in sveins:
        #         sveins.append(svein)
        #         svein = []

        simg = self.image[sample_slice]
        svimg = self.veins_image[sample_slice]
        return simg, svimg


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += pred.round().eq(y).sum() / np.prod(np.shape(pred))

    test_loss /= num_batches
    correct /= size
    print("Test Error:")
    print(f" Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")


NARROW = slice(None, -5)
SAMPLE_SHAPE = (11, 11, 11)

leftbox_data = VeinDataset(
    os.path.join("data", "804893_SWI_TE1_leftbox.nii"),
    os.path.join("data", "901726_804893_TE1_leftbox_path.txt"),
    sample_shape=SAMPLE_SHAPE,
    narrow=NARROW,
)

rightbox_data = VeinDataset(
    os.path.join("data", "804893_SWI_TE1_rightbox.nii"),
    os.path.join("data", "901726_804893_TE1_rightbox_path.txt"),
    sample_shape=SAMPLE_SHAPE,
    narrow=NARROW,
)

learning_rate = 0.001
batch_size = 2

leftbox_dataloader = DataLoader(
    leftbox_data,
    batch_size=batch_size,
    shuffle=True,
)
rightbox_dataloader = DataLoader(
    rightbox_data,
    batch_size=batch_size,
    shuffle=True,
)

train_dataloader = leftbox_dataloader
test_dataloader = rightbox_dataloader

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

MODEL_PATH = "model.pth"

if os.path.exists(MODEL_PATH):
    model = torch.load(MODEL_PATH)
else:
    model = nn.Sequential(
        UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(1,) * 3,
            strides=(1,) * 2,
        ),
        nn.Sigmoid(),
    )

model.double()

loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epoch = 1
while True:
    print(f"Epoch {epoch}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
    torch.save(model, MODEL_PATH)
    epoch += 1
