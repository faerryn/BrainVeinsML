#!/usr/bin/env python

import os

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def chunkify(li, n):
    # Chunkfy (unflatten) li into a 2D array with depth of n
    return [li[i : i + n] for i in range(0, len(li), n)]


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class VeinDataset(Dataset):
    def __init__(self, image_file, path_file, sample_shape, skip=0):
        image = nib.load(image_file)
        self.image = image.get_fdata()[:-skip]
        self.image_shape = np.shape(self.image)  # shape of the image
        self.sample_shape = sample_shape  # shape of each sample chunk of the image
        self.sample_space = (
            self.image_shape[0] - self.sample_shape[0] + 1,
            self.image_shape[1] - self.sample_shape[1] + 1,
            self.image_shape[2] - self.sample_shape[2] + 1,
        )  # shape of space of all possible chunks
        veins = []
        with open(path_file) as f:
            for line in f:
                vein = []
                for x, y, z in chunkify([int(i) for i in line.split()], 3):
                    if x < self.image_shape[0]:
                        vein.append([x, y, z])
                    elif len(vein) > 0 and vein not in veins:
                        veins.append(vein)
                        vein = []
                if len(vein) > 0 and vein not in veins:
                    veins.append(vein)
                    vein = []
        self.veins = veins

    def __len__(self):
        return np.prod(self.sample_space)

    def __getitem__(self, idx):
        x = idx // (self.sample_space[1] * self.sample_space[2])
        y = (idx // self.sample_space[2]) % self.sample_space[1]
        z = idx % self.sample_space[2]

        simg = self.image[
            x : x + self.sample_shape[0],
            y : y + self.sample_shape[1],
            z : z + self.sample_shape[2],
        ]

        xr = range(x, x + self.sample_shape[0])
        yr = range(y, y + self.sample_shape[1])
        zr = range(z, z + self.sample_shape[2])

        sveins = []
        for vein in self.veins:
            svein = []
            for xv, yv, zv in vein:
                if (xv in xr) and (yv in yr) and (zv in zr):
                    svein.append([xv, yv, zv])
                elif len(svein) > 0 and svein not in sveins:
                    sveins.append(svein)
                    svein = []
            if len(svein) > 0 and svein not in sveins:
                sveins.append(svein)
                svein = []

        return simg, sveins


SAMPLE_SHAPE = (11, 11, 11)

dataset = VeinDataset(
    os.path.join("data", "804893_SWI_TE1_leftbox.nii"),
    os.path.join("data", "901726_804893_TE1_leftbox_path.txt"),
    SAMPLE_SHAPE,
    skip=10,
)


# I want the image of a specific vein in an 11x11x11 block
class VeinsNetwork(nn.Module):
    def __init__(self):
        super(VeinsNetwork, self).__init__()
        # Conv, ReLU, Pool, and repeat.  Then UnPool, ReLU, UnConv.
        self.conv_stack = nn.Sequential(
            nn.Conv3d(2, 2, (3, 3, 3)),
            nn.ReLU(),
            nn.MaxPool3d((3, 3, 3)),
            nn.Conv3d(2, 2, (3, 3, 3)),
            nn.ReLU(),
            nn.MaxPool3d((3, 3, 3)),
            nn.Flatten(),
            nn.Unflatten(1, ()),
        )

    def forward(self, x):
        return self.conv_stack(x)


model = VeinsNetwork().to(device)
print(model)
