#!/usr/bin/env python

import os

import nibabel as nib
import numpy as np

SKIP = 5
IMG = os.path.join("data", "804893_SWI_TE1_leftbox.nii")
PATH = os.path.join("data", "901726_804893_TE1_leftbox_path.txt")
OUT = os.path.join("data", "901726_804893_TE1_leftbox_path.nii")


def chunks(li, n):
    """Split a given list in to chunks with the given length.  The
    last chunk may have less than the specified length.

    """
    return [li[i : min(i + n, len(li))] for i in range(0, len(li), n)]


leftbox_img = nib.load(IMG)

leftbox_path = np.zeros(leftbox_img.header.get_data_shape())
with open(PATH) as f:
    for vein in f:
        for [x, y, z] in chunks([int(i) for i in vein.split()], 3)[SKIP:]:
            leftbox_path[x, y, z] = 1

leftbox_path_img = nib.Nifti1Image(leftbox_path, leftbox_img.affine)
nib.save(leftbox_path_img, OUT)
