import torch.utils.data as data
import torch
from skimage.io import imread, imsave
import torchvision.transforms as tf
import numpy as np
import random
import os
import glob
import h5py
import sys
import os
from PIL import Image
sys.path.append('..')
from skimage.transform import rescale
from os.path import join
from skimage.transform import rotate
from skimage import img_as_float

#=================== For Testing ===================#
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", "JPG"])

#
class DataValSet(data.Dataset):
    def __init__(self, root, mean=(128, 128, 128), isReal= False):
        self.root = root
        self.mean = mean
        self.isReal = isReal
        self.input_dir = os.path.join(self.root, 'HR_hazy')
        #self.target_lr_dir = os.path.join(self.root, 'LR')
        self.target_dir = os.path.join(self.root, 'HR')


        # for split in ["train", "trainval", "val"]:
        self.input_ids = [x for x in sorted(os.listdir(self.input_dir)) if is_image_file(x)]
       # self.target_lr_ids = [x for x in sorted(os.listdir(self.target_lr_dir)) if is_image_file(x)]
        if not self.isReal:
            self.target_ids = [x for x in sorted(os.listdir(self.target_dir)) if is_image_file(x)]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        name = self.input_ids[index]
        input_image = imread(os.path.join(self.input_dir, "%s" % name))
        input_image = input_image.transpose((2, 0, 1))
        input_image = np.asarray(input_image, np.float32)
        input_image /= 255

        if not self.isReal:
            name = self.target_ids[index]
            target_image = imread(os.path.join(self.target_dir, "%s" % name))
            target_image = target_image.transpose((2, 0, 1))
            target_image = np.asarray(target_image, np.float32)
            target_image /= 255

        if not self.isReal:
            return input_image.copy(), target_image.copy(), name
        else:
            return input_image.copy(), name

#=================== For Training ===================#


class DataSet_HDF5(data.Dataset):
    def __init__(self, file_path):
        super(DataSet_HDF5, self).__init__()
        hf = h5py.File(file_path,'r')
        self.data = hf.get("data")
        self.target = hf.get("label")
        # hf.close()

    def __getitem__(self, index):
        # randomly flip
        #print(index)
        #data shppe: C*H*W
        LR_patch = self.data[index, :, :, :]
        HR_patch = self.target[index, :, :, :]
        LR_patch = np.clip(LR_patch, 0, 1)  # we might get out of bounds due to noise
        HR_patch = np.clip(HR_patch, 0, 1)  # we might get out of bounds due to noise
        LR_patch = np.asarray(LR_patch, np.float32)
        HR_patch = np.asarray(HR_patch, np.float32)

        flip_channel = random.randint(0, 1)
        if flip_channel != 0:
            LR_patch = np.flip(LR_patch, 2)
            HR_patch = np.flip(HR_patch, 2)
        # randomly rotation
        rotation_degree = random.randint(0, 3)
        LR_patch = np.rot90(LR_patch, rotation_degree, (1,2))
        HR_patch = np.rot90(HR_patch, rotation_degree, (1,2))
        return LR_patch.copy(), \
               HR_patch.copy()

    def __len__(self):
        return self.data.shape[0]

