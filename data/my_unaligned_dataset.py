#%%

#%%
import os
from xml.etree.ElementTree import tostring
import numpy as np
from data.base_dataset import BaseDataset, get_transform, get_transform_pre, get_transform_post
from data.image_folder import make_dataset
from PIL import Image
import random
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF

from natsort import natsorted

class MyUnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A    = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA' RGBD
        self.dir_B    = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB' RGBD
        self.dir_BSeg = os.path.join(opt.dataroot, opt.phase + 'BSeg') # create a path '/path/to/data/trainBSeg' RGBD


        self.A_paths    = sorted(make_dataset(self.dir_A,    opt.max_dataset_size))    # load images from '/path/to/data/trainA'
        self.B_paths    = sorted(make_dataset(self.dir_B,    opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.BSeg_paths = sorted(make_dataset(self.dir_BSeg, opt.max_dataset_size))    # load images from '/path/to/data/trainBSeg'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image        
        self.transform_rgb_pre   = get_transform_pre(self.opt)
        self.transform_depth_pre = get_transform_pre(self.opt,grayscale=True)
        self.transform_post   = get_transform_post(self.opt)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A    (tensor)       -- an image in the input domain
            B    (tensor)       -- its corresponding image in the target domain
            ASeg (tensor)       -- a segmentation image in the input domain corresponding to A image
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within the range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        if self.opt.input_nc >= 4:
            A_img = Image.open(A_path)
        else:
            A_img = Image.open(A_path).convert('RGB')
        if self.opt.output_nc >= 4:
            B_img = Image.open(B_path)
        else:
            B_img = Image.open(B_path).convert('RGB')

        # here add A_Seg_path
        BSeg_path = self.BSeg_paths[index_B]
        BSeg_img = Image.open(BSeg_path).convert('RGB')
        
        # apply image transformation
        # A    = self._preprocess(A_img)
        # B    = self._preprocess(B_img)
        A, _    = self._transform(A_img)
        B, BSeg = self._transform(B_img, BSeg_img)
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'BSeg': BSeg, 'BSeg_paths': BSeg_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

    def _preprocess(self, image):
        image_np  = np.array(image).astype(np.uint8)
        image_rgb = Image.fromarray(image_np[...,:3].astype(np.uint8))
        image_d   = Image.fromarray(image_np[...,3].astype(np.uint8))
        # first make things for rgb and depth separatelly those depending on channels
        image_tensor_rgb  = self.transform_rgb_pre(image_rgb)
        image_tensor_d    = self.transform_depth_pre(image_d)
        # concatenate them
        image_tensor_rgbd = torch.cat((image_tensor_rgb,image_tensor_d[0].unsqueeze(0)),0)
        # only then apply transformations on a concatenated image such as crop, scale or rotation, so that RGB and Depth are in consistancy
        image_tensor_rgbd = self.transform_post(image_tensor_rgbd)

        return image_tensor_rgbd

    def _transform(self, image, image_segment=None):
        image_np         = np.array(image).astype(np.uint8)
        image_np_segment = np.array(image_segment).astype(np.uint8) if image_segment is not None else None
        image_rgb        = Image.fromarray(image_np[...,:3].astype(np.uint8))
        image_depth      = Image.fromarray(image_np[...,3].astype(np.uint8))
        image_segment    = Image.fromarray(image_np_segment.astype(np.uint8)) if image_segment is not None else None

        grayscale = transforms.Grayscale(1)
        image_depth = grayscale(image_depth)
        image_segment = grayscale(image_segment) if image_segment is not None else None
        if 'resize' in self.opt.preprocess:
            osize = [self.opt.load_size, self.opt.load_size]
            resize = transforms.Resize(osize, TF.InterpolationMode.BICUBIC)
            image_depth   = resize(image_depth)
            image_rgb     = resize(image_rgb)
            image_segment = resize(image_segment) if image_segment is not None else None

        if 'crop' in self.opt.preprocess:
            i, j, h, w = transforms.RandomCrop.get_params(image_rgb, output_size=(self.opt.crop_size, self.opt.crop_size))
            image_depth   = TF.crop(image_depth, i, j, h, w)
            image_rgb     = TF.crop(image_rgb, i, j, h, w)
            image_segment = TF.crop(image_segment, i, j, h, w) if image_segment is not None else None

        
        if not self.opt.no_flip:
            if random.random() > 0.5:
                image_depth   = TF.hflip(image_depth)
                image_rgb     = TF.hflip(image_rgb)
                image_segment = TF.hflip(image_segment) if image_segment is not None else None
            
        image_depth          = TF.to_tensor(image_depth)
        image_rgb            = TF.to_tensor(image_rgb)
        image_tensor_segment = TF.to_tensor(image_segment) if image_segment is not None else None

        normalize_grayscale = transforms.Normalize((0.5,), (0.5,))
        image_depth = normalize_grayscale(image_depth)
        normalize_rgb = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        image_rgb = normalize_rgb(image_rgb)

        image_tensor_rgbd = torch.cat((image_rgb,image_depth[0].unsqueeze(0)),0)
        return image_tensor_rgbd, image_tensor_segment
