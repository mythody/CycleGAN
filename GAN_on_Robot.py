import os

import torch
from options.test_options import TestOptions
from models import create_model
import sys

class GANModelOnRobot():
    def __init__(self, model_name = 'franka_2022_01_segment_unet_not_mean_D_multi_rnd_128'):
        # sys.argv=["dummy_of_the_program", "--dataroot", "dummy_root", "--gpu_ids", "-1"]
        opt = TestOptions().parse() # get test options

        # parser = BaseOptions.initialize(self, parser)  # define shared options
        opt.aspect_ratio = 1.0
        opt.phase        = 'train'
        opt.isTrain      = False

        # those may be changable
        opt.segmentChannels = 3
        opt.load_size       = 128
        opt.crop_size       = opt.load_size
        opt.name            = model_name
        opt.model           = 'cycle_gan_segment'
        opt.input_nc        = 4
        opt.output_nc       = 4
        opt.netD            = 'segmentatorUnet_128'
        opt.netG            = 'resnet_9blocks'
        opt.n_layers_D      = 3
        opt.onRobot         = True
        opt.dataset_mode    = 'my_unaligned'
        opt.direction       = 'AtoB'
        opt.eval            = False
        opt.no_dropout      = True
        opt.gpu_ids         = []


        # hard-code some parameters for test
        opt.num_threads = 0   # test code only supports num_threads = 0
        opt.batch_size = 1    # test code only supports batch_size = 1
        opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.

        opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
        
        self.model = create_model(opt)      # create a model given opt.model and other options
        self.model.setup(opt)               # regular setup: load and print networks; create schedulers
        
        if opt.eval:
            self.model.eval()
        
    
    def real_to_sim(self, real_rgbd_img):
        real_rgbd_img = torch.from_numpy(real_rgbd_img.copy()).unsqueeze(0).type(torch.float32) # shape has to be (1,4,128,128) # first 3 - RGB # last 1 - Depth

        self.model.set_input(real_rgbd_img)
        self.model.test()
        visuals = self.model.get_current_visuals()

        rgb   = ((visuals['fake_B_RGB'][0].permute(1, 2, 0) + 1.0) / 2.0)   # between 0 and 1 
        depth = ((visuals['fake_B_Depth'][0].permute(1, 2, 0) + 1.0) / 2.0) # between 0 and 1
        segment = (visuals['pred_fakeBSeg'][0].permute(1, 2, 0)*255)

        return torch.cat((rgb,depth,segment), 2).cpu().detach().numpy()