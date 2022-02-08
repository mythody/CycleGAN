import torch
import itertools
from util.image_pool import ImagePool, MyImagePoolWithSegmentation
from .base_model import BaseModel
from . import networks


class CycleGANSegmentModel(BaseModel):
    """
    This class implements the CycleGAN plus Segmentation model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B','DSeg'] # , 'DSegA', 'DSegB']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A = ['real_A_RGB','fake_B_RGB','real_A_Depth','fake_B_Depth','real_BSeg','pred_realBSeg','pred_fakeASeg','real_A', 'fake_B', 'rec_A', 'idt_B']
            visual_names_B = ['real_B', 'fake_A', 'rec_B', 'real_B_RGB', 'fake_A_RGB', 'real_B_Depth', 'fake_A_Depth','idt_A']
        else:
            # visual_names_A = ['real_A_RGB','fake_B_RGB','real_A_Depth','fake_B_Depth','real_BSeg','pred_fakeASeg','real_A', 'fake_B', 'rec_A']
            # visual_names_B = ['real_B', 'fake_A', 'rec_B', 'real_B_RGB', 'fake_A_RGB', 'real_B_Depth', 'fake_A_Depth']
            visual_names_A = ['real_A_RGB','fake_B_RGB','real_A_Depth','fake_B_Depth','pred_fakeBSeg', 'real_BSeg', 'pred_realBSeg', 'pred_fakeASeg', 'real_Seg_for_fake_A', 'pred_fakeASegDA', 'real_A', 'fake_B', 'rec_A']
            visual_names_B = ['real_B', 'fake_A', 'rec_B', 'real_B_RGB', 'fake_A_RGB', 'real_B_Depth', 'fake_A_Depth']

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B', 'D_A_Seg', 'D_B_Seg']
        else:  # during test time, only load Gs
            # self.model_names = ['G_A', 'G_B', 'D_A', 'DSeg_A', 'DSeg_B']
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_A_Seg', 'D_B_Seg']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
        #                                 opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        # self.netDSeg_A = networks.define_D(opt.output_nc, opt.ndf, 'segmentator',
        #                                 opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        # self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
        #                                 opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        # self.netDSeg_B = networks.define_D(opt.output_nc, opt.ndf, 'segmentator',
        #                                 opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        self.netD_A = networks.define_D(opt.output_nc, opt.ndf, 'segmentatorEncoder',
                                        opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netD_B = networks.define_D(opt.input_nc, opt.ndf, 'segmentatorEncoder',
                                        opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        self.netD_A_Seg = networks.define_D(opt.output_nc, opt.ndf, 'segmentatorDecoder',
                                        opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netD_B_Seg = networks.define_D(opt.input_nc, opt.ndf, 'segmentatorDecoder',
                                        opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = MyImagePoolWithSegmentation(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.real_B_pool = MyImagePoolWithSegmentation(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionDSeg = torch.nn.BCEWithLogitsLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_DSeg = torch.optim.Adam(itertools.chain(self.netD_A_Seg.parameters(), self.netD_B_Seg.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_DSeg)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_BSeg = input['BSeg'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))
        # self.seg_fake_A = self.netDSeg_A((self.netD_A(self.fake_A.detach())))
        # self.seg_fake_A = self.netDSeg_A((self.netD_A(self.fake_A)))
        # self.seg_real_B = self.netDSeg_B(self.netD_B(self.real_B))
        # self.seg_fake_A = self.netD_A(self.fake_A.detach())[1]
        # self.seg_real_B = self.netD_B(self.real_B)[1]
        # if not self.isTrain:
        #     self.seg_fake_A = self.netDSeg_A((self.netD_A(self.fake_A.detach()))) # here we use fake As not only from the current batch, but from a larger selection
        #     self.seg_real_B = self.netDSeg_B(self.netD_B(self.real_B))
        if not self.isTrain:
            sigmoid_seg_fake_A = torch.sigmoid(self.netD_B_Seg(self.netD_B(self.fake_A,segment=True)))
            sigmoid_seg_fake_A_with_A = torch.sigmoid(self.netD_A_Seg(self.netD_A(self.fake_A,segment=True)))
            sigmoid_seg_real_B = torch.sigmoid(self.netD_A_Seg(self.netD_A(self.real_B,segment=True)))
            sigmoid_seg_fake_B = torch.sigmoid(self.netD_A_Seg(self.netD_A(self.fake_B,segment=True)))
            self.seg_fake_A = torch.where(sigmoid_seg_fake_A>0.6,1,0)
            self.seg_real_B = torch.where(sigmoid_seg_real_B>0.6,1,0)
            self.seg_fake_B = torch.where(sigmoid_seg_fake_B>0.6,1,0)
            self.seg_fake_A_with_A = torch.where(sigmoid_seg_fake_A_with_A>0.6,1,0)

            #self.seg_fake_A = self.netD_B_Seg(self.netD_B(self.fake_A,segment=True)) 
            #self.seg_real_B = self.netD_A_Seg(self.netD_A(self.real_B,segment=True))
            #self.seg_fake_B = self.netD_A_Seg(self.netD_A(self.fake_B,segment=True))
            
        

    def backward_D_basic(self, netD, real, fake, seg_loss):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        # loss_D = (loss_D_real + loss_D_fake) * 0.5 + self.loss_DSeg # or minus
        loss_D = (loss_D_real + loss_D_fake) * 0.5 + seg_loss
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        real_B_for_D, real_BSeg_for_D = self.real_B_pool.query(self.real_B,self.real_BSeg)
        # segmentation loss
        seg_real_B = self.netD_A_Seg(self.netD_A(real_B_for_D,segment=True))
        loss_DSeg = self.criterionDSeg(seg_real_B, real_BSeg_for_D)

        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B, loss_DSeg)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        self.fake_A_for_D, self.real_BSeg_for_D = self.fake_A_pool.query(self.fake_A,self.real_BSeg)
        # segmentation loss
        seg_fake_A = self.netD_B_Seg(self.netD_B(self.fake_A_for_D,segment=True))
        loss_DSeg = self.criterionDSeg(seg_fake_A, self.real_BSeg_for_D)

        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, self.fake_A_for_D, loss_DSeg)


    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # Segmentation loss
        self.seg_real_B = self.netD_A_Seg(self.netD_A(self.real_B,segment=True))
        self.seg_fake_A = self.netD_B_Seg(self.netD_B(self.fake_A,segment=True))
        self.loss_DSeg = self.criterionDSeg(self.seg_fake_A, self.real_BSeg) + self.criterionDSeg(self.seg_real_B, self.real_BSeg)
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_DSeg
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        with torch.autograd.set_detect_anomaly(True):
            # forward
            self.forward()      # compute fake images and reconstruction images.
            # self.loss_DSeg_A = self.criterionDSeg(self.seg_fake_A, self.real_BSeg)
            # self.loss_DSeg_B = self.criterionDSeg(self.seg_real_B, self.real_BSeg)
            # self.loss_DSeg = self.loss_DSeg_A + self.loss_DSeg_B
            # G_A and G_B
            self.set_requires_grad([self.netD_A, self.netD_B, self.netD_A_Seg, self.netD_B_Seg], False)  # Ds require no gradients when optimizing Gs
            self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
            self.backward_G()             # calculate gradients for G_A and G_B
            self.optimizer_G.step()       # update G_A and G_B's weights

            # D_A and D_B
            self.set_requires_grad([self.netD_A, self.netD_B, self.netD_A_Seg, self.netD_B_Seg], True)
            self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
            self.backward_D_A()      # calculate gradients for D_A
            self.backward_D_B()      # calculate gradients for D_A
            self.optimizer_D.step()  # update D_A and D_B's weights

        # DSeg_A and DSeg_B
        # self.set_requires_grad([self.netD_A, self.netD_B, self.netG_A, self.netG_B], False)  # Ds encoders require no gradients when optimizing Ds decoders ? 
        # self.optimizer_DSeg.zero_grad()  # set DSeg_A and DSeg_B's gradients to zero
        # self.backward_DSeg_A()           # calculate gradients for DSeg_A
        # self.backward_DSeg_B()           # calculate gradients for DSeg_B
        # self.optimizer_DSeg.step()       # update DSeg_A and DSeg_B's weights
        # self.set_requires_grad([self.netD_A, self.netD_B, self.netG_A, self.netG_B], True)

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        self.real_B_RGB = self.real_B[:,:3,...]
        self.real_A_RGB = self.real_A[:,:3,...]
        self.fake_B_RGB = self.fake_B[:,:3,...]
        self.fake_A_RGB = self.fake_A[:,:3,...]
        self.real_B_Depth = (self.real_B[:,3,...]).unsqueeze(1)
        self.real_A_Depth = (self.real_A[:,3,...]).unsqueeze(1)
        self.fake_B_Depth = (self.fake_B[:,3,...]).unsqueeze(1)
        self.fake_A_Depth = (self.fake_A[:,3,...]).unsqueeze(1)
        if not self.isTrain:
            self.pred_fakeBSeg   = self.seg_fake_B
            self.pred_fakeASegDA = self.seg_fake_A_with_A
            self.pred_fakeASeg   = self.seg_fake_A
            self.pred_realBSeg   = self.seg_real_B
            self.real_Seg_for_fake_A = self.real_BSeg
        else:
            self.pred_fakeASeg = torch.where(torch.sigmoid(self.seg_fake_A)>0.6,1,0)
            self.pred_realBSeg = torch.where(torch.sigmoid(self.seg_real_B)>0.6,1,0)
