import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from .base_model import BaseModel
from . import networks


class PanModel(BaseModel):
    def name(self):
        return 'PanModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.isTrain = opt.isTrain

        # define tensors
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize)

        # load/define networks
        # init_type: he(kaiming) or xavier
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            # train from pre-trained weights
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:

            # hyper parameters
            self.pan_lambdas = opt.pan_lambdas
            self.pan_mergin_m = opt.pan_mergin_m

            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = networks.GANLoss(tensor=self.Tensor)
            self.criterionPAN = torch.nn.L1Loss()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.fake_B = self.netG.forward(self.real_A)
        self.real_B = Variable(self.input_B)

    # no backprop gradients
    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG.forward(self.real_A)
        self.real_B = Variable(self.input_B, volatile=True)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        # Fake
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        #fake_AB --> torch.Size([1, 6, 256, 256])

        # detach: make fake_AB volatile
        self.pred_fake = self.netD.forward(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)

        # outputs of intermediate layers
        fake_inters = self.netD.get_intermediate_outputs()

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        self.pred_real = self.netD.forward(real_AB)
        self.loss_D_real = self.criterionGAN(self.pred_real, True)

        # outputs of intermediate layers
        real_inters = self.netD.get_intermediate_outputs()

        # calc Parceptual Adversarial Loss
        self.loss_PAN = 0
        for (fake_i, real_i, lam) in zip(fake_inters, real_inters, self.pan_lambdas):
            self.loss_PAN += self.criterionPAN(fake_i, real_i) * lam

        if self.loss_PAN.data[0] > self.pan_mergin_m:
            loss_PAN = Variable(self.Tensor(np.array([0], dtype=np.float)), requires_grad=False)
        else:
            loss_PAN = Variable(self.Tensor(np.array([self.pan_mergin_m], dtype=np.float)), requires_grad=False) - self.loss_PAN

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 + loss_PAN

        self.loss_D.backward()

    def backward_G(self, retain=False):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD.forward(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Combined loss
        self.loss_G = self.loss_G_GAN + self.loss_PAN

        self.loss_G.backward(retain_graph=retain)

    def optimize_parameters(self):
        self.forward()

        # update D
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # # update G 3 times
        # for i in range(3):
        #     self.optimizer_G.zero_grad()
        #     if i == 2:
        #         self.backward_G(retain=False)
        #     else:
        #         self.backward_G(retain=True)
        #     self.optimizer_G.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.data[0]),
                            ('PAN', self.loss_PAN.data[0]),
                            ('D_real', self.loss_D_real.data[0]),
                            ('D_fake', self.loss_D_fake.data[0])
                            ])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)
