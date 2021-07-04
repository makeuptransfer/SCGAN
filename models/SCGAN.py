import numpy as np
import torch
import os
import os.path as osp
from torch.autograd import Variable
from torchvision.utils import save_image
from .base_model import BaseModel
from . import net_utils
from .SCDis import SCDis
from .vgg import VGG
from .losses import GANLoss, HistogramLoss
from models.SCGen import SCGen

class SCGAN(BaseModel):
    def name(self):
        return 'SCGAN'
    def __init__(self,dataset):
        super(SCGAN, self).__init__()
        self.dataloader = dataset

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.layers=['r41']
        self.phase=opt.phase
        self.lips = True
        self.eye = True
        self.skin = True
        self.num_epochs = opt.num_epochs
        self.num_epochs_decay = opt.epochs_decay
        self.g_lr = opt.g_lr
        self.d_lr = opt.d_lr
        self.g_step = opt.g_step
        self.beta1 = opt.beta1
        self.beta2 = opt.beta2
        self.img_size = opt.img_size
        self.lambda_idt = opt.lambda_idt
        self.lambda_A = opt.lambda_A
        self.lambda_B = opt.lambda_B
        self.lambda_his_lip = opt.lambda_his_lip
        self.lambda_his_skin_1 = opt.lambda_his_skin
        self.lambda_his_skin_2 = opt.lambda_his_skin
        self.lambda_his_eye = opt.lambda_his_eye
        self.lambda_vgg = opt.lambda_vgg
        self.snapshot_step = opt.snapshot_step
        self.save_step = opt.save_step
        self.log_step = opt.log_step
        self.result_path = opt.save_path
        self.snapshot_path = opt.snapshot_path
        self.d_conv_dim = opt.d_conv_dim
        self.d_repeat_num = opt.d_repeat_num
        self.norm1 = opt.norm1
        self.mask_A = {}
        self.mask_B = {}
        self.ispartial=opt.partial
        self.isinterpolation=opt.interpolation
        self.SCGen = SCGen(opt.ngf, opt.style_dim, opt.n_downsampling, opt.n_res, opt.mlp_dim, opt.n_componets,
                         opt.input_nc,  opt.phase,  ispartial=opt.partial, isinterpolation=opt.interpolation)
        self.D_A = SCDis(self.img_size, self.d_conv_dim, self.d_repeat_num, self.norm1)
        self.D_B = SCDis(self.img_size, self.d_conv_dim, self.d_repeat_num, self.norm1)

        self.D_A.apply(net_utils.weights_init_xavier)
        self.D_B.apply(net_utils.weights_init_xavier)
        self.SCGen.apply(net_utils.weights_init_xavier)
        self.load_checkpoint()
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionL2 = torch.nn.MSELoss()

        self.criterionGAN = GANLoss(use_lsgan=True, tensor=torch.cuda.FloatTensor)
        self.vgg = VGG()
        if self.phase == 'train':
            self.vgg.load_state_dict(torch.load('vgg_conv.pth'))
        self.criterionHis = HistogramLoss()

        self.g_optimizer = torch.optim.Adam(self.SCGen.parameters(), self.g_lr, [opt.beta1, opt.beta2])
        self.d_A_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D_A.parameters()), opt.d_lr,
                                              [self.beta1, self.beta2])
        self.d_B_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D_B.parameters()), opt.d_lr,
                                              [opt.beta1, opt.beta2])
        self.SCGen.cuda()
        self.vgg.cuda()
        self.criterionHis.cuda()
        self.criterionGAN.cuda()
        self.criterionL1.cuda()
        self.criterionL2.cuda()
        self.D_A.cuda()
        self.D_B.cuda()

        print('---------- Networks initialized -------------')
        net_utils.print_network(self.SCGen)

    def load_checkpoint(self):
        G_path = os.path.join(self.snapshot_path ,'G.pth')
        if os.path.exists(G_path):
            dict=torch.load(G_path)
            self.SCGen.load_state_dict(dict)
            print('loaded trained generator {}..!'.format(G_path))
        D_A_path = os.path.join(self.snapshot_path, 'D_A.pth')
        if os.path.exists(D_A_path):
            self.D_A.load_state_dict(torch.load(D_A_path))
            print('loaded trained discriminator A {}..!'.format(D_A_path))

        D_B_path = os.path.join(self.snapshot_path, 'D_B.pth')
        if os.path.exists(D_B_path):
            self.D_B.load_state_dict(torch.load(D_B_path))
            print('loaded trained discriminator B {}..!'.format(D_B_path))


    def set_input(self, input):
        self.mask_A=input['mask_A']
        self.mask_B=input['mask_B']
        makeup=input['makeup_img']
        nonmakeup=input['nonmakeup_img']
        makeup_seg=input['makeup_seg']
        nonmakeup_seg=input['nonmakeup_seg']
        self.makeup=makeup
        self.nonmakeup=nonmakeup
        self.makeup_seg=makeup_seg
        self.nonmakeup_seg=nonmakeup_seg
        self.makeup_unchanged=input['makeup_unchanged']
        self.nonmakeup_unchanged=input['nonmakeup_unchanged']




    def to_var(self, x, requires_grad=False):
        if isinstance(x, list):
            return x
        if torch.cuda.is_available():
            x = x.cuda()
        if not requires_grad:
            return Variable(x, requires_grad=requires_grad)
        else:
            return Variable(x)


    def train(self):
        # forward
        self.iters_per_epoch = len(self.dataloader)
        g_lr = self.g_lr
        d_lr = self.d_lr
        start = 0

        for self.e in range(start, self.num_epochs):
            for self.i, data in enumerate(self.dataloader):

                if (len(data) == 0):
                    print("No eyes!!")
                    continue
                self.set_input(data)
                makeup, nonmakeup = self.to_var(self.makeup), self.to_var(self.nonmakeup),
                makeup_seg, nonmakeup_seg = self.to_var(self.makeup_seg), self.to_var(self.nonmakeup_seg)
                # makeup_unchanged=self.to_var(self.makeup_unchanged)
                # nonmakeup_unchanged=self.to_var(self.nonmakeup_unchanged)
                mask_makeup = {key: self.to_var(self.mask_B[key]) for key in self.mask_B}
                mask_nonmakeup = {key: self.to_var(self.mask_A[key]) for key in self.mask_A}

                # ================== Train D ================== #
                # training D_A, D_A aims to distinguish class B
                # Real
                out = self.D_A(makeup)

                d_loss_real = self.criterionGAN(out, True)

                # Fake
                fake_makeup = self.SCGen(nonmakeup, nonmakeup_seg, makeup, makeup_seg, makeup, makeup_seg)

                fake_makeup = Variable(fake_makeup.data).detach()
                out = self.D_A(fake_makeup)

                d_loss_fake = self.criterionGAN(out, False)


                # Backward + Optimize
                d_loss = (d_loss_real.mean() + d_loss_fake.mean()) * 0.5
                self.d_A_optimizer.zero_grad()
                d_loss.backward(retain_graph=False)
                self.d_A_optimizer.step()

                # Logging
                self.loss = {}
                self.loss['D-A-loss_real'] = d_loss_real.mean().item()

                # training D_B, D_B aims to distinguish class A
                # Real
                out = self.D_B(nonmakeup)
                d_loss_real = self.criterionGAN(out, True)
                # Fake

                fake_nonmakeup = self.SCGen(makeup, makeup_seg, nonmakeup, nonmakeup_seg, nonmakeup, nonmakeup_seg)

                fake_nonmakeup = Variable(fake_nonmakeup.data).detach()
                out = self.D_B(fake_nonmakeup)
                d_loss_fake = self.criterionGAN(out, False)

                # Backward + Optimize
                d_loss = (d_loss_real.mean() + d_loss_fake.mean()) * 0.5
                self.d_B_optimizer.zero_grad()
                d_loss.backward(retain_graph=False)
                self.d_B_optimizer.step()

                # Logging
                self.loss['D-B-loss_real'] = d_loss_real.mean().item()

                # ================== Train G ================== #
                if (self.i + 1) % self.g_step == 0:
                    # identity loss
                    assert self.lambda_idt > 0

                    # G should be identity if ref_B or org_A is fed
                    idt_A = self.SCGen(makeup, makeup_seg, makeup, makeup_seg, makeup, makeup_seg)
                    idt_B = self.SCGen(nonmakeup, nonmakeup_seg, nonmakeup, nonmakeup_seg,nonmakeup, nonmakeup_seg)
                    loss_idt_A = self.criterionL1(idt_A, makeup) * self.lambda_A * self.lambda_idt
                    loss_idt_B = self.criterionL1(idt_B, nonmakeup) * self.lambda_B * self.lambda_idt
                    # loss_idt
                    loss_idt = (loss_idt_A + loss_idt_B) * 0.5
                    # loss_idt = loss_idt_A * 0.5


                    # GAN loss D_A(G_A(A))
                    # fake_A in class B,
                    fake_makeup = self.SCGen(nonmakeup, nonmakeup_seg, makeup, makeup_seg, makeup, makeup_seg)
                    pred_fake = self.D_A(fake_makeup)
                    g_A_loss_adv = self.criterionGAN(pred_fake, True)

                    # GAN loss D_B(G_B(B))
                    fake_nonmakeup = self.SCGen(makeup, makeup_seg, nonmakeup, nonmakeup_seg, nonmakeup, nonmakeup_seg)
                    pred_fake = self.D_B(fake_nonmakeup)
                    g_B_loss_adv = self.criterionGAN(pred_fake, True)



                    # histogram loss
                    g_A_loss_his = 0
                    g_B_loss_his = 0
                    if self.lips == True:
                        g_A_lip_loss_his = self.criterionHis(fake_makeup, makeup, mask_nonmakeup["mask_A_lip"],
                                                             mask_makeup['mask_B_lip'],
                                                             mask_nonmakeup["index_A_lip"],
                                                             nonmakeup) * self.lambda_his_lip
                        g_B_lip_loss_his = self.criterionHis(fake_nonmakeup, nonmakeup, mask_makeup["mask_B_lip"],
                                                             mask_nonmakeup['mask_A_lip'],
                                                             mask_makeup["index_B_lip"], makeup) * self.lambda_his_lip
                        g_A_loss_his += g_A_lip_loss_his
                        g_B_loss_his += g_B_lip_loss_his
                    if self.skin == True:
                        g_A_skin_loss_his = self.criterionHis(fake_makeup, makeup, mask_nonmakeup["mask_A_skin"],
                                                              mask_makeup['mask_B_skin'],
                                                              mask_nonmakeup["index_A_skin"],
                                                              nonmakeup) * self.lambda_his_skin_1
                        g_B_skin_loss_his = self.criterionHis(fake_nonmakeup, nonmakeup, mask_makeup["mask_B_skin"],
                                                              mask_nonmakeup['mask_A_skin'],
                                                              mask_makeup["index_B_skin"],
                                                              makeup) * self.lambda_his_skin_2
                        g_A_loss_his += g_A_skin_loss_his
                        g_B_loss_his += g_B_skin_loss_his
                    if self.eye == True:
                        g_A_eye_left_loss_his = self.criterionHis(fake_makeup, makeup,
                                                                  mask_nonmakeup["mask_A_eye_left"],
                                                                  mask_makeup["mask_B_eye_left"],
                                                                  mask_nonmakeup["index_A_eye_left"],
                                                                  nonmakeup) * self.lambda_his_eye
                        g_B_eye_left_loss_his = self.criterionHis(fake_nonmakeup, nonmakeup,
                                                                  mask_makeup["mask_B_eye_left"],
                                                                  mask_nonmakeup["mask_A_eye_left"],
                                                                  mask_makeup["index_B_eye_left"],
                                                                  makeup) * self.lambda_his_eye
                        g_A_eye_right_loss_his = self.criterionHis(fake_makeup, makeup,
                                                                   mask_nonmakeup["mask_A_eye_right"],
                                                                   mask_makeup["mask_B_eye_right"],
                                                                   mask_nonmakeup["index_A_eye_right"],
                                                                   nonmakeup) * self.lambda_his_eye
                        g_B_eye_right_loss_his = self.criterionHis(fake_nonmakeup, nonmakeup,
                                                                   mask_makeup["mask_B_eye_right"],
                                                                   mask_nonmakeup["mask_A_eye_right"],
                                                                   mask_makeup["index_B_eye_right"],
                                                                   makeup) * self.lambda_his_eye
                        g_A_loss_his += g_A_eye_left_loss_his + g_A_eye_right_loss_his
                        g_B_loss_his += g_B_eye_left_loss_his + g_B_eye_right_loss_his



                    # cycle loss
                    rec_A = self.SCGen(fake_makeup, nonmakeup_seg, nonmakeup, nonmakeup_seg, nonmakeup, nonmakeup_seg)
                    rec_B = self.SCGen(fake_nonmakeup, makeup_seg, makeup, makeup_seg, makeup, makeup_seg)

                    g_loss_rec_A = self.criterionL1(rec_A, nonmakeup) * self.lambda_A
                    g_loss_rec_B = self.criterionL1(rec_B, makeup) * self.lambda_B


                    # vgg loss
                    vgg_s = self.vgg(makeup, self.layers)[0]
                    vgg_s = Variable(vgg_s.data).detach()
                    vgg_fake_makeup = self.vgg(fake_makeup, self.layers)[0]
                    g_loss_A_vgg = self.criterionL2(vgg_fake_makeup, vgg_s) * self.lambda_A * self.lambda_vgg


                    vgg_r = self.vgg(nonmakeup, self.layers)[0]
                    vgg_r = Variable(vgg_r.data).detach()
                    vgg_fake_nonmakeup = self.vgg(fake_nonmakeup, self.layers)[0]
                    g_loss_B_vgg = self.criterionL2(vgg_fake_nonmakeup, vgg_r) * self.lambda_B * self.lambda_vgg
                    #local-per
                    # vgg_fake_makeup_unchanged=self.vgg(fake_makeup*nonmakeup_unchanged,self.layers)
                    # vgg_makeup_masked=self.vgg(makeup*makeup_unchanged,self.layers)
                    # vgg_nonmakeup_masked=self.vgg(nonmakeup*nonmakeup_unchanged,self.layers)
                    # vgg_fake_nonmakeup_unchanged=self.vgg(fake_nonmakeup*makeup_unchanged,self.layers)
                    # g_loss_unchanged=(self.criterionL2(vgg_fake_makeup_unchanged, vgg_nonmakeup_masked)+self.criterionL2(vgg_fake_nonmakeup_unchanged,vgg_makeup_masked))

                    loss_rec = (g_loss_rec_A + g_loss_rec_B + g_loss_A_vgg + g_loss_B_vgg) * 0.5


                    # Combined loss
                    g_loss = (g_A_loss_adv + g_B_loss_adv + loss_rec + loss_idt + g_A_loss_his + g_B_loss_his).mean()


                    self.g_optimizer.zero_grad()
                    g_loss.backward(retain_graph=False)
                    self.g_optimizer.step()
                    # self.track("Generator backward")

                    # Logging
                    # self.loss['G-loss-unchanged']=g_loss_unchanged.mean().item()
                    self.loss['G-A-loss-adv'] = g_A_loss_adv.mean().item()
                    self.loss['G-B-loss-adv'] = g_B_loss_adv.mean().item()
                    self.loss['G-loss-org'] = g_loss_rec_A.mean().item()
                    self.loss['G-loss-ref'] = g_loss_rec_B.mean().item()
                    self.loss['G-loss-idt'] = loss_idt.mean().item()
                    self.loss['G-loss-img-rec'] = (g_loss_rec_A + g_loss_rec_B).mean().item()
                    self.loss['G-loss-vgg-rec'] = (g_loss_A_vgg + g_loss_B_vgg).mean().item()
                    self.loss['G-A-loss-his'] = g_A_loss_his.mean().item()

                    # Print out log info
                if (self.i + 1) % self.log_step == 0:
                    self.log_terminal()

                # save the images
                if (self.i) % self.save_step == 0:
                    print("Saving middle output...")
                    self.imgs_save([nonmakeup, makeup, fake_makeup])

                # Save model checkpoints

            # Decay learning rate
            if (self.e + 1) % self.snapshot_step == 0:
                self.save_models()
            if (self.e + 1) > (self.num_epochs - self.num_epochs_decay):
                g_lr -= (self.g_lr / float(self.num_epochs_decay))
                d_lr -= (self.d_lr / float(self.num_epochs_decay))
                self.update_lr(g_lr, d_lr)
                print('Decay learning rate to g_lr: {}, d_lr:{}.'.format(g_lr, d_lr))

    def imgs_save(self, imgs_list):
        if self.phase == 'test':
            length = len(imgs_list)
            for i in range(0, length):
                imgs_list[i] = torch.cat(imgs_list[i], dim=3)
            imgs_list = torch.cat(imgs_list, dim=2)

            if not osp.exists(self.result_path):
                os.makedirs(self.result_path)
            save_path = os.path.join(self.result_path,
                                     '{}{}transferred.jpg'.format("partial_" if self.ispartial else "global_",
                                                                  "interpolation_" if self.isinterpolation else ""))
            save_image(self.de_norm(imgs_list.data), save_path, normalize=True)
        if self.phase == 'train':
            img_train_list = torch.cat(imgs_list, dim=3)
            if not osp.exists(self.result_path):
                os.makedirs(self.result_path)
            save_path = os.path.join(self.result_path, 'train/'+str(self.e)+'_'+str(self.i) + ".jpg")
            save_image(self.de_norm(img_train_list.data), save_path, normalize=True)

    def log_terminal(self):
        log = " Epoch [{}/{}], Iter [{}/{}]".format(
            self.e+1, self.num_epochs, self.i+1, self.iters_per_epoch)

        for tag, value in self.loss.items():
            log += ", {}: {:.4f}".format(tag, value)
        print(log)


    def save_models(self):

        if not osp.exists(self.snapshot_path):
            os.makedirs(self.snapshot_path)
        torch.save(
            self.SCGen.state_dict(),
            os.path.join(
                self.snapshot_path, '{}_{}_G.pth'.format(self.e + 1, self.i + 1)))
        torch.save(
            self.D_A.state_dict(),
            os.path.join(
                self.snapshot_path, '{}_{}_D_A.pth'.format(self.e + 1, self.i + 1)))
        torch.save(
            self.D_B.state_dict(),
            os.path.join(
                self.snapshot_path, '{}_{}_D_B.pth'.format(self.e + 1, self.i + 1)))

    def test(self):
        self.SCGen.eval()
        self.D_A.eval()
        self.D_B.eval()
        makeups = []
        makeups_seg = []
        nonmakeups=[]
        nonmakeups_seg = []
        for self.i, data in enumerate(self.dataloader):
            if (len(data) == 0):
                print("No eyes!!")
                continue
            self.set_input(data)
            makeup, nonmakeup = self.to_var(self.makeup), self.to_var(self.nonmakeup),
            makeup_seg, nonmakeup_seg = self.to_var(self.makeup_seg), self.to_var(self.nonmakeup_seg)
            makeups.append(makeup)
            makeups_seg.append(makeup_seg)
            nonmakeups.append(nonmakeup)
            nonmakeups_seg.append(nonmakeup_seg)
        source, ref1, ref2 = nonmakeups[0], makeups[0], makeups[1]
        source_seg, ref1_seg, ref2_seg = nonmakeups_seg[0], makeups_seg[0], makeups_seg[1]
        with torch.no_grad():
            transfered = self.SCGen(source, source_seg, ref1, ref1_seg, ref2, ref2_seg)
        if not self.ispartial and not self.isinterpolation:
            results = [[source, ref1],
                    [source, ref2],
                    [ref1, source],
                    [ref2, source]
                    ]
            for i, img in zip(range(0, len(results)), transfered):
                results[i].append(img)
            self.imgs_save(results)
        elif not self.ispartial and self.isinterpolation:
            results = [[source, ref1],
                       [source, ref2],
                       [ref1, source],
                       [ref2, source],
                       [ref2, ref1]
                       ]
            for i, imgs in zip(range(0, len(results)-1), transfered):
                for img in imgs:
                    results[i].append(img)
            for img in transfered[-1]:
                results[-1].insert(1, img)
            results[-1].reverse()

            self.imgs_save(results)
        elif self.ispartial and not self.isinterpolation:
            results = [[source, ref1],
                       [source, ref2],
                       [source, ref1, ref2],
                       ]
            for i, imgs in zip(range(0, len(results)), transfered):
                for img in imgs:
                    results[i].append(img)
            self.imgs_save(results)
        elif self.ispartial and self.isinterpolation:
            results = [[source, ref1],
                       [source, ref1],
                       [source, ref1],
                       [source, ref2],
                       [source, ref2],
                       [source, ref2],
                       [ref2, ref1],
                       [ref2, ref1],
                       [ref2, ref1],
                       ]
            for i, imgs in zip(range(0, len(results)-3), transfered):
                for img in imgs:
                    results[i].append(img)

            for i, imgs in zip(range(len(results)-3, len(results)), transfered[-3:]):
                for img in imgs:
                    results[i].insert(1, img)

                results[i].reverse()
            self.imgs_save(results)

    def de_norm(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)