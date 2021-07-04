import argparse
import os
import torch


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--init_type', type=str, default='xavier',
                                 help='network initialization [normal|xavier|kaiming|orthogonal]')
        self.parser.add_argument('--dataroot', type=str, default='MT-Dataset/images', help='path to images')
        self.parser.add_argument('--dirmap', type=str, default='MT-Dataset/parsing', help='path to parsing maps)')
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--img_size', type=int, default=256, help='img size')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        self.parser.add_argument('--d_conv_dim', type=int, default=64)
        self.parser.add_argument('--d_repeat_num', type=int, default=3)
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--gpu_ids', type=str, default='0')
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--norm1', type=str, default='SN',
                                 help='instance normalization or batch normalization')
        self.parser.add_argument('--serial_batches', action='store_false', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--n_componets', type=int, default=3, help='# of componets')
        self.parser.add_argument('--n_res', type=int, default=3, help='# of resblocks')
        self.parser.add_argument('--padding_type', type=str, default='reflect')
        self.parser.add_argument('--use_flip', type=int, default=0, help='flip or not')
        self.parser.add_argument('--n_downsampling', type=int, default=2, help='down-sampling blocks')
        self.parser.add_argument('--style_dim', type=int, default=192, help='dim of z')
        self.parser.add_argument('--mlp_dim', type=int, default=256, help='# of hidden units')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()


        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        
        return self.opt
