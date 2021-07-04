import os
import torch
import torch.nn as nn
import torchvision.models.vgg as models


class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()

    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.vgg_path = os.path.join('vgg.pth')


        self.device = "cuda"
    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass





