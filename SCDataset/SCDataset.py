import os.path
import torchvision.transforms as transforms

from PIL import Image
import PIL
import numpy as np
import torch
from torch.autograd import Variable


def ToTensor(pic):
    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float()
    else:
        return img


class SCDataset():
    def __init__(self, opt):
        self.random = None
        self.phase=opt.phase
        self.opt = opt
        self.root = opt.dataroot
        self.dir_makeup = opt.dataroot
        self.dir_nonmakeup = opt.dataroot
        self.dir_seg = opt.dirmap  # parsing maps
        self.n_componets = opt.n_componets
        self.makeup_names = []
        self.non_makeup_names = []
        if self.phase == 'train':
            self.makeup_names = [name.strip() for name in
                                 open(os.path.join('MT-Dataset', 'makeup.txt'), "rt").readlines()]
            self.non_makeup_names = [name.strip() for name in
                                     open(os.path.join('MT-Dataset', 'non-makeup.txt'), "rt").readlines()]
        if self.phase == 'test':
            with open("test.txt", 'r') as f:
                for line in f.readlines():
                    non_makeup_name, make_upname = line.strip().split()
                    self.non_makeup_names.append(non_makeup_name)
                    self.makeup_names.append(make_upname)
        self.transform = transforms.Compose([
            transforms.Resize((opt.img_size, opt.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.transform_mask = transforms.Compose([
            transforms.Resize((opt.img_size, opt.img_size), interpolation=PIL.Image.NEAREST),
            ToTensor])

    def __getitem__(self, index):
        if self.phase == 'test':
            makeup_name = self.makeup_names[index]
            nonmakeup_name = self.non_makeup_names[index]
        if self.phase == 'train':
            index = self.pick()
            makeup_name = self.makeup_names[index[0]]
            nonmakeup_name = self.non_makeup_names[index[1]]
        # self.f.write(nonmakeup_name+' '+makeup_name+'\n')
        # self.f.flush()
        nonmakeup_path = os.path.join(self.dir_nonmakeup, nonmakeup_name)


        makeup_path = os.path.join(self.dir_makeup, makeup_name)


        makeup_img = Image.open(makeup_path).convert('RGB')
        nonmakeup_img = Image.open(nonmakeup_path).convert('RGB')


        makeup_seg_img = Image.open(os.path.join(self.dir_seg, makeup_name))
        nonmakeup_seg_img = Image.open(os.path.join(self.dir_seg, nonmakeup_name))
        # makeup_img = makeup_img.transpose(Image.FLIP_LEFT_RIGHT)
        # makeup_seg_img = makeup_seg_img.transpose(Image.FLIP_LEFT_RIGHT)
        # nonmakeup_img=nonmakeup_img.rotate(40)
        # nonmakeup_seg_img=nonmakeup_seg_img.rotate(40)
        # makeup_img=makeup_img.rotate(90)
        # makeup_seg_img=makeup_seg_img.rotate(90)
        makeup_img = self.transform(makeup_img)
        nonmakeup_img = self.transform(nonmakeup_img)
        mask_B = self.transform_mask(makeup_seg_img)  # makeup
        mask_A = self.transform_mask(nonmakeup_seg_img)  # nonmakeup
        makeup_seg = torch.zeros([self.n_componets, 256, 256], dtype=torch.float)
        nonmakeup_seg = torch.zeros([self.n_componets, 256, 256], dtype=torch.float)
        makeup_unchanged = (mask_B == 7).float() + (mask_B == 2).float() + (mask_B == 6).float() + (
                    mask_B == 1).float() + (mask_B == 11).float()
        nonmakeup_unchanged = (mask_A == 7).float() + (mask_A == 2).float() + (mask_A == 6).float() + (
                mask_A == 1).float() + (mask_A == 11).float()
        mask_A_lip = (mask_A == 9).float() + (mask_A == 13).float()
        mask_B_lip = (mask_B == 9).float() + (mask_B == 13).float()
        mask_A_lip, mask_B_lip, index_A_lip, index_B_lip = self.mask_preprocess(mask_A_lip, mask_B_lip)
        makeup_seg[0] = mask_B_lip
        nonmakeup_seg[0] = mask_A_lip
        mask_A_skin = (mask_A == 4).float() + (mask_A == 8).float() + (mask_A == 10).float()
        mask_B_skin = (mask_B == 4).float() + (mask_B == 8).float() + (mask_B == 10).float()
        mask_A_skin, mask_B_skin, index_A_skin, index_B_skin = self.mask_preprocess(mask_A_skin, mask_B_skin)
        makeup_seg[1] = mask_B_skin
        nonmakeup_seg[1] = mask_A_skin
        mask_A_eye_left = (mask_A == 6).float()
        mask_A_eye_right = (mask_A == 1).float()
        mask_B_eye_left = (mask_B == 6).float()
        mask_B_eye_right = (mask_B == 1).float()
        mask_A_face = (mask_A == 4).float() + (mask_A == 8).float()
        mask_B_face = (mask_B == 4).float() + (mask_B == 8).float()
        # avoid the es of ref are closed
        if not ((mask_B_eye_left > 0).any() and \
                (mask_B_eye_right > 0).any()):
            return {}
        # mask_A_eye_left, mask_A_eye_right = self.rebound_box(mask_A_eye_left, mask_A_eye_right, mask_A_face)
        mask_B_eye_left, mask_B_eye_right = self.rebound_box(mask_B_eye_left, mask_B_eye_right, mask_B_face)
        mask_A_eye_left, mask_B_eye_left, index_A_eye_left, index_B_eye_left = \
            self.mask_preprocess(mask_A_eye_left, mask_B_eye_left)
        mask_A_eye_right, mask_B_eye_right, index_A_eye_right, index_B_eye_right = \
            self.mask_preprocess(mask_A_eye_right, mask_B_eye_right)
        makeup_seg[2] = mask_B_eye_left + mask_B_eye_right
        nonmakeup_seg[2] = mask_A_eye_left + mask_A_eye_right

        mask_A = {}
        mask_A["mask_A_eye_left"] = mask_A_eye_left
        mask_A["mask_A_eye_right"] = mask_A_eye_right
        mask_A["index_A_eye_left"] = index_A_eye_left
        mask_A["index_A_eye_right"] = index_A_eye_right
        mask_A["mask_A_skin"] = mask_A_skin
        mask_A["index_A_skin"] = index_A_skin
        mask_A["mask_A_lip"] = mask_A_lip
        mask_A["index_A_lip"] = index_A_lip

        mask_B = {}
        mask_B["mask_B_eye_left"] = mask_B_eye_left
        mask_B["mask_B_eye_right"] = mask_B_eye_right
        mask_B["index_B_eye_left"] = index_B_eye_left
        mask_B["index_B_eye_right"] = index_B_eye_right
        mask_B["mask_B_skin"] = mask_B_skin
        mask_B["index_B_skin"] = index_B_skin
        mask_B["mask_B_lip"] = mask_B_lip
        mask_B["index_B_lip"] = index_B_lip
        return {'nonmakeup_seg': nonmakeup_seg, 'makeup_seg': makeup_seg, 'nonmakeup_img': nonmakeup_img,
                'makeup_img': makeup_img,
                'mask_A': mask_A, 'mask_B': mask_B,
                'makeup_unchanged': makeup_unchanged,
                'nonmakeup_unchanged': nonmakeup_unchanged
                }

    def pick(self):
        if self.random is None:
            self.random = np.random.RandomState(np.random.seed())
        a_index = self.random.randint(0, len(self.makeup_names))
        another_index = self.random.randint(0, len(self.non_makeup_names))
        return [a_index, another_index]

    def __len__(self):
        if self.opt.phase == 'train':
            return len(self.non_makeup_names)
        elif self.opt.phase == 'test':
            return len(self.makeup_names)

    def name(self):
        return 'SCDataset'



    def rebound_box(self, mask_A, mask_B, mask_A_face):
        mask_A = mask_A.unsqueeze(0)
        mask_B = mask_B.unsqueeze(0)
        mask_A_face = mask_A_face.unsqueeze(0)

        index_tmp = torch.nonzero(mask_A, as_tuple=False)
        x_A_index = index_tmp[:, 2]
        y_A_index = index_tmp[:, 3]
        index_tmp = torch.nonzero(mask_B, as_tuple=False)
        x_B_index = index_tmp[:, 2]
        y_B_index = index_tmp[:, 3]
        mask_A_temp = mask_A.copy_(mask_A)
        mask_B_temp = mask_B.copy_(mask_B)
        mask_A_temp[:, :, min(x_A_index) - 5:max(x_A_index) + 6, min(y_A_index) - 5:max(y_A_index) + 6] = \
            mask_A_face[:, :, min(x_A_index) - 5:max(x_A_index) + 6, min(y_A_index) - 5:max(y_A_index) + 6]
        mask_B_temp[:, :, min(x_B_index) - 5:max(x_B_index) + 6, min(y_B_index) - 5:max(y_B_index) + 6] = \
            mask_A_face[:, :, min(x_B_index) - 5:max(x_B_index) + 6, min(y_B_index) - 5:max(y_B_index) + 6]
        # mask_A_temp = self.to_var(mask_A_temp, requires_grad=False)
        # mask_B_temp = self.to_var(mask_B_temp, requires_grad=False)
        mask_A_temp = mask_A_temp.squeeze(0)
        mask_A = mask_A.squeeze(0)
        mask_B = mask_B.squeeze(0)
        mask_A_face = mask_A_face.squeeze(0)
        mask_B_temp = mask_B_temp.squeeze(0)

        return mask_A_temp, mask_B_temp

    def mask_preprocess(self, mask_A, mask_B):
        mask_A = mask_A.unsqueeze(0)
        mask_B = mask_B.unsqueeze(0)
        index_tmp = torch.nonzero(mask_A, as_tuple=False)
        x_A_index = index_tmp[:, 2]

        y_A_index = index_tmp[:, 3]
        index_tmp = torch.nonzero(mask_B, as_tuple=False)
        x_B_index = index_tmp[:, 2]
        y_B_index = index_tmp[:, 3]
        # mask_A = self.to_var(mask_A, requires_grad=False)
        # mask_B = self.to_var(mask_B, requires_grad=False)
        index = [x_A_index, y_A_index, x_B_index, y_B_index]
        index_2 = [x_B_index, y_B_index, x_A_index, y_A_index]
        mask_A = mask_A.squeeze(0)
        mask_B = mask_B.squeeze(0)
        return mask_A, mask_B, index, index_2

    def to_var(self, x, requires_grad=True):
        if torch.cuda.is_available():
            x = x.cuda()
        if not requires_grad:
            return Variable(x, requires_grad=requires_grad)
        else:
            return Variable(x)


class SCDataLoader():
    def __init__(self, opt):
        self.dataset = SCDataset(opt)
        print("Dataset loaded")
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))


    def name(self):
        return 'SCDataLoader'


    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data
