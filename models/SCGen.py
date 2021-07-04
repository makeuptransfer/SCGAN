import torch.nn as nn
import torch
import torch.nn.functional as F
import os
import torchvision.models.vgg as models


class SCGen(nn.Module):

    def __init__(self, dim, style_dim, n_downsample, n_res, mlp_dim, n_componets, input_dim,  phase='train',activ='relu',
                 pad_type='reflect', ispartial=False, isinterpolation=False):
        super(SCGen, self).__init__()
        self.phase = phase
        self.ispartial = ispartial
        self.isinterpolation = isinterpolation
        self.PSEnc = PartStyleEncoder(input_dim, dim, int(style_dim / n_componets), norm='none', activ=activ,
                                      pad_type=pad_type, \
                                      phase=self.phase, ispartial=self.ispartial, isinterpolation=self.isinterpolation)
        self.FIEnc = FaceEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
        self.MFDec = MakeupFuseDecoder(n_downsample, n_res, self.FIEnc.output_dim, input_dim, res_norm='adain',
                                       activ=activ, pad_type=pad_type)
        self.MLP = MLP(style_dim, self.get_num_adain_params(self.MFDec), mlp_dim, 3, norm='none', activ=activ)

    def forward(self, x, map_x, y1, map_y1, y2, map_y2):
        if self.phase == 'train':
            fid_x = self.FIEnc(x)
            if self.ispartial or self.isinterpolation:
                exit()
            code = self.PSEnc(y1, map_y1, y1, map_y1, y1, map_y1)
            result = self.fuse(fid_x, code, code)
            return result

        if self.phase == 'test':
            fid_x = self.FIEnc(x)
            fid_y1 = self.FIEnc(y1)
            fid_y2 = self.FIEnc(y2)
            # global
            if not self.ispartial and not self.isinterpolation:
                results = []
                codes = self.PSEnc(y1, map_y1, y2, map_y2, x, map_x)
                fids = [fid_x, fid_x, fid_y1, fid_y2]
                codes.append(codes[-1])  # for demakeup
                length = len(fids)
                for i in range(0, length):
                    result = self.fuse(fids[i], codes[i], codes[i])
                    results.append(result)

                return results
            # global interpolation

            elif not self.ispartial and self.isinterpolation:

                resultss = []
                codes = self.PSEnc(y1, map_y1, y2, map_y2, x, map_x)
                fids = [fid_x, fid_x, fid_y2, fid_y1]
                length = len(fids)
                # shade control makeup
                for i in range(0, 2):
                    code_x = codes[-1]
                    code_y = codes[i]
                    results = []
                    for a in range(0, 11, 1):
                        a = a / 10
                        result = self.fuse(fids[i], code_x, code_y, a)
                        results.append(result)
                    resultss.append(results)
                # shade control demakeup
                for i in range(0, 2):
                    code_x = codes[i]
                    code_y = codes[-1]
                    results = []
                    for a in range(0, 11, 1):
                        a = a / 10
                        result = self.fuse(fids[length - i - 1], code_x, code_y, a)
                        results.append(result)
                    resultss.append(results)
                # interpolation between two refs
                code_x = codes[0]
                code_y = codes[1]
                results = []
                for a in range(0, 11, 1):
                    a = a / 10
                    result = self.fuse(fids[0], code_x, code_y, a)
                    results.append(result)
                resultss.append(results)
                return resultss
            elif self.ispartial and not self.isinterpolation:
                codes = self.PSEnc(x, map_x, y1, map_y1, y2, map_y2)
                resultss = []
                for i in range(0, 2):
                    results = []
                    for j in range(0, 3):
                        results.append(self.fuse(fid_x, codes[i * 3 + j], codes[i * 3 + j]))
                    resultss.append(results)
                results = []
                for i in range(6, 8):
                    results.append(self.fuse(fid_x, codes[i], codes[i]))
                resultss.append(results)
                return resultss
            elif self.ispartial and self.isinterpolation:
                codes = self.PSEnc(x, map_x, y1, map_y1, y2, map_y2)
                resultss = []
                for i in range(0, 2):
                    for j in range(0, 3):
                        results = []
                        for a in range(0, 11, 1):
                            a = a / 10
                            result = self.fuse(fid_x, codes[-1], codes[i * 3 + j], a)
                            results.append(result)
                        resultss.append(results)
                for i in range(0, 3):
                    results = []
                    for a in range(0, 11, 1):
                        a = a / 10
                        result = self.fuse(fid_x, codes[i], codes[i + 3], a)
                        results.append(result)
                    resultss.append(results)
                return resultss


    def fuse(self, content, style0, style1, a=0):
        adain_params = self.MLP(style0, style1, a)
        self.assign_adain_params(adain_params, self.MFDec)
        images = self.MFDec(content)
        return images

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2 * m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2 * m.num_features
        return num_adain_params


class PartStyleEncoder(nn.Module):
    def __init__(self, input_dim, dim, style_dim, norm, activ, pad_type, phase, ispartial, isinterpolation):
        super(PartStyleEncoder, self).__init__()
        self.phase = phase
        self.isinterpolation = isinterpolation
        self.ispartial = ispartial
        vgg19 = models.vgg19(pretrained=False)
        vgg19.load_state_dict(torch.load('./vgg.pth'))
        self.vgg = vgg19.features

        for param in self.vgg.parameters():
            param.requires_grad_(False)

        self.conv1 = ConvBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)  # 3->64,concat
        dim = dim * 2
        self.conv2 = ConvBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)  # 128->128
        dim = dim * 2
        self.conv3 = ConvBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)  # 256->256
        dim = dim * 2
        self.conv4 = ConvBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)  # 512->512
        dim = dim * 2

        self.model = []
        self.model += [nn.AdaptiveAvgPool2d(1)]  # global average pooling
        self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def get_features(self, image, model, layers=None):
        if layers is None:
            layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1', '19': 'conv4_1'}
        features = {}
        x = image
        # model._modules is a dictionary holding each module in the model
        for name, layer in model._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

    def componet_enc(self, x):
        vgg_aux = self.get_features(x, self.vgg)
        x = self.conv1(x)
        x = torch.cat([x, vgg_aux['conv1_1']], dim=1)
        x = self.conv2(x)
        x = torch.cat([x, vgg_aux['conv2_1']], dim=1)
        x = self.conv3(x)
        x = torch.cat([x, vgg_aux['conv3_1']], dim=1)
        x = self.conv4(x)
        x = torch.cat([x, vgg_aux['conv4_1']], dim=1)
        x = self.model(x)
        return x

    def forward(self, x, map_x, y1, map_y1, y2, map_y2):
        if self.phase == 'test':
            # global
            if not self.ispartial:
                codes = []
                refs = [[x, map_x], [y1, map_y1], [y2, map_y2]]
                for k in range(0, 3):
                    for i in range(0, map_x.size(1)):
                        yi = refs[k][1][:, i, :, :]
                        yi = torch.unsqueeze(yi, 1).repeat(1, x.size(1), 1, 1)
                        yi = refs[k][0].mul(yi)
                        if i == 0:
                            code = self.componet_enc(yi)
                        else:
                            code = torch.cat([code, self.componet_enc(yi)], dim=1)
                    codes.append(code)

                return codes
            else:
                # partial
                codes = []
                refs = [[y1, map_y1], [y2, map_y2]]
                # one ref
                for k in range(0, 2):
                    for part in range(0, 3):
                        for i in range(0, map_x.size(1)):
                            if i == part:
                                yi = refs[k][1][:, i, :, :]
                                yi = torch.unsqueeze(yi, 1).repeat(1, x.size(1), 1, 1)
                                yi = refs[k][0].mul(yi)
                            else:
                                yi = map_x[:, i, :, :]
                                yi = torch.unsqueeze(yi, 1).repeat(1, x.size(1), 1, 1)
                                yi = x.mul(yi)
                            if i == 0:
                                code = self.componet_enc(yi)
                            else:
                                code = torch.cat([code, self.componet_enc(yi)], dim=1)
                        codes.append(code)
                # two refs
                for k in range(0, 2):
                    for i in range(0, map_x.size(1)):
                        if i == 0:
                            yi = refs[k][1][:, i, :, :]
                            yi = torch.unsqueeze(yi, 1).repeat(1, x.size(1), 1, 1)
                            yi = refs[k][0].mul(yi)
                        else:
                            yi = refs[1 - k][1][:, i, :, :]
                            yi = torch.unsqueeze(yi, 1).repeat(1, x.size(1), 1, 1)
                            yi = refs[1 - k][0].mul(yi)
                        if i == 0:
                            code = self.componet_enc(yi)
                        else:
                            code = torch.cat([code, self.componet_enc(yi)], dim=1)
                    codes.append(code)
                for i in range(0, map_x.size(1)):
                    yi = map_x[:, i, :, :]
                    yi = torch.unsqueeze(yi, 1).repeat(1, x.size(1), 1, 1)
                    yi = x.mul(yi)
                    if i == 0:
                        code = self.componet_enc(yi)
                    else:
                        code = torch.cat([code, self.componet_enc(yi)], dim=1)
                codes.append(code)
                return codes
        if self.phase == 'train':
            for i in range(0, map_x.size(1)):
                yi = map_x[:, i, :, :]
                yi = torch.unsqueeze(yi, 1).repeat(1, x.size(1), 1, 1)
                yi = x.mul(yi)
                if i == 0:
                    code = self.componet_enc(yi)
                else:
                    code = torch.cat([code, self.componet_enc(yi)], dim=1)
            return code


class FaceEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(FaceEncoder, self).__init__()
        self.model = []
        self.model += [ConvBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [ConvBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


class MakeupFuseDecoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='adain', activ='relu', pad_type='zero'):
        super(MakeupFuseDecoder, self).__init__()

        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           ConvBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [ConvBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


##################################################################################
# Sequential Models
##################################################################################
class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):
        super(MLP, self).__init__()
        self.model = []
        self.model += [linearBlock(input_dim, input_dim, norm=norm, activation=activ)]
        self.model += [linearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [linearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [linearBlock(dim, output_dim, norm='none', activation='none')]  # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, style0, style1, a=0):
        return self.model[3]((1 - a) * self.model[0:3](style0.view(style0.size(0), -1)) + a * self.model[0:3](
            style1.view(style1.size(0), -1)))


##################################################################################
# Basic Blocks
##################################################################################
class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [ConvBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [ConvBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(ConvBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            # self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class linearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(linearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        if norm == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        else:

            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):

        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


##################################################################################
# Normalization layers
##################################################################################
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    """
    Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """

    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)
