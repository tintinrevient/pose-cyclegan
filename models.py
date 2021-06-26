import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import weights_init_normal


class ResidualBlock(nn.Module):

    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):

    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block       
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x, layer_ids=[], encode_only=False):

        if encode_only:
            feat = x
            feats = []
            for layer_id, layer in enumerate(self.model):
                # feed into the model layer by layer
                feat = layer(feat)

                # for the desired layers
                if layer_id in layer_ids:
                    feats.append(feat)

            return feats

        else:
            return self.model(x)


class Discriminator(nn.Module):

    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

        # Average pooling and flatten
        # x =  self.model(x)
        # return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

# PatchGAN discriminator
class PatchDiscriminator(Discriminator):

    def __init__(self, input_nc):
        super().__init__(input_nc)

    def forward(self, input):
        B, C, H, W = input.size(0), input.size(1), input.size(2), input.size(3)
        size = 32
        # size = 256 # the same as Discriminator

        Y = H // size
        X = W // size
        input = input.view(B, C, Y, size, X, size)
        input = input.permute(0, 2, 4, 1, 3, 5).contiguous().view(B * Y * X, C, size, size)

        return super().forward(input)

# Patch MLP
# Potential issues: currently, we use the same patch_ids for multiple images in the batch
class PatchMLP(nn.Module):
    def __init__(self, input_nc):
        super(PatchMLP, self).__init__()
        self.model = nn.Sequential(*[nn.Linear(input_nc, 256), nn.ReLU(), nn.Linear(256, 256)])

    def forward(self, x):
        return self.model(x)

class PatchSample(nn.Module):

    def __init__(self, netMLPs):
        super(PatchSample, self).__init__()
        self.netMLPs = netMLPs
        # print('netMLPs size:', len(netMLPs))

    def forward(self, feats, num_patches=64, patch_ids=None):

        # print('feats size:', len(feats))
        # print('feats 0 shape:', feats[0].shape)
        # print('feats 1 shape:', feats[1].shape)
        # print('feats 2 shape:', feats[2].shape)
        # print('num of patches:', num_patches)
        # print('patch ids:', patch_ids)

        return_ids = []
        return_feats = []

        for feat_id, feat in enumerate(feats):

            # print('feat_id:', feat_id)
            # print('feat shape', feat.shape)
            # print('num_patches:', num_patches)

            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)

            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    patch_id = torch.randperm(feat_reshape.shape[1], device=feats[0].device)
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
                # print('inner x_sample shape:', x_sample.shape)
            else:
                x_sample = feat_reshape
                patch_id = []

            mlp = self.netMLPs[feat_id]
            # print('feat_id:', feat_id)
            # print(mlp)
            x_sample = mlp(x_sample)

            # print('outer x_sample shape:', x_sample.shape)

            return_ids.append(patch_id)
            x_sample = x_sample.div(torch.norm(x_sample))

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])

            # print('x_sample shape:', x_sample.shape)

            return_feats.append(x_sample)

        return return_feats, return_ids