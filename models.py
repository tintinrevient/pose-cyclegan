import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image


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
                    nn.Conv2d(64, 3, 7),
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

    def __init__(self, input_nc, patch_size):
        super().__init__(input_nc)
        self.patch_size = patch_size

    def forward(self, input):
        B, C, H, W = input.size(0), input.size(1), input.size(2), input.size(3)
        size = self.patch_size
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


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class PatchSample(nn.Module):

    def __init__(self, netMLPs):
        super(PatchSample, self).__init__()
        self.netMLPs = netMLPs
        self.l2norm = Normalize(2)

        # Debug - Check whether each MLP network has been updated
        # network_id = 2
        # for name, param in netMLPs[network_id].named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data.shape)
        #         print(name, param.data)

    def forward(self, feats, num_patches=64, patch_ids=None):

        return_ids = []
        return_feats = []

        for feat_id, feat in enumerate(feats):

            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)

            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    patch_id = torch.randperm(feat_reshape.shape[1], device=feats[0].device)
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
            else:
                x_sample = feat_reshape
                patch_id = []

            mlp = self.netMLPs[feat_id]
            x_sample = mlp(x_sample)

            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])

            # Debug - Save the embedding space image for each feature
            # print('x_sample {} shape:'.format(feat_id), x_sample.shape)
            # sample_img = 0.5 * (x_sample.data + 1.0)
            # save_image(sample_img, 'sample-{}.png'.format(feat_id))

            return_feats.append(x_sample)

        return return_feats, return_ids


# Amplify 16x16 patch of segment
class SegmentAmplifier(nn.Module):
    def __init__(self, input_nc, n_layers):
        super(SegmentAmplifier, self).__init__()

        model = []
        output_nc = input_nc // 2

        for _ in range(n_layers):
            model += [nn.ConvTranspose2d(input_nc, output_nc, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(output_nc),
                      nn.ReLU(inplace=True)]
            input_nc = output_nc
            output_nc = input_nc // 2

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(input_nc, 3, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)