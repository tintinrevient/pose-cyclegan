import argparse
import itertools
import os.path
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchsummary import summary
from PIL import Image
from tqdm import tqdm

from models import Generator, PatchDiscriminator, PatchMLP, PatchSample, SegmentAmplifier
from losses import PatchNCELoss
from utils import ReplayBuffer, LambdaLR, LossLogger, weights_init_normal
from datasets import ImageDataset
from pycocotools.coco import COCO
from contour_coco_woman import get_segm_patches as get_segm_patches_from_source
from contour_impressionism import get_segm_patches as get_segm_patches_from_target

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--dataset', type=str, default='horse2zebra', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=150,
                    help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--patch_size', type=int, default=32, help='size of the patch for PatchGAN discriminator')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--print_freq', type=int, default=100, help='every how many images to save the sample images')
parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
parser.add_argument('--nce_temperature', type=float, default=0.07, help='temperature for NCE loss')
parser.add_argument('--nce_lambda', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')

opt = parser.parse_args()
opt.nce_layers = [int(i) for i in opt.nce_layers.split(',')]
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

####### Definition of variables ######
# Networks
## Generators
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)

## PatchGAN discriminators
netD_A = PatchDiscriminator(opt.input_nc, opt.patch_size)
netD_B = PatchDiscriminator(opt.output_nc, opt.patch_size)

# print('netG_A2B:\n', netG_A2B)
# print(summary(netG_A2B, (3, 256, 256)))
# print('netD_A:\n', netD_A)
# print(summary(netD_A, (3, 256, 256)))

## Patch MLP
netMLP_1 = PatchMLP(input_nc=3)
netMLP_2 = PatchMLP(input_nc=128)
netMLP_3 = PatchMLP(input_nc=256)
netMLP_4 = PatchMLP(input_nc=256)
netMLP_5 = PatchMLP(input_nc=256)

# Segment
netS_in_small = SegmentAmplifier(128, 4)
netS_in_large = SegmentAmplifier(256, 4)
netS_out_large = SegmentAmplifier(256, 4)
netS_out_small = SegmentAmplifier(128, 4)

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()
    netMLP_1.cuda()
    netMLP_2.cuda()
    netMLP_3.cuda()
    netMLP_4.cuda()
    netMLP_5.cuda()
    netS_in_small.cuda()
    netS_in_large.cuda()
    netS_out_large.cuda()
    netS_out_small.cuda()

netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)
netMLP_1.apply(weights_init_normal)
netMLP_2.apply(weights_init_normal)
netMLP_3.apply(weights_init_normal)
netMLP_4.apply(weights_init_normal)
netMLP_5.apply(weights_init_normal)
netS_in_small.apply(weights_init_normal)
netS_in_large.apply(weights_init_normal)
netS_out_large.apply(weights_init_normal)
netS_out_small.apply(weights_init_normal)

# Losses
criterion_GAN = torch.nn.MSELoss()  # LSGAN
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

criterion_NCE = []
for nce_layer in opt.nce_layers:
    criterion_NCE.append(PatchNCELoss(opt).cuda() if opt.cuda else PatchNCELoss(opt))

criterion_segm = torch.nn.L1Loss()

# Optimizers + LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_MLP_1 = torch.optim.Adam(netMLP_1.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_MLP_2 = torch.optim.Adam(netMLP_2.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_MLP_3 = torch.optim.Adam(netMLP_3.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_MLP_4 = torch.optim.Adam(netMLP_4.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_MLP_5 = torch.optim.Adam(netMLP_5.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_S_in_small = torch.optim.Adam(netS_in_small.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_S_in_large = torch.optim.Adam(netS_in_large.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_S_out_large = torch.optim.Adam(netS_out_large.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_S_out_small = torch.optim.Adam(netS_out_small.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_MLP_1 = torch.optim.lr_scheduler.LambdaLR(optimizer_MLP_1, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_MLP_2 = torch.optim.lr_scheduler.LambdaLR(optimizer_MLP_2, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_MLP_3 = torch.optim.lr_scheduler.LambdaLR(optimizer_MLP_3, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_MLP_4 = torch.optim.lr_scheduler.LambdaLR(optimizer_MLP_4, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_MLP_5 = torch.optim.lr_scheduler.LambdaLR(optimizer_MLP_5, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_S_in_small = torch.optim.lr_scheduler.LambdaLR(optimizer_S_in_small, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_S_in_large = torch.optim.lr_scheduler.LambdaLR(optimizer_S_in_large, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_S_out_large = torch.optim.lr_scheduler.LambdaLR(optimizer_S_out_large, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_S_out_small = torch.optim.lr_scheduler.LambdaLR(optimizer_S_out_small, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs + Targets: Memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batch_size, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batch_size, opt.output_nc, opt.size, opt.size)
target_real = Variable(Tensor(opt.batch_size).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batch_size).fill_(0.0), requires_grad=False)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Dataset loader
transforms_ = [ transforms.Resize(int(opt.size), Image.BICUBIC),
                transforms.CenterCrop(opt.size),  # change from RandomCrop to CenterCrop
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ]
dataloader = DataLoader(ImageDataset(os.path.join('datasets', opt.dataset), transforms_=transforms_, unaligned=True),
                        batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

# Directories + Files: Initialization
## Output
output_dir = os.path.join('output', opt.dataset, 'segm')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

## Weights
weights_dir = os.path.join('weights', opt.dataset, 'segm')
if not os.path.exists(weights_dir):
    os.makedirs(weights_dir)

## Loss plot
# logger = Logger(opt.n_epochs, len(dataloader))
losses_fname = os.path.join(output_dir, 'losses.csv')
with open(losses_fname, 'w') as csv_file:
    pass
######################################

######################################
def calculate_NCE_loss(source, target):
    netMLP = PatchSample(netMLPs=[netMLP_1, netMLP_2, netMLP_3, netMLP_4, netMLP_5])

    feat_k = netG_A2B(source, opt.nce_layers, encode_only=True)
    feat_q = netG_A2B(target, opt.nce_layers, encode_only=True)

    feat_k_pool, sample_ids = netMLP(feat_k, opt.num_patches, None)
    feat_q_pool, _ = netMLP(feat_q, opt.num_patches, sample_ids)

    total_nce_loss = 0.0
    for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, criterion_NCE, opt.nce_layers):
        loss = crit(f_q, f_k) * opt.nce_lambda
        total_nce_loss += loss.mean()

    loss_NCE = total_nce_loss / len(opt.nce_layers)

    return loss_NCE


def _get_matched_patches(patches_source, patches_target):

    patches = {}

    if 'Head' in patches_source and 'Head' in patches_target:
        patches['Head'] = {}
        patches['Head']['A'] = patches_source['Head']
        patches['Head']['B'] = patches_target['Head']

    if 'Torso' in patches_source and 'Torso' in patches_target:
        patches['Torso'] = {}
        patches['Torso']['A'] = patches_source['Torso']
        patches['Torso']['B'] = patches_target['Torso']

    if 'RUpperArm' in patches_source and 'RUpperArm' in patches_target:
        patches['RUpperArm'] = {}
        patches['RUpperArm']['A'] = patches_source['RUpperArm']
        patches['RUpperArm']['B'] = patches_target['RUpperArm']

    if 'RLowerArm' in patches_source and 'RLowerArm' in patches_target:
        patches['RLowerArm'] = {}
        patches['RLowerArm']['A'] = patches_source['RLowerArm']
        patches['RLowerArm']['B'] = patches_target['RLowerArm']

    if 'LUpperArm' in patches_source and 'LUpperArm' in patches_target:
        patches['LUpperArm'] = {}
        patches['LUpperArm']['A'] = patches_source['LUpperArm']
        patches['LUpperArm']['B'] = patches_target['LUpperArm']

    if 'LLowerArm' in patches_source and 'LLowerArm' in patches_target:
        patches['LLowerArm'] = {}
        patches['LLowerArm']['A'] = patches_source['LLowerArm']
        patches['LLowerArm']['B'] = patches_target['LLowerArm']

    if 'RThigh' in patches_source and 'RThigh' in patches_target:
        patches['RThigh'] = {}
        patches['RThigh']['A'] = patches_source['RThigh']
        patches['RThigh']['B'] = patches_target['RThigh']

    if 'RCalf' in patches_source and 'RCalf' in patches_target:
        patches['RCalf'] = {}
        patches['RCalf']['A'] = patches_source['RCalf']
        patches['RCalf']['B'] = patches_target['RCalf']

    if 'LThigh' in patches_source and 'LThigh' in patches_target:
        patches['LThigh'] = {}
        patches['LThigh']['A'] = patches_source['LThigh']
        patches['LThigh']['B'] = patches_target['LThigh']

    if 'LCalf' in patches_source and 'LCalf' in patches_target:
        patches['LCalf'] = {}
        patches['LCalf']['A'] = patches_source['LCalf']
        patches['LCalf']['B'] = patches_target['LCalf']

    print('patches:', patches)

    return patches


def calculate_segment_loss(source, target, patches, patch_size):

    loss_segm = None

    half_patch_size = patch_size / 2

    half_patch_size_in_layer = half_patch_size / 2 # patch_size = 16 for shallow and deep layers
    # half_patch_size_in_layer = half_patch_size / 4 # patch_size = 8 for shallow and deep layers

    # A - features
    source_in_features = netG_A2B(source, [4, 8], encode_only=True)
    source_out_features = netG_A2B(source, [16, 20], encode_only=True)

    # B - features
    target_in_features = netG_B2A(target, [4, 8], encode_only=True)
    target_out_features = netG_B2A(target, [16, 20], encode_only=True)

    for name, midpoints in patches.items():

        patch_small = (np.array(midpoints) / 2).astype(int)
        patch_large = (np.array(midpoints) / 4).astype(int)

        # small -> small patch size + large feature size
        # large -> large patch size + small feature size
        # A - patches from features
        # 128 x 128 - in - shallow feature - A - small patch
        source_segm_in_small = netS_in_small(source_in_features[0][:, :,
                                             int(patch_small[1] - half_patch_size_in_layer):int(patch_small[1] + half_patch_size_in_layer),
                                             int(patch_small[0] - half_patch_size_in_layer):int(patch_small[0] + half_patch_size_in_layer)])

        # 64 x 64 - out - deep feature - A - large patch
        target_segm_out_large = netS_in_large(target_out_features[0][:, :,
                                              int(patch_large[1] - half_patch_size_in_layer):int(patch_large[1] + half_patch_size_in_layer),
                                              int(patch_large[0] - half_patch_size_in_layer):int(patch_large[0] + half_patch_size_in_layer)])

        # B - patches from features
        # 128 x 128 - in - shallow feature - B - small
        target_segm_in_small = netS_out_small(target_in_features[0][:, :,
                                              int(patch_small[1] - half_patch_size_in_layer):int(patch_small[1] + half_patch_size_in_layer),
                                              int(patch_small[0] - half_patch_size_in_layer):int(patch_small[0] + half_patch_size_in_layer)])

        # 64 x 64 - out - deep feature - B - large
        source_segm_out_large = netS_out_large(source_out_features[0][:, :,
                                               int(patch_large[1] - half_patch_size_in_layer):int(patch_large[1] + half_patch_size_in_layer),
                                               int(patch_large[0] - half_patch_size_in_layer):int(patch_large[0] + half_patch_size_in_layer)])

        loss_source_in_small_target_out_large = criterion_segm(source_segm_in_small, target_segm_out_large) * 10.0
        loss_target_in_small_source_out_large = criterion_segm(target_segm_in_small, source_segm_out_large) * 10.0

        if loss_segm is None and loss_source_in_small_target_out_large is not None and loss_target_in_small_source_out_large is not None:
            loss_segm = loss_source_in_small_target_out_large + loss_target_in_small_source_out_large
        elif loss_segm is not None and loss_source_in_small_target_out_large is not None and loss_target_in_small_source_out_large is not None:
            loss_segm += loss_source_in_small_target_out_large + loss_target_in_small_source_out_large

    return loss_segm
######################################

############## Training ##############
# dataset setting
coco_folder = os.path.join('datasets', 'coco')

# dense_pose annotation
dp_coco = COCO(os.path.join(coco_folder, 'annotations', 'densepose_train2014.json'))

for epoch in range(opt.epoch, opt.n_epochs):

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    logger = LossLogger()

    for i, batch in progress_bar:

        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        path_A = batch['path_A']
        shape_A = batch['shape_A']

        real_B = Variable(input_B.copy_(batch['B']))
        path_B = batch['path_B']
        shape_B = batch['shape_B']

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()
        optimizer_MLP_1.zero_grad()
        optimizer_MLP_2.zero_grad()
        optimizer_MLP_3.zero_grad()
        optimizer_MLP_4.zero_grad()
        optimizer_MLP_5.zero_grad()

        optimizer_S_in_small.zero_grad()
        optimizer_S_in_large.zero_grad()
        optimizer_S_out_small.zero_grad()
        optimizer_S_out_large.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B) * 5.0
        # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A) * 5.0

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0

        # NCE loss
        loss_NCE_A = calculate_NCE_loss(real_A, fake_B)
        loss_NCE_B = calculate_NCE_loss(real_B, same_B)
        loss_NCE = loss_NCE_A + loss_NCE_B

        # Segment loss
        patches_A = get_segm_patches_from_source(dp_coco=dp_coco,
                                                 image_tensor=real_A[0], image_fpath=path_A[0],
                                                 image_shape=shape_A,
                                                 image_size=opt.size, patch_size=opt.patch_size)

        patches_B = get_segm_patches_from_target(image_tensor=real_B[0], image_fpath=path_B[0],
                                                 image_shape=shape_B,
                                                 image_size=opt.size, patch_size=opt.patch_size)

        loss_segm_real = calculate_segment_loss(source=real_A, target=fake_B, patches=patches_A, patch_size=opt.patch_size)
        loss_segm_fake = calculate_segment_loss(source=fake_A, target=real_B, patches=patches_B, patch_size=opt.patch_size)

        if loss_segm_real is not None and loss_segm_fake is not None:

            loss_segm = loss_segm_real + loss_segm_fake

            # Total loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + loss_NCE + loss_segm
            loss_G.backward()

            optimizer_G.step()
            optimizer_MLP_1.step()
            optimizer_MLP_2.step()
            optimizer_MLP_3.step()
            optimizer_MLP_4.step()
            optimizer_MLP_5.step()

            optimizer_S_in_small.step()
            optimizer_S_in_large.step()
            optimizer_S_out_small.step()
            optimizer_S_out_large.step()
        else:
            # Total loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + loss_NCE
            loss_G.backward()

            optimizer_G.step()
            optimizer_MLP_1.step()
            optimizer_MLP_2.step()
            optimizer_MLP_3.step()
            optimizer_MLP_4.step()
            optimizer_MLP_5.step()
        ####################################

        ######### Discriminator A ##########
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real.expand_as(pred_real)).mean()

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake.expand_as(pred_fake)).mean()

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ####################################

        ######### Discriminator B ##########
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real.expand_as(pred_real)).mean()

        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake.expand_as(pred_fake)).mean()

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        loss_D_B.backward()

        optimizer_D_B.step()
        ####################################

        # Display the losses
        if loss_segm is not None and loss_segm_real is not None and loss_segm_fake is not None and loss_segm_same is not None:
            progress_bar.set_description(
                f"[{epoch}/{opt.n_epochs - 1}][{i}/{len(dataloader) - 1}] "
                f"Loss_D: {(loss_D_A + loss_D_B).item():.2f} "
                f"Loss_G: {loss_G.item():.2f} "
                f"Loss_G_identity: {(loss_identity_A + loss_identity_B).item():.2f} "
                f"Loss_G_GAN: {(loss_GAN_A2B + loss_GAN_B2A).item():.2f} "
                f"Loss_G_cycle: {(loss_cycle_ABA + loss_cycle_BAB).item():.2f} "
                f"Loss_G_NCE: {loss_NCE.item():.2f} "
                f"Loss_G_segm: {loss_segm.item():.2f}")

            # Log the losses of each batch
            logger.log({
                'Loss_D': (loss_D_A + loss_D_B).item(),
                'Loss_D_A': loss_D_A.item(),
                'Loss_D_B': loss_D_B.item(),
                'Loss_G': loss_G.item(),
                'Loss_G_identity': (loss_identity_A + loss_identity_B).item(),
                'Loss_G_identity_A': loss_identity_A.item(),
                'Loss_G_identity_B': loss_identity_B.item(),
                'Loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A).item(),
                'Loss_G_GAN_A2B': loss_GAN_A2B.item(),
                'Loss_G_GAN_B2A': loss_GAN_B2A.item(),
                'Loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB).item(),
                'Loss_G_cycle_ABA': loss_cycle_ABA.item(),
                'Loss_G_cycle_BAB': loss_cycle_BAB.item(),
                'Loss_G_NCE': loss_NCE.item(),
                'Loss_G_NCE_A': loss_NCE_A.item(),
                'Loss_G_NCE_B': loss_NCE_B.item(),
                'Loss_G_segm': loss_segm.item(),
                'Loss_G_segm_real': loss_segm_real.item(),
                'Loss_G_segm_fake': loss_segm_fake.item(),
                'Loss_G_segm_same': loss_segm_same.item()
            })
        else:
            progress_bar.set_description(
                f"[{epoch}/{opt.n_epochs - 1}][{i}/{len(dataloader) - 1}] "
                f"Loss_D: {(loss_D_A + loss_D_B).item():.2f} "
                f"Loss_G: {loss_G.item():.2f} "
                f"Loss_G_identity: {(loss_identity_A + loss_identity_B).item():.2f} "
                f"Loss_G_GAN: {(loss_GAN_A2B + loss_GAN_B2A).item():.2f} "
                f"Loss_G_cycle: {(loss_cycle_ABA + loss_cycle_BAB).item():.2f} "
                f"Loss_G_NCE: {loss_NCE.item():.2f}")

            # Log the losses of each batch
            logger.log({
                'Loss_D': (loss_D_A + loss_D_B).item(),
                'Loss_D_A': loss_D_A.item(),
                'Loss_D_B': loss_D_B.item(),
                'Loss_G': loss_G.item(),
                'Loss_G_identity': (loss_identity_A + loss_identity_B).item(),
                'Loss_G_identity_A': loss_identity_A.item(),
                'Loss_G_identity_B': loss_identity_B.item(),
                'Loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A).item(),
                'Loss_G_GAN_A2B': loss_GAN_A2B.item(),
                'Loss_G_GAN_B2A': loss_GAN_B2A.item(),
                'Loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB).item(),
                'Loss_G_cycle_ABA': loss_cycle_ABA.item(),
                'Loss_G_cycle_BAB': loss_cycle_BAB.item(),
                'Loss_G_NCE': loss_NCE.item(),
                'Loss_G_NCE_A': loss_NCE_A.item(),
                'Loss_G_NCE_B': loss_NCE_B.item()
            })

        # Save the sample images every print_freq
        if i % opt.print_freq == 0:

            # Make the directory of each epoch
            epoch_dir = os.path.join(output_dir, 'epoch_{}'.format(epoch))
            if not os.path.exists(epoch_dir):
                os.makedirs(epoch_dir)

            # real images
            vutils.save_image(real_A,
                              f"{epoch_dir}/A_real_batch_{i}.png",
                              normalize=True)
            vutils.save_image(real_B,
                              f"{epoch_dir}/B_real_batch_{i}.png",
                              normalize=True)

            # fake images
            fake_A_image = 0.5 * (fake_A.data + 1.0)
            fake_B_image = 0.5 * (fake_B.data + 1.0)

            vutils.save_image(fake_A_image.detach(),
                              f"{epoch_dir}/A_fake_G_B2A_batch_{i}.png",
                              normalize=True)
            vutils.save_image(fake_B_image.detach(),
                              f"{epoch_dir}/B_fake_G_A2B_batch_{i}.png",
                              normalize=True)

            # identity images
            same_A_image = 0.5 * (same_A.data + 1.0)
            same_B_image = 0.5 * (same_B.data + 1.0)

            vutils.save_image(same_A_image.detach(),
                              f"{epoch_dir}/A_same_G_B2A_batch_{i}.png",
                              normalize=True)
            vutils.save_image(same_B_image.detach(),
                              f"{epoch_dir}/B_same_G_A2B_batch_{i}.png",
                              normalize=True)

            # cycle images
            recovered_A_image = 0.5 * (recovered_A.data + 1.0)
            recovered_B_image = 0.5 * (recovered_B.data + 1.0)

            vutils.save_image(recovered_A_image.detach(),
                              f"{epoch_dir}/A_recovered_G_ABA_batch_{i}.png",
                              normalize=True)
            vutils.save_image(recovered_B_image.detach(),
                              f"{epoch_dir}/B_recovered_G_BAB_batch_{i}.png",
                              normalize=True)

    # Save the losses at the end of each epoch
    logger.save(fname=losses_fname, epoch=epoch, n_batches=len(dataloader))

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # Save models checkpoints
    torch.save(netG_A2B.state_dict(), os.path.join(weights_dir, 'netG_A2B.pth'))
    torch.save(netG_B2A.state_dict(), os.path.join(weights_dir, 'netG_B2A.pth'))
    torch.save(netD_A.state_dict(), os.path.join(weights_dir, 'netD_A.pth'))
    torch.save(netD_B.state_dict(), os.path.join(weights_dir, 'netD_B.pth'))
    torch.save(netMLP_1.state_dict(), os.path.join(weights_dir, 'netMLP_1.pth'))
    torch.save(netMLP_2.state_dict(), os.path.join(weights_dir, 'netMLP_2.pth'))
    torch.save(netMLP_3.state_dict(), os.path.join(weights_dir, 'netMLP_3.pth'))
    torch.save(netMLP_4.state_dict(), os.path.join(weights_dir, 'netMLP_4.pth'))
    torch.save(netMLP_5.state_dict(), os.path.join(weights_dir, 'netMLP_5.pth'))

    torch.save(netS_in_small.state_dict(), os.path.join(weights_dir, 'netS_in_small.pth'))
    torch.save(netS_in_large.state_dict(), os.path.join(weights_dir, 'netS_in_large.pth'))
    torch.save(netS_out_small.state_dict(), os.path.join(weights_dir, 'netS_out_small.pth'))
    torch.save(netS_out_large.state_dict(), os.path.join(weights_dir, 'netS_out_large.pth'))
######################################