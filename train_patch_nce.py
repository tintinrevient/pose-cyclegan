import argparse
import itertools
import os.path

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchsummary import summary
from PIL import Image
from tqdm import tqdm

from models import Generator, PatchDiscriminator, PatchMLP, PatchSample
from losses import PatchNCELoss
from utils import ReplayBuffer, LambdaLR, LossLogger, weights_init_normal
from datasets import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--dataset', type=str, default='horse2zebra', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=150, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
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
netD_A = PatchDiscriminator(opt.input_nc)
netD_B = PatchDiscriminator(opt.output_nc)

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

netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)
netMLP_1.apply(weights_init_normal)
netMLP_2.apply(weights_init_normal)
netMLP_3.apply(weights_init_normal)
netMLP_4.apply(weights_init_normal)
netMLP_5.apply(weights_init_normal)

# Losses
criterion_GAN = torch.nn.MSELoss() # LSGAN
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

criterion_NCE = []
for nce_layer in opt.nce_layers:
    criterion_NCE.append(PatchNCELoss(opt).cuda() if opt.cuda else PatchNCELoss(opt))

# Optimizers + LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_MLP_1 = torch.optim.Adam(netMLP_1.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_MLP_2 = torch.optim.Adam(netMLP_2.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_MLP_3 = torch.optim.Adam(netMLP_3.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_MLP_4 = torch.optim.Adam(netMLP_4.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_MLP_5 = torch.optim.Adam(netMLP_5.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_MLP1 = torch.optim.lr_scheduler.LambdaLR(optimizer_MLP_1, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_MLP2 = torch.optim.lr_scheduler.LambdaLR(optimizer_MLP_2, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_MLP3 = torch.optim.lr_scheduler.LambdaLR(optimizer_MLP_3, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_MLP4 = torch.optim.lr_scheduler.LambdaLR(optimizer_MLP_4, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_MLP5 = torch.optim.lr_scheduler.LambdaLR(optimizer_MLP_5, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs + Targets: Memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batch_size, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batch_size, opt.output_nc, opt.size, opt.size)
target_real = Variable(Tensor(opt.batch_size).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batch_size).fill_(0.0), requires_grad=False)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Dataset loader
transforms_ = [ transforms.Resize(int(opt.size*1.12), Image.BICUBIC), 
                transforms.CenterCrop(opt.size), # change from RandomCrop to CenterCrop
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
dataloader = DataLoader(ImageDataset(os.path.join('datasets', opt.dataset), transforms_=transforms_, unaligned=True),
                        batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

# Directories + Files: Initialization
## Output
output_dir = os.path.join('output', opt.dataset, 'nce{}'.format(opt.num_patches))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

## Weights
weights_dir = os.path.join('weights', opt.dataset, 'nce{}'.format(opt.num_patches))
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
######################################

############## Training ##############
for epoch in range(opt.epoch, opt.n_epochs):

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    logger = LossLogger()

    for i, batch in progress_bar:

        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()
        optimizer_MLP_1.zero_grad()
        optimizer_MLP_2.zero_grad()
        optimizer_MLP_3.zero_grad()
        optimizer_MLP_4.zero_grad()
        optimizer_MLP_5.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B)*5.0
        # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A)*5.0

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

        # NCE loss
        loss_NCE_A = calculate_NCE_loss(real_A, fake_B)
        loss_NCE_B = calculate_NCE_loss(real_B, same_B)
        loss_NCE = loss_NCE_A + loss_NCE_B

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
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
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
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()

        optimizer_D_B.step()
        ####################################

        # Display the losses
        progress_bar.set_description(
            f"[{epoch}/{opt.n_epochs - 1}][{i}/{len(dataloader) - 1}] "
            f"Loss_D: {(loss_D_A + loss_D_B).item():.4f} "
            f"Loss_G: {loss_G.item():.4f} "
            f"Loss_G_identity: {(loss_identity_A + loss_identity_B).item():.4f} "
            f"Loss_G_GAN: {(loss_GAN_A2B + loss_GAN_B2A).item():.4f} "
            f"Loss_G_cycle: {(loss_cycle_ABA + loss_cycle_BAB).item():.4f} "
            f"Loss_G_NCE: {(loss_NCE).item():.4f}")

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
######################################