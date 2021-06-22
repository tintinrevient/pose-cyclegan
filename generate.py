import argparse
import torch
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

from models import Generator
from models import Discriminator

parser = argparse.ArgumentParser()
parser.add_argument('--input_img', type=str, default='', help='input image')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--generator', type=str, default='A2B', help='select A2B generator or B2A generator')
parser.add_argument('--generator_A2B', type=str, default='weights/surf2nude/netG_A2B.pth', help='A2B generator checkpoint file')
parser.add_argument('--generator_B2A', type=str, default='weights/surf2nude/netG_B2A.pth', help='B2A generator checkpoint file')
parser.add_argument('--discriminator_A', type=str, default='weights/surf2nude/netD_A.pth', help='A discriminator checkpoint file')
parser.add_argument('--discriminator_B', type=str, default='weights/surf2nude/netD_B.pth', help='B discriminator checkpoint file')

opt = parser.parse_args()
print(opt)

# Load the weights
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)
netD_A = Discriminator(opt.input_nc)
netD_B = Discriminator(opt.output_nc)

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()

netG_A2B.load_state_dict(torch.load(opt.generator_A2B, map_location=lambda storage, loc: storage))
netG_B2A.load_state_dict(torch.load(opt.generator_B2A, map_location=lambda storage, loc: storage))
netD_A.load_state_dict(torch.load(opt.discriminator_A, map_location=lambda storage, loc: storage))
netD_B.load_state_dict(torch.load(opt.discriminator_B, map_location=lambda storage, loc: storage))

if opt.generator == 'A2B':
    netG = netG_A2B
    netD = netD_B
elif opt.generator == 'B2A':
    netG = netG_B2A
    netD = netD_A

# Allocate the input memory
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batch_size, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batch_size, opt.output_nc, opt.size, opt.size)

if opt.generator == 'A2B':
    input_tensor = input_A
elif opt.generator == 'B2A':
    input_tensor = input_B

# Transform the image to tensor
preprocess = transforms.Compose([
    transforms.Resize(opt.size),
    transforms.CenterCrop(opt.size),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5,0.5,0.5),
        std=(0.5,0.5,0.5)
    )
])

# Load the input image
input_img = Image.open(opt.input_img)
real_img = Variable(input_tensor.copy_(torch.unsqueeze(preprocess(input_img), 0)))

# Generate output
fake_img = 0.5*(netG(real_img).data + 1.0)
save_image(fake_img, 'output.png')

# Discriminator inference
pred = netD(fake_img)
print(pred.shape)