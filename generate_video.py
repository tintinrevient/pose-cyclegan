import argparse
import torch
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

from models import Generator
from models import Discriminator

import glob, os, cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='', help='input image directory')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
parser.add_argument('--generator_A2B', type=str, default='weights/surf2nude/patchnce/netG_A2B.pth', help='A2B generator checkpoint file')

opt = parser.parse_args()
print(opt)

video_dir = 'video'
if not os.path.exists(video_dir):
    os.makedirs(video_dir)


def style_transfer():

    # Load the weights
    netG = Generator(opt.input_nc, opt.output_nc)
    netG.load_state_dict(torch.load(opt.generator_A2B, map_location=lambda storage, loc: storage))

    # Allocate the input memory
    input_tensor = torch.Tensor(opt.batch_size, opt.input_nc, opt.size, opt.size)

    # Transform the image to tensor
    preprocess = transforms.Compose([
        transforms.Resize(opt.size),
        transforms.CenterCrop(opt.size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    ])

    for fname in glob.glob('{}/*.jpg'.format(opt.input_dir)):
        print(fname[fname.rfind('/') + 1:])

        # Load the input image
        input_img = Image.open(fname)
        real_img = Variable(input_tensor.copy_(torch.unsqueeze(preprocess(input_img), 0)))

        # Generate output
        fake_img = 0.5 * (netG(real_img).data + 1.0)
        save_image(fake_img, '{}/{}'.format(video_dir, fname[fname.rfind('/') + 1:]))


def image_sequence_to_video():

    img_array = []
    for fname in sorted(glob.glob('{}/*.jpg'.format(opt.input_dir))):

        print(fname[fname.rfind('/') + 1:])

        img = cv2.imread(fname)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter('{}/project.avi'.format(video_dir), cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


if __name__ == '__main__':

    # step 1: style transfer
    # style_transfer()

    # step 2: image sequence to video
    image_sequence_to_video()