import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/C' % mode) + '/*.*'))

    def __getitem__(self, index):

        # iterate along the index of the domain that has the maximum size of images
        if len(self.files_A) == max(len(self.files_A), len(self.files_B)):

            path_A = self.files_A[index % len(self.files_A)]
            image_A = Image.open(path_A)
            item_A = self.transform(image_A)
            shape_A = tuple(image_A.size) # (width, height)

            if self.unaligned:
                path_B = self.files_B[random.randint(0, len(self.files_B) - 1)]
                image_B = Image.open(path_B)
                item_B = self.transform(image_B)
                shape_B = tuple(image_B.size)
            else:
                path_B = self.files_B[index % len(self.files_B)]
                image_B = Image.open(path_B)
                item_B = self.transform(image_B)
                shape_B = tuple(image_B.size)

        else:

            path_B = self.files_B[index % len(self.files_B)]
            image_B = Image.open(path_B)
            item_B = self.transform(image_B)
            shape_B = tuple(image_B.size)

            if self.unaligned:
                path_A = self.files_A[random.randint(0, len(self.files_A) - 1)]
                image_A = Image.open(path_A)
                item_A = self.transform(image_A)
                shape_A = tuple(image_A.size)  # (width, height)
            else:
                path_A = self.files_A[index % len(self.files_A)]
                image_A = Image.open(path_A)
                item_A = self.transform(image_A)
                shape_A = tuple(image_A.size)  # (width, height)

        return {'A': item_A, 'path_A': path_A, 'shape_A': shape_A, 'B': item_B, 'path_B': path_B, 'shape_B': shape_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))