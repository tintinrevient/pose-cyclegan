import os, re, glob, cv2
import numpy as np
import pandas as pd

from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
from matplotlib import pyplot as plt
from datasets import ImageDataset

# the path to the data of contour.csv
fname_contour = os.path.join('output', 'contour.csv')

# the path to the data of keypoints
openpose_keypoints_dir = os.path.join('output', 'data', 'impressionism')

# the path to the data of norm_segm.csv
# fname_norm_segm = os.path.join('output', 'norm_segm_impressionism_keypoints.csv')
fname_norm_segm_impressionism = os.path.join('output', 'norm_segm_impressionism.csv')

# dataset setting
coco_folder = os.path.join('datasets', 'coco')

# Body 25 Keypoints
JOINT_ID = [
    'Nose', 'Neck',
    'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist',
    'MidHip',
    'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
    'REye', 'LEye', 'REar', 'LEar',
    'LBigToe', 'LSmallToe', 'LHeel', 'RBigToe', 'RSmallToe', 'RHeel',
    'Background'
]

# BGRA -> alpha channel: 0 = transparent, 255 = non-transparent
COARSE_TO_COLOR = {
    'Background': [255, 255, 255, 255],
    'Torso': [191, 78, 22, 255],
    'RThigh': [167, 181, 44, 255],
    'LThigh': [141, 187, 91, 255],
    'RCalf': [114, 191, 147, 255],
    'LCalf': [96, 188, 192, 255],
    'LUpperArm': [87, 207, 112, 255],
    'RUpperArm': [55, 218, 162, 255],
    'LLowerArm': [25, 226, 216, 255],
    'RLowerArm': [37, 231, 253, 255],
    'Head': [14, 251, 249, 255]
}

for key, value in list(COARSE_TO_COLOR.items()):
    COARSE_TO_COLOR[key] = np.array(value) / 256


def _euclidian(point1, point2):

    return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)


def _get_keypoints(infile, xy_scaler):

    painting_number = infile[infile.rfind('/') + 1:infile.rfind('.')]

    file_keypoints = os.path.join(openpose_keypoints_dir, '{}_keypoints.npy'.format(painting_number))
    people_keypoints = np.load(file_keypoints, allow_pickle='TRUE').item()['keypoints']

    # ONLY use the first person in the image
    person_index = 1
    keypoints = people_keypoints[0]

    keypoints = np.array(keypoints).astype(int)
    keypoints = dict(zip(JOINT_ID, keypoints))
    keypoints = {key: (np.array(value[0:2]) * xy_scaler[2] - np.array(xy_scaler[0:2])) for key, value in keypoints.items()}

    # remove the non-existent keypoints
    for key, value in list(keypoints.items()):
        if np.sum(value) == 0:  # [x, y, sore] -> score = 0
            keypoints.pop(key, None)

    return keypoints


def calc_scaler():

    adjust = 1.5

    for infile in glob.glob('datasets/surf2nude/train/C/*.jpg'):

        painting_number = infile[infile.rfind('/') + 1:infile.rfind('.')]
        person_index = 1

        # get the dict of keypoints
        keypoints = _get_keypoints(infile, [0, 0, 1])

        # check if the keypoints of nose + neck exist!
        if 'Nose' in keypoints and 'Neck' in keypoints:
            dist = _euclidian(keypoints['Nose'], keypoints['Neck'])
            scaler = 58 / (dist * adjust)
        else:
            print('image {} does not have nose or neck in the keypoints.'.format(painting_number))
            continue

        # write the scaler into 'norm_segm_impressionism'
        norm_segm_dict = {'scaler': scaler}

        index_name = _generate_index_name(infile, person_index)
        df = pd.DataFrame(data=norm_segm_dict, index=[index_name])
        with open(fname_norm_segm_impressionism, 'a') as csv_file:
            df.to_csv(csv_file, index=True, header=False)

        norm_segm_dict = {}


def _generate_index_name(infile, person_index):

    artist = 'Impressionism'
    painting_number = infile[infile.rfind('/') + 1:infile.rfind('.')]
    index_name = '{}_{}_{}'.format(artist, painting_number, person_index)

    return index_name


def _get_midpoints(infile, keypoints):

    midpoints = {}

    # head centroid
    if 'Nose' in keypoints:
        midpoints['Head'] = keypoints['Nose']

    # torso midpoint
    if 'Neck' in keypoints and 'MidHip' in keypoints:
        midpoints['Torso'] = (keypoints['Neck'] + keypoints['MidHip']) / 2

    # upper limbs
    if 'RShoulder' in keypoints and 'RElbow' in keypoints:
        midpoints['RUpperArm'] = (keypoints['RShoulder'] + keypoints['RElbow']) / 2

    if 'RElbow' in keypoints and 'RWrist' in keypoints:
        midpoints['RLowerArm'] = (keypoints['RElbow'] + keypoints['RWrist']) / 2

    if 'LShoulder' in keypoints and 'LElbow' in keypoints:
        midpoints['LUpperArm'] = (keypoints['LShoulder'] + keypoints['LElbow']) / 2

    if 'LElbow' in keypoints and 'LWrist' in keypoints:
        midpoints['LLowerArm'] = (keypoints['LElbow'] + keypoints['LWrist']) / 2

    # lower limbs
    if 'RHip' in keypoints and 'RKnee' in keypoints:
        midpoints['RThigh'] = (keypoints['RHip'] + keypoints['RKnee']) / 2

    if 'RKnee' in keypoints and 'RAnkle' in keypoints:
        midpoints['RCalf'] = (keypoints['RKnee'] + keypoints['RAnkle']) / 2

    if 'LHip' in keypoints and 'LKnee' in keypoints:
        midpoints['LThigh'] = (keypoints['LHip'] + keypoints['LKnee']) / 2

    if 'LKnee' in keypoints and 'LAnkle' in keypoints:
        midpoints['LCalf'] = (keypoints['LKnee'] + keypoints['LAnkle']) / 2

    return midpoints


def _calc_angle(point1, center, point2):

    try:
        a = np.array(point1)[0:2] - np.array(center)[0:2]
        b = np.array(point2)[0:2] - np.array(center)[0:2]

        cos_theta = np.dot(a, b)
        sin_theta = np.cross(a, b)

        rad = np.arctan2(sin_theta, cos_theta)
        deg = np.rad2deg(rad)

        if np.isnan(rad):
            return 0, 0

        return rad, deg

    except:
        return 0, 0


def _get_rotated_angles(keypoints, midpoints):

    rotated_angles = {}

    # head
    if 'Neck' in keypoints and 'Head' in midpoints:
        reference_point = np.array(keypoints['Neck']) + np.array((0, -100, 0))
        rad, deg = _calc_angle(point1=midpoints['Head'], center=keypoints['Neck'], point2=reference_point)
        # rotate back to original in reverse direction
        rotated_angles['Head'] = -deg

    # torso
    if 'MidHip' in keypoints and 'Neck' in keypoints:
        reference_point = np.array(keypoints['MidHip']) + np.array((0, -100, 0))
        rad, deg = _calc_angle(point1=keypoints['Neck'], center=keypoints['MidHip'], point2=reference_point)
        rotated_angles['Torso'] = -deg

    # upper limbs
    if 'RShoulder' in keypoints and 'RElbow' in keypoints:
        reference_point = np.array(keypoints['RShoulder']) + np.array((-100, 0, 0))
        rad, deg = _calc_angle(point1=keypoints['RElbow'], center=keypoints['RShoulder'], point2=reference_point)
        rotated_angles['RUpperArm'] = -deg

    if 'RElbow' in keypoints and 'RWrist' in keypoints:
        reference_point = np.array(keypoints['RElbow']) + np.array((-100, 0, 0))
        rad, deg = _calc_angle(point1=keypoints['RWrist'], center=keypoints['RElbow'], point2=reference_point)
        rotated_angles['RLowerArm'] = -deg

    if 'LShoulder' in keypoints and 'LElbow' in keypoints:
        reference_point = np.array(keypoints['LShoulder']) + np.array((100, 0, 0))
        rad, deg = _calc_angle(point1=keypoints['LElbow'], center=keypoints['LShoulder'], point2=reference_point)
        rotated_angles['LUpperArm'] = -deg

    if 'LElbow' in keypoints and 'LWrist' in keypoints:
        reference_point = np.array(keypoints['LElbow']) + np.array((100, 0, 0))
        rad, deg = _calc_angle(point1=keypoints['LWrist'], center=keypoints['LElbow'], point2=reference_point)
        rotated_angles['LLowerArm'] = -deg

    # lower limbs
    if 'RHip' in keypoints and 'RKnee' in keypoints:
        reference_point = np.array(keypoints['RHip']) + np.array((0, 100, 0))
        rad, deg = _calc_angle(point1=keypoints['RKnee'], center=keypoints['RHip'], point2=reference_point)
        rotated_angles['RThigh'] = -deg

    if 'RKnee' in keypoints and 'RAnkle' in keypoints:
        reference_point = np.array(keypoints['RKnee']) + np.array((0, 100, 0))
        rad, deg = _calc_angle(point1=keypoints['RAnkle'], center=keypoints['RKnee'], point2=reference_point)
        rotated_angles['RCalf'] = -deg

    if 'LHip' in keypoints and 'LKnee' in keypoints:
        reference_point = np.array(keypoints['LHip']) + np.array((0, 100, 0))
        rad, deg = _calc_angle(point1=keypoints['LKnee'], center=keypoints['LHip'], point2=reference_point)
        rotated_angles['LThigh'] = -deg

    if 'LKnee' in keypoints and 'LAnkle' in keypoints:
        reference_point = np.array(keypoints['LKnee']) + np.array((0, 100, 0))
        rad, deg = _calc_angle(point1=keypoints['LAnkle'], center=keypoints['LKnee'], point2=reference_point)
        rotated_angles['LCalf'] = -deg

    return rotated_angles


def _draw_norm_segm(image, midpoints, rotated_angles, dict_norm_segm):

    # scaler
    scaler = 1 / dict_norm_segm['scaler']

    # head
    if 'Head' in midpoints and 'Head' in rotated_angles:
        rect = ((midpoints['Head'][0], midpoints['Head'][1]),
                (dict_norm_segm['Head_w'] * scaler, dict_norm_segm['Head_h'] * scaler),
                rotated_angles['Head'])
        box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
        box = np.int0(box)
        cv2.drawContours(image, [box], 0, color=COARSE_TO_COLOR['Head'], thickness=thickness)

    # torso
    if 'Torso' in midpoints and 'Torso' in rotated_angles:
        rect = ((midpoints['Torso'][0], midpoints['Torso'][1]),
                (dict_norm_segm['Torso_w'] * scaler, dict_norm_segm['Torso_h'] * scaler),
                rotated_angles['Torso'])
        box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
        box = np.int0(box)
        cv2.drawContours(image, [box], 0, color=COARSE_TO_COLOR['Torso'], thickness=thickness)

    # upper limbs
    if 'RUpperArm' in midpoints and 'RUpperArm' in rotated_angles:
        rect = ((midpoints['RUpperArm'][0], midpoints['RUpperArm'][1]),
                (dict_norm_segm['RUpperArm_w'] * scaler, dict_norm_segm['RUpperArm_h'] * scaler),
                rotated_angles['RUpperArm'])
        box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
        box = np.int0(box)
        cv2.drawContours(image, [box], 0, color=COARSE_TO_COLOR['RUpperArm'], thickness=thickness)

    if 'RLowerArm' in midpoints and 'RLowerArm' in rotated_angles:
        rect = ((midpoints['RLowerArm'][0], midpoints['RLowerArm'][1]),
                (dict_norm_segm['RLowerArm_w'] * scaler, dict_norm_segm['RLowerArm_h'] * scaler),
                rotated_angles['RLowerArm'])
        box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
        box = np.int0(box)
        cv2.drawContours(image, [box], 0, color=COARSE_TO_COLOR['RLowerArm'], thickness=thickness)

    if 'LUpperArm' in midpoints and 'LUpperArm' in rotated_angles:
        rect = ((midpoints['LUpperArm'][0], midpoints['LUpperArm'][1]),
                (dict_norm_segm['LUpperArm_w'] * scaler, dict_norm_segm['LUpperArm_h'] * scaler),
                rotated_angles['LUpperArm'])
        box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
        box = np.int0(box)
        cv2.drawContours(image, [box], 0, color=COARSE_TO_COLOR['LUpperArm'], thickness=thickness)

    if 'LLowerArm' in midpoints and 'LLowerArm' in rotated_angles:
        rect = ((midpoints['LLowerArm'][0], midpoints['LLowerArm'][1]),
                (dict_norm_segm['LLowerArm_w'] * scaler, dict_norm_segm['LLowerArm_h'] * scaler),
                rotated_angles['LLowerArm'])
        box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
        box = np.int0(box)
        cv2.drawContours(image, [box], 0, color=COARSE_TO_COLOR['LLowerArm'], thickness=thickness)

    # lower limbs
    if 'RThigh' in midpoints and 'RThigh' in rotated_angles:
        rect = ((midpoints['RThigh'][0], midpoints['RThigh'][1]),
                (dict_norm_segm['RThigh_w'] * scaler, dict_norm_segm['RThigh_h'] * scaler),
                rotated_angles['RThigh'])
        box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
        box = np.int0(box)
        cv2.drawContours(image, [box], 0, color=COARSE_TO_COLOR['RThigh'], thickness=thickness)

    if 'RCalf' in midpoints and 'RCalf' in rotated_angles:
        rect = ((midpoints['RCalf'][0], midpoints['RCalf'][1]),
                (dict_norm_segm['RCalf_w'] * scaler, dict_norm_segm['RCalf_h'] * scaler),
                rotated_angles['RCalf'])
        box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
        box = np.int0(box)
        cv2.drawContours(image, [box], 0, color=COARSE_TO_COLOR['RCalf'], thickness=thickness)

    if 'LThigh' in midpoints and 'LThigh' in rotated_angles:
        rect = ((midpoints['LThigh'][0], midpoints['LThigh'][1]),
                (dict_norm_segm['LThigh_w'] * scaler, dict_norm_segm['LThigh_h'] * scaler),
                rotated_angles['LThigh'])
        box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
        box = np.int0(box)
        cv2.drawContours(image, [box], 0, color=COARSE_TO_COLOR['LThigh'], thickness=thickness)

    if 'LCalf' in midpoints and 'LCalf' in rotated_angles:
        rect = ((midpoints['LCalf'][0], midpoints['LCalf'][1]),
                (dict_norm_segm['LCalf_w'] * scaler, dict_norm_segm['LCalf_h'] * scaler),
                rotated_angles['LCalf'])
        box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
        box = np.int0(box)
        cv2.drawContours(image, [box], 0, color=COARSE_TO_COLOR['LCalf'], thickness=thickness)


def visualize(infile, category):

    painting_number = infile[infile.rfind('/') + 1:infile.rfind('.')]
    person_index = 1

    # step 1: get the dict of keypoints
    keypoints = _get_keypoints(infile, [0, 0, 1])

    # step 2: get all the midpoints
    midpoints = _get_midpoints(infile, keypoints)

    # step 3: get the rotation angles
    rotated_angles = _get_rotated_angles(keypoints, midpoints)

    # step 4: load the data of norm_segm
    df_contour = pd.read_csv(fname_contour, index_col=0).astype('float32')
    contour_dict = df_contour.loc[category]

    df_norm_segm = pd.read_csv(fname_norm_segm_impressionism, index_col=0)
    index_name = generate_index_name(infile, person_index)
    contour_dict['scaler'] = df_norm_segm.loc[index_name]['scaler']

    # step 5: draw the norm_segm on the original image
    # load the original image
    image = cv2.imread(infile)
    im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    im_gray = np.tile(im_gray[:, :, np.newaxis], [1, 1, 3])
    # draw norm_segm
    _draw_norm_segm(im_gray, midpoints, rotated_angles, contour_dict)

    # show the final image
    cv2.imshow('Contour of {}'.format(painting_number), im_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def generate_index_name(infile, person_index):

    artist = 'Impressionism'
    painting_number = int(infile[infile.rfind('/')+1:infile.rfind('.')])

    index_name = '{}_{}_{}'.format(artist, painting_number, person_index)

    return index_name


def _is_inside(midpoint, patch_size, image_size):

    point_top_left = np.array(midpoint[0:2]) - np.array([patch_size, patch_size])
    point_bottom_right = np.array(midpoint[0:2]) + np.array([patch_size, patch_size])

    if (point_top_left > 0).all() and (point_top_left < image_size).all() and (point_bottom_right > 0).all() and (point_bottom_right < image_size).all():
        return True
    else:
        return False


def _get_patches(midpoints, patch_size, image_size, contour_dict):

    patches = {}

    # scaler
    scaler = 1 / contour_dict['scaler']

    # head
    if 'Head' in midpoints and _is_inside(midpoints['Head'], patch_size, image_size):
        patches['Head'] = midpoints['Head'][0:2]

    # torso
    if 'Torso' in midpoints and _is_inside(midpoints['Torso'], patch_size, image_size):
        patches['Torso'] = midpoints['Torso'][0:2]

    # upper limbs
    if 'RUpperArm' in midpoints and _is_inside(midpoints['RUpperArm'], patch_size, image_size):
        patches['RUpperArm'] = midpoints['RUpperArm'][0:2]

    if 'RLowerArm' in midpoints and _is_inside(midpoints['RLowerArm'], patch_size, image_size):
        patches['RLowerArm'] = midpoints['RLowerArm'][0:2]

    if 'LUpperArm' in midpoints and _is_inside(midpoints['LUpperArm'], patch_size, image_size):
        patches['LUpperArm'] = midpoints['LUpperArm'][0:2]

    if 'LLowerArm' in midpoints and _is_inside(midpoints['LLowerArm'], patch_size, image_size):
        patches['LLowerArm'] = midpoints['LLowerArm'][0:2]

    # lower limbs
    if 'RThigh' in midpoints and _is_inside(midpoints['RThigh'], patch_size, image_size):
        patches['RThigh'] = midpoints['RThigh'][0:2]

    if 'RCalf' in midpoints and _is_inside(midpoints['RCalf'], patch_size, image_size):
        patches['RCalf'] = midpoints['RCalf'][0:2]

    if 'LThigh' in midpoints and _is_inside(midpoints['LThigh'], patch_size, image_size):
        patches['LThigh'] = midpoints['LThigh'][0:2]

    if 'LCalf' in midpoints and _is_inside(midpoints['LCalf'], patch_size, image_size):
        patches['LCalf'] = midpoints['LCalf'][0:2]

    return patches


def _draw_rect(image, midpoint, patch_size):

    midpoint_key, midpoint_value = midpoint

    rect = ((midpoint_value[0], midpoint_value[1]),
            (patch_size, patch_size),
            0)
    box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, color=COARSE_TO_COLOR[midpoint_key], thickness=thickness)


def get_segm_patches(image_tensor, image_fpath, image_shape, image_size, patch_size):

    # for debug use only!
    image_array = np.array(image_tensor.permute(1, 2, 0))  # (C, H, W) -> (H, W, C)
    # grayscale
    image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
    # color
    # image = image_array[:, :, ::-1].copy()  # RGB -> BGR

    painting_number = image_fpath[image_fpath.rfind('/') + 1:image_fpath.rfind('.')]
    # ONLY use the first person in the image
    person_index = 1

    # step 0: get the scaler
    w, h = image_shape[0].item(), image_shape[1].item()
    if w >= h:
        scaler = image_size / h
        translated_x = (w * scaler - image_size) / 2
        translated_y = 0
    else:
        scaler = image_size / w
        translated_x = 0
        translated_y = (h * scaler - image_size) / 2
    xy_scaler = [translated_x, translated_y, scaler]

    # step 1: get all the keypoints
    keypoints = _get_keypoints(image_fpath, xy_scaler)

    # debug - keypoints
    # for key, value in keypoints.items():
    #     cv2.circle(image, tuple(value[0:2].astype(int)), 3, (255, 0, 255), -1)

    # step 2: get all the midpoints
    midpoints = _get_midpoints(image_fpath, keypoints)

    # debug - midpoints
    for key, value in midpoints.items():
        cv2.circle(image, tuple(value[0:2].astype(int)), 3, (255, 0, 255), -1)

    # step 3: load the data of norm_segm
    df_contour = pd.read_csv(fname_contour, index_col=0).astype('float32')
    contour_dict = df_contour.loc['impressionism']

    df_norm_segm = pd.read_csv(fname_norm_segm_impressionism, index_col=0)
    index_name = generate_index_name(image_fpath, person_index)
    contour_dict['scaler'] = df_norm_segm.loc[index_name]['scaler']

    # step 4: draw the norm_segm on the original image
    patches = _get_patches(midpoints, patch_size=patch_size, image_size=image_size, contour_dict=contour_dict)

    # debug - patches
    for midpoint in patches.items():
        _draw_rect(image, midpoint=midpoint, patch_size=patch_size)

    # debug - show the whole image
    cv2.imshow('Contour of {}'.format(painting_number), image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return patches


if __name__ == '__main__':

    # step 1: calculate and save the scalers in 'norm_segm_impressionism_xx.csv'
    # calc_scaler()

    # step 2: visualize contour on image
    thickness = 2

    # option 1 - visualize the contour
    # for a single image
    # visualize(infile='datasets/surf2nude/train/C/17245.jpg', category='impressionism')

    # for multiple images
    # for image_idx, infile in enumerate(glob.glob('datasets/surf2nude/train/C/*.jpg')):
    #
    #     painting_number = infile[infile.rfind('/') + 1:infile.rfind('.')]
    #     print('image {}:'.format(image_idx), painting_number)
    #
    #     visualize(infile=infile, category='impressionism')

    # option 2 - use the contour for the patch of segments
    # global settings
    image_size = 256
    patch_size = 32

    transforms_ = [ transforms.Resize(int(image_size), Image.BICUBIC),
                    transforms.CenterCrop(image_size),  # change from RandomCrop to CenterCrop
                    transforms.ToTensor() ]

    dataloader = DataLoader(ImageDataset(os.path.join('datasets', 'surf2nude'),
                                         transforms_=transforms_, unaligned=True),
                            batch_size=1, shuffle=True, num_workers=8)

    input_A = torch.Tensor(1, 3, image_size, image_size)
    input_B = torch.Tensor(1, 3, image_size, image_size)

    for i, batch in enumerate(dataloader):
        # A
        real_A = Variable(input_A.copy_(batch['A']))[0]
        path_A = batch['path_A'][0]
        shape_A = batch['shape_A']

        # B
        real_B = Variable(input_B.copy_(batch['B']))[0]
        path_B = batch['path_B'][0]
        shape_B = batch['shape_B']

        patches = get_segm_patches(image_tensor=real_B, image_fpath=path_B, image_shape=shape_B,
                                   image_size=image_size, patch_size=patch_size)