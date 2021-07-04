import os, cv2, re
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
import pycocotools.mask as mask_util
from scipy.stats import wasserstein_distance


# the path to the data of contour.csv
fname_contour = os.path.join('output', 'contour.csv')
# the path to the data of norm_segm.csv
fname_norm_segm_coco_woman = os.path.join('output', 'norm_segm_coco_woman.csv')

# dataset setting
coco_folder = os.path.join('datasets', 'coco')

# dense_pose annotation
dp_coco = COCO(os.path.join(coco_folder, 'annotations', 'densepose_train2014.json'))


# coarse segmentation:
# 0 = Background
# 1 = Torso,
# 2 = Right Hand, 3 = Left Hand, 4 = Left Foot, 5 = Right Foot,
# 6 = Upper Leg Right, 7 = Upper Leg Left, 8 = Lower Leg Right, 9 = Lower Leg Left,
# 10 = Upper Arm Left, 11 = Upper Arm Right, 12 = Lower Arm Left, 13 = Lower Arm Right,
# 14 = Head

COARSE_ID = [
    'Background',
    'Torso',
    'RHand', 'LHand', 'LFoot', 'RFoot',
    'RThigh', 'LThigh', 'RCalf', 'LCalf',
    'LUpperArm', 'RUpperArm', 'LLowerArm', 'RLowerArm',
    'Head'
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


# DensePose JOINT_ID
JOINT_ID = [
    'Nose', 'LEye', 'REye', 'LEar', 'REar',
    'LShoulder', 'RShoulder', 'LElbow', 'RElbow', 'LWrist', 'RWrist',
    'LHip', 'RHip', 'LKnee', 'RKnee', 'LAnkle', 'RAnkle'
]


def _get_dp_mask(polys):

    mask_gen = np.zeros([256,256])

    for i in range(1,15):

        if(polys[i-1]):
            current_mask = mask_util.decode(polys[i-1])
            mask_gen[current_mask>0] = i

    return mask_gen


def _segm_xy(segm, segm_id, box_xywh):

    # bbox
    box_x, box_y, box_w, box_h = np.array(box_xywh).astype(int)

    y, x = np.where(segm == segm_id)

    # translate from the bbox coordinate to the original image coordinate
    return list(zip(x+box_x, y+box_y))


def _get_dict_of_segm_and_keypoints(segm, keypoints, box_xywh):

    segm_xy_list = []

    bg_xy = [] # 0
    segm_xy_list.append(bg_xy)

    torso_xy = _segm_xy(segm=segm, segm_id=1, box_xywh=box_xywh)
    segm_xy_list.append(torso_xy)

    r_hand_xy = [] # 2
    l_hand_xy = [] # 3
    l_foot_xy = [] # 4
    r_foot_xy = [] # 5
    segm_xy_list.append(r_hand_xy)
    segm_xy_list.append(l_hand_xy)
    segm_xy_list.append(l_foot_xy)
    segm_xy_list.append(r_foot_xy)

    r_thigh_xy = _segm_xy(segm=segm, segm_id=6, box_xywh=box_xywh)
    l_thigh_xy = _segm_xy(segm=segm, segm_id=7, box_xywh=box_xywh)
    r_calf_xy = _segm_xy(segm=segm, segm_id=8, box_xywh=box_xywh)
    l_calf_xy = _segm_xy(segm=segm, segm_id=9, box_xywh=box_xywh)
    segm_xy_list.append(r_thigh_xy)
    segm_xy_list.append(l_thigh_xy)
    segm_xy_list.append(r_calf_xy)
    segm_xy_list.append(l_calf_xy)

    l_upper_arm_xy = _segm_xy(segm=segm, segm_id=10, box_xywh=box_xywh)
    r_upper_arm_xy = _segm_xy(segm=segm, segm_id=11, box_xywh=box_xywh)
    l_lower_arm_xy = _segm_xy(segm=segm, segm_id=12, box_xywh=box_xywh)
    r_lower_arm_xy = _segm_xy(segm=segm, segm_id=13, box_xywh=box_xywh)
    segm_xy_list.append(l_upper_arm_xy)
    segm_xy_list.append(r_upper_arm_xy)
    segm_xy_list.append(l_lower_arm_xy)
    segm_xy_list.append(r_lower_arm_xy)

    head_xy = _segm_xy(segm=segm, segm_id=14, box_xywh=box_xywh)
    segm_xy_list.append(head_xy)

    # segments dictionary
    segm_xy_dict = dict(zip(COARSE_ID, segm_xy_list))

    # keypoints dictionary
    keypoints_dict = dict(zip(JOINT_ID, zip(keypoints[0::3].copy(), keypoints[1::3].copy(), keypoints[2::3].copy())))
    keypoints_dict = {key:np.array(value) for key, value in keypoints_dict.items()}

    # infer the keypoints of neck and midhip, which are missing!
    keypoints_dict['Neck'] = ((keypoints_dict['LShoulder'] + keypoints_dict['RShoulder']) / 2).astype(int)
    keypoints_dict['MidHip'] = ((keypoints_dict['LHip'] + keypoints_dict['RHip']) / 2).astype(int)

    return segm_xy_dict, keypoints_dict


def _segm_xy_centroid(segm_xy):

    size = len(segm_xy)

    x = [x for x, y in segm_xy if not np.isnan(x)]
    y = [y for x, y in segm_xy if not np.isnan(y)]
    centroid = (sum(x) / size, sum(y) / size)

    return centroid


def _get_dict_of_midpoints(segm_xy_dict, keypoints_dict):

    midpoints_dict = {}

    # head centroid
    head_centroid_x, head_centroid_y = _segm_xy_centroid(segm_xy_dict['Head'])
    midpoints_dict['Head'] = np.array([head_centroid_x, head_centroid_y])

    # torso midpoint
    midpoints_dict['Torso'] = (keypoints_dict['Neck'] + keypoints_dict['MidHip']) / 2

    # upper limbs
    midpoints_dict['RUpperArm'] = (keypoints_dict['RShoulder'] + keypoints_dict['RElbow']) / 2
    midpoints_dict['RLowerArm'] = (keypoints_dict['RElbow'] + keypoints_dict['RWrist']) / 2
    midpoints_dict['LUpperArm'] = (keypoints_dict['LShoulder'] + keypoints_dict['LElbow']) / 2
    midpoints_dict['LLowerArm'] = (keypoints_dict['LElbow'] + keypoints_dict['LWrist']) / 2

    # lower limbs
    midpoints_dict['RThigh'] = (keypoints_dict['RHip'] + keypoints_dict['RKnee']) / 2
    midpoints_dict['RCalf'] = (keypoints_dict['RKnee'] + keypoints_dict['RAnkle']) / 2
    midpoints_dict['LThigh'] = (keypoints_dict['LHip'] + keypoints_dict['LKnee']) / 2
    midpoints_dict['LCalf'] = (keypoints_dict['LKnee'] + keypoints_dict['LAnkle']) / 2

    return midpoints_dict


def _get_rotated_angles(keypoints, midpoints):

    rotated_angles = {}

    # head
    reference_point = np.array(keypoints['Neck']) + np.array((0, -100, 0))
    rad, deg = _calc_angle(point1=midpoints['Head'], center=keypoints['Neck'], point2=reference_point)
    # rotate back to original in reverse direction
    rotated_angles['Head'] = -deg

    # torso
    reference_point = np.array(keypoints['MidHip']) + np.array((0, -100, 0))
    rad, deg = _calc_angle(point1=keypoints['Neck'], center=keypoints['MidHip'], point2=reference_point)
    rotated_angles['Torso'] = -deg

    # upper limbs
    reference_point = np.array(keypoints['RShoulder']) + np.array((-100, 0, 0))
    rad, deg = _calc_angle(point1=keypoints['RElbow'], center=keypoints['RShoulder'], point2=reference_point)
    rotated_angles['RUpperArm'] = -deg

    reference_point = np.array(keypoints['RElbow']) + np.array((-100, 0, 0))
    rad, deg = _calc_angle(point1=keypoints['RWrist'], center=keypoints['RElbow'], point2=reference_point)
    rotated_angles['RLowerArm'] = -deg

    reference_point = np.array(keypoints['LShoulder']) + np.array((100, 0, 0))
    rad, deg = _calc_angle(point1=keypoints['LElbow'], center=keypoints['LShoulder'], point2=reference_point)
    rotated_angles['LUpperArm'] = -deg

    reference_point = np.array(keypoints['LElbow']) + np.array((100, 0, 0))
    rad, deg = _calc_angle(point1=keypoints['LWrist'], center=keypoints['LElbow'], point2=reference_point)
    rotated_angles['LLowerArm'] = -deg

    # lower limbs
    reference_point = np.array(keypoints['RHip']) + np.array((0, 100, 0))
    rad, deg = _calc_angle(point1=keypoints['RKnee'], center=keypoints['RHip'], point2=reference_point)
    rotated_angles['RThigh'] = -deg

    reference_point = np.array(keypoints['RKnee']) + np.array((0, 100, 0))
    rad, deg = _calc_angle(point1=keypoints['RAnkle'], center=keypoints['RKnee'], point2=reference_point)
    rotated_angles['RCalf'] = -deg

    reference_point = np.array(keypoints['LHip']) + np.array((0, 100, 0))
    rad, deg = _calc_angle(point1=keypoints['LKnee'], center=keypoints['LHip'], point2=reference_point)
    rotated_angles['LThigh'] = -deg

    reference_point = np.array(keypoints['LKnee']) + np.array((0, 100, 0))
    rad, deg = _calc_angle(point1=keypoints['LAnkle'], center=keypoints['LKnee'], point2=reference_point)
    rotated_angles['LCalf'] = -deg

    return rotated_angles


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


def _get_head_segm_centroid(head_xy):

    # get the centroid of the head segment
    head_centroid_x, head_centroid_y = _segm_xy_centroid(head_xy)

    return head_centroid_x, head_centroid_y


def _get_dict_of_rotated_angles(keypoints_dict, midpoints_dict):

    rotated_angles_dict = {}

    # head
    reference_point = np.array(keypoints_dict['Neck']) + np.array((0, -100, 0))
    rad, deg = _calc_angle(point1=midpoints_dict['Head'], center=keypoints_dict['Neck'], point2=reference_point)
    rotated_angles_dict['Head'] = rad

    # torso
    reference_point = np.array(keypoints_dict['MidHip']) + np.array((0, -100, 0))
    rad, deg = _calc_angle(point1=keypoints_dict['Neck'], center=keypoints_dict['MidHip'], point2=reference_point)
    rotated_angles_dict['Torso'] = rad

    # upper limbs
    reference_point = np.array(keypoints_dict['RShoulder']) + np.array((-100, 0, 0))
    rad, deg = _calc_angle(point1=keypoints_dict['RElbow'], center=keypoints_dict['RShoulder'], point2=reference_point)
    rotated_angles_dict['RUpperArm'] = rad

    reference_point = np.array(keypoints_dict['RElbow']) + np.array((-100, 0, 0))
    rad, deg = _calc_angle(point1=keypoints_dict['RWrist'], center=keypoints_dict['RElbow'], point2=reference_point)
    rotated_angles_dict['RLowerArm'] = rad

    reference_point = np.array(keypoints_dict['LShoulder']) + np.array((100, 0, 0))
    rad, deg = _calc_angle(point1=keypoints_dict['LElbow'], center=keypoints_dict['LShoulder'], point2=reference_point)
    rotated_angles_dict['LUpperArm'] = rad

    reference_point = np.array(keypoints_dict['LElbow']) + np.array((100, 0, 0))
    rad, deg = _calc_angle(point1=keypoints_dict['LWrist'], center=keypoints_dict['LElbow'], point2=reference_point)
    rotated_angles_dict['LLowerArm'] = rad

    # lower limbs
    reference_point = np.array(keypoints_dict['RHip']) + np.array((0, 100, 0))
    rad, deg = _calc_angle(point1=keypoints_dict['RKnee'], center=keypoints_dict['RHip'], point2=reference_point)
    rotated_angles_dict['RThigh'] = rad

    reference_point = np.array(keypoints_dict['RKnee']) + np.array((0, 100, 0))
    rad, deg = _calc_angle(point1=keypoints_dict['RAnkle'], center=keypoints_dict['RKnee'], point2=reference_point)
    rotated_angles_dict['RCalf'] = rad

    reference_point = np.array(keypoints_dict['LHip']) + np.array((0, 100, 0))
    rad, deg = _calc_angle(point1=keypoints_dict['LKnee'], center=keypoints_dict['LHip'], point2=reference_point)
    rotated_angles_dict['LThigh'] = rad

    reference_point = np.array(keypoints_dict['LKnee']) + np.array((0, 100, 0))
    rad, deg = _calc_angle(point1=keypoints_dict['LAnkle'], center=keypoints_dict['LKnee'], point2=reference_point)
    rotated_angles_dict['LCalf'] = rad

    return rotated_angles_dict


def _rotate_image(image, center, angle):

  rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
  rotated_image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

  return rotated_image


def _get_patch_img_list(image, midpoints, rotated_angles, dict_norm_segm):

    patch_img_list = []

    # scaler
    scaler = 1 / dict_norm_segm['scaler']

    # head
    rect = ((midpoints['Head'][0], midpoints['Head'][1]),
            (dict_norm_segm['Head_w'] * scaler, dict_norm_segm['Head_h'] * scaler),
            rotated_angles['Head'])
    box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    rotated_image = _rotate_image(image, tuple(midpoints['Head'][0:2]), rotated_angles['Head'])
    x1, y1 = np.min(box, axis=0)
    x2, y2 = np.max(box, axis=0)
    patch_img_list.append(rotated_image[y1:y2, x1:x2])

    # torso
    rect = ((midpoints['Torso'][0], midpoints['Torso'][1]),
            (dict_norm_segm['Torso_w'] * scaler, dict_norm_segm['Torso_h'] * scaler),
            rotated_angles['Torso'])
    box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    rotated_image = _rotate_image(image, tuple(midpoints['Torso'][0:2]), rotated_angles['Torso'])
    x1, y1 = np.min(box, axis=0)
    x2, y2 = np.max(box, axis=0)
    patch_img_list.append(rotated_image[y1:y2, x1:x2])

    # upper limbs
    rect = ((midpoints['RUpperArm'][0], midpoints['RUpperArm'][1]),
            (dict_norm_segm['RUpperArm_w'] * scaler, dict_norm_segm['RUpperArm_h'] * scaler),
            rotated_angles['RUpperArm'])
    box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    rotated_image = _rotate_image(image, tuple(midpoints['RUpperArm'][0:2]), rotated_angles['RUpperArm'])
    x1, y1 = np.min(box, axis=0)
    x2, y2 = np.max(box, axis=0)
    patch_img_list.append(rotated_image[y1:y2, x1:x2])

    rect = ((midpoints['RLowerArm'][0], midpoints['RLowerArm'][1]),
            (dict_norm_segm['RLowerArm_w'] * scaler, dict_norm_segm['RLowerArm_h'] * scaler),
            rotated_angles['RLowerArm'])
    box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    rotated_image = _rotate_image(image, tuple(midpoints['RLowerArm'][0:2]), rotated_angles['RLowerArm'])
    x1, y1 = np.min(box, axis=0)
    x2, y2 = np.max(box, axis=0)
    patch_img_list.append(rotated_image[y1:y2, x1:x2])

    rect = ((midpoints['LUpperArm'][0], midpoints['LUpperArm'][1]),
            (dict_norm_segm['LUpperArm_w'] * scaler, dict_norm_segm['LUpperArm_h'] * scaler),
            rotated_angles['LUpperArm'])
    box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    rotated_image = _rotate_image(image, tuple(midpoints['LUpperArm'][0:2]), rotated_angles['LUpperArm'])
    x1, y1 = np.min(box, axis=0)
    x2, y2 = np.max(box, axis=0)
    patch_img_list.append(rotated_image[y1:y2, x1:x2])

    rect = ((midpoints['LLowerArm'][0], midpoints['LLowerArm'][1]),
            (dict_norm_segm['LLowerArm_w'] * scaler, dict_norm_segm['LLowerArm_h'] * scaler),
            rotated_angles['LLowerArm'])
    box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    rotated_image = _rotate_image(image, tuple(midpoints['LLowerArm'][0:2]), rotated_angles['LLowerArm'])
    x1, y1 = np.min(box, axis=0)
    x2, y2 = np.max(box, axis=0)
    patch_img_list.append(rotated_image[y1:y2, x1:x2])

    # lower limbs
    rect = ((midpoints['RThigh'][0], midpoints['RThigh'][1]),
            (dict_norm_segm['RThigh_w'] * scaler, dict_norm_segm['RThigh_h'] * scaler),
            rotated_angles['RThigh'])
    box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    rotated_image = _rotate_image(image, tuple(midpoints['RThigh'][0:2]), rotated_angles['RThigh'])
    x1, y1 = np.min(box, axis=0)
    x2, y2 = np.max(box, axis=0)
    patch_img_list.append(rotated_image[y1:y2, x1:x2])

    rect = ((midpoints['RCalf'][0], midpoints['RCalf'][1]),
            (dict_norm_segm['RCalf_w'] * scaler, dict_norm_segm['RCalf_h'] * scaler),
            rotated_angles['RCalf'])
    box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    rotated_image = _rotate_image(image, tuple(midpoints['RCalf'][0:2]), rotated_angles['RCalf'])
    x1, y1 = np.min(box, axis=0)
    x2, y2 = np.max(box, axis=0)
    patch_img_list.append(rotated_image[y1:y2, x1:x2])

    rect = ((midpoints['LThigh'][0], midpoints['LThigh'][1]),
            (dict_norm_segm['LThigh_w'] * scaler, dict_norm_segm['LThigh_h'] * scaler),
            rotated_angles['LThigh'])
    box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    rotated_image = _rotate_image(image, tuple(midpoints['LThigh'][0:2]), rotated_angles['LThigh'])
    x1, y1 = np.min(box, axis=0)
    x2, y2 = np.max(box, axis=0)
    patch_img_list.append(rotated_image[y1:y2, x1:x2])

    rect = ((midpoints['LCalf'][0], midpoints['LCalf'][1]),
            (dict_norm_segm['LCalf_w'] * scaler, dict_norm_segm['LCalf_h'] * scaler),
            rotated_angles['LCalf'])
    box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    rotated_image = _rotate_image(image, tuple(midpoints['LCalf'][0:2]), rotated_angles['LCalf'])
    x1, y1 = np.min(box, axis=0)
    x2, y2 = np.max(box, axis=0)
    patch_img_list.append(rotated_image[y1:y2, x1:x2])

    return patch_img_list


def calculate_hist_loss(image_id, person_index, from_category, to_category):

    entry = dp_coco.loadImgs(image_id)[0]

    image_fpath = os.path.join('datasets/surf2nude/train/A', entry['file_name'])

    print('image_fpath:', image_fpath)

    img_color = cv2.imread(image_fpath)

    dp_annotation_ids = dp_coco.getAnnIds(imgIds=entry['id'])
    dp_annotations = dp_coco.loadAnns(dp_annotation_ids)

    # iterate through all the people in one image
    for dp_annotation in dp_annotations:

        # 1. keypoints
        keypoints = dp_annotation['keypoints']

        # 2. bbox
        bbox_xywh = np.array(dp_annotation["bbox"]).astype(int)

        # 3. segments of dense_pose
        if ('dp_masks' in dp_annotation.keys()):
            mask = _get_dp_mask(dp_annotation['dp_masks'])

            x1, y1, x2, y2 = bbox_xywh[0], bbox_xywh[1], bbox_xywh[0] + bbox_xywh[2], bbox_xywh[1] + bbox_xywh[3]

            x2 = min([x2, img_color.shape[1]])
            y2 = min([y2, img_color.shape[0]])

            segm = cv2.resize(mask, (int(x2 - x1), int(y2 - y1)), interpolation=cv2.INTER_NEAREST)

        # step 1. get segm_xy + keypoints dict
        segm_xy_dict, keypoints_dict = _get_dict_of_segm_and_keypoints(segm, keypoints, bbox_xywh)

        # step 2: get all the midpoints
        midpoints_dict = _get_dict_of_midpoints(segm_xy_dict, keypoints_dict)

        # step 3: get the rotation angles
        rotated_angles = _get_rotated_angles(keypoints_dict, midpoints_dict)

        # step 4: load the data of contour
        df_contour = pd.read_csv(fname_contour, index_col=0).astype('float32')
        dict_from_contour = df_contour.loc[from_category]
        dict_to_contour = df_contour.loc[to_category]

        df_norm_segm = pd.read_csv(fname_norm_segm_coco_woman, index_col=0)
        index_name = generate_index_name(image_id, person_index)
        dict_from_contour['scaler'] = df_norm_segm.loc[index_name]['scaler']
        dict_to_contour['scaler'] = df_norm_segm.loc[index_name]['scaler']

        # step 5: draw the norm_segm on the original image
        from_patch_img_list = _get_patch_img_list(img_color, midpoints_dict, rotated_angles, dict_from_contour)
        to_patch_img_list = _get_patch_img_list(img_color, midpoints_dict, rotated_angles, dict_to_contour)

        # test - histogram
        hist1 = cv2.calcHist([from_patch_img_list[7]], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([to_patch_img_list[7]], [0], None, [256], [0, 256])
        print('dist:', wasserstein_distance(hist2.transpose()[0], hist2.transpose()[0]))

        # test - patch
        image_window = 'image'
        cv2.imshow(image_window, to_patch_img_list[7])
        cv2.setWindowProperty(image_window, cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def generate_index_name(image_id, person_index):

    index_name = '{}_{}'.format(image_id, person_index)

    return index_name


if __name__ == '__main__':

    # settings
    thickness = 4

    # category = coco
    image_id = 362884

    calculate_hist_loss(image_id=image_id, person_index=1, from_category='coco', to_category='test')