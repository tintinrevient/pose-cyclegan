from PIL import Image
import cv2
import math
import numpy as np

src_color = (255, 0, 255)
dst_color = (0, 255, 0)

def _euclidian(point1, point2):
  return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


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


def vector_length(vector):
  return math.sqrt(vector[0] ** 2 + vector[1] ** 2)


def points_distance(point1, point2):
  return vector_length((point1[0] - point2[0],point1[1] - point2[1]))


def clamp(value, minimum, maximum):
  return max(min(value,maximum),minimum)


# Warp an image according to the given points and shift vectors
# @param image input image
# @param points list of (x, y, dx, dy) tuples
# @return warped image
def warp(image, points):

  result = Image.new("RGB",image.size,"black")

  image_pixels = image.load()
  result_pixels = result.load()

  for y in range(image.size[1]):
    for x in range(image.size[0]):

      offset = [0,0]

      for point in points:
        point_position = (point[0] + point[2],point[1] + point[3])
        shift_vector = (point[2],point[3])

        helper = 1.0 / (3 * (points_distance((x,y),point_position) / vector_length(shift_vector)) ** 4 + 1)

        offset[0] -= helper * shift_vector[0]
        offset[1] -= helper * shift_vector[1]

      coords = (clamp(x + int(offset[0]),0,image.size[0] - 1),clamp(y + int(offset[1]),0,image.size[1] - 1))

      result_pixels[x,y] = image_pixels[coords[0],coords[1]]

  return result


def get_points():

  points = []

  # (x, y)
  lsho = (294, 110)
  rsho = (350, 111)

  lelb = (290, 171)
  relb = (382, 176)

  lhip = (309, 220)
  rhip = (355, 218)

  lknee = (303, 308)
  rknee = (342, 307)

  lankle = (292, 401)
  rankle = (329, 398)

  # debug
  image = cv2.imread("warp/83964.jpg")

  # RUpperArm: rsho + relb
  dist = _euclidian(rsho, relb)
  rad, deg = _calc_angle(relb, rsho, np.array(rsho) + np.array([0, 100]))
  scaler = 23 / 18 - 1 # RUpperArm_h in 'norm_segm_coco_woman.csv' and 'contour.csv' for '83964_1' and 'impressionism'
  dx, dy = dist * scaler * np.sin(abs(rad)), dist * scaler * np.cos(abs(rad))
  points.append((relb[0], relb[1], dx, dy))

  # debug
  cv2.circle(image, rsho, radius=3, color=src_color, thickness=-1)
  cv2.circle(image, relb, radius=3, color=src_color, thickness=-1)
  cv2.circle(image, (int(relb[0] + dx), int(relb[1] + dy)), radius=3, color=dst_color, thickness=-1)

  # LUpperArm: lsho + lelb
  dist = _euclidian(lsho, lelb)
  rad, deg = _calc_angle(lelb, lsho, np.array(lsho) + np.array([0, 100]))
  scaler = 23 / 18 - 1  # LUpperArm_h in 'norm_segm_coco_woman.csv' and 'contour.csv' for '83964_1' and 'impressionism'
  dx, dy = - dist * scaler * np.sin(abs(rad)), dist * scaler * np.cos(abs(rad))
  points.append((lelb[0], lelb[1], dx, dy))

  # debug
  cv2.circle(image, lsho, radius=3, color=src_color, thickness=-1)
  cv2.circle(image, lelb, radius=3, color=src_color, thickness=-1)
  cv2.circle(image, (int(lelb[0] + dx), int(lelb[1] + dy)), radius=3, color=dst_color, thickness=-1)

  # RThigh: rhip + rknee
  dist = _euclidian(rhip, rknee)
  rad, deg = _calc_angle(rknee, rhip, np.array(rhip) + np.array([0, 100]))
  scaler =  1 - 80 / 90  # LUpperArm_h in 'norm_segm_coco_woman.csv' and 'contour.csv' for '83964_1' and 'impressionism'
  dx, dy = - dist * scaler * np.sin(abs(rad)), - dist * scaler * np.cos(abs(rad))
  points.append((rknee[0], rknee[1], dx, dy))

  # debug
  cv2.circle(image, rhip, radius=3, color=src_color, thickness=-1)
  cv2.circle(image, rknee, radius=3, color=src_color, thickness=-1)
  cv2.circle(image, (int(rknee[0] + dx), int(rknee[1] + dy)), radius=3, color=dst_color, thickness=-1)

  # LThigh: lhip + lknee
  dist = _euclidian(lhip, lknee)
  rad, deg = _calc_angle(lknee, lhip, np.array(lhip) + np.array([0, 100]))
  scaler = 1 - 80 / 90  # LUpperArm_h in 'norm_segm_coco_woman.csv' and 'contour.csv' for '83964_1' and 'impressionism'
  dx, dy = - dist * scaler * np.sin(abs(rad)), - dist * scaler * np.cos(abs(rad))
  points.append((lknee[0], lknee[1], dx, dy))

  # debug
  cv2.circle(image, lhip, radius=3, color=src_color, thickness=-1)
  cv2.circle(image, lknee, radius=3, color=src_color, thickness=-1)
  cv2.circle(image, (int(lknee[0] + dx), int(lknee[1] + dy)), radius=3, color=dst_color, thickness=-1)

  # RCalf: rknee + rankle
  dist = _euclidian(rknee, rankle)
  rad, deg = _calc_angle(rankle, rknee, np.array(rknee) + np.array([0, 100]))
  scaler = 23 / 18 - 1  # LUpperArm_h in 'norm_segm_coco_woman.csv' and 'contour.csv' for '83964_1' and 'impressionism'
  dx, dy = - dist * scaler * np.sin(abs(rad)), dist * scaler * np.cos(abs(rad))
  points.append((rankle[0], rankle[1], dx, dy))

  # debug
  cv2.circle(image, rankle, radius=3, color=src_color, thickness=-1)
  cv2.circle(image, (int(rankle[0] + dx), int(rankle[1] + dy)), radius=3, color=dst_color, thickness=-1)

  # LCalf: lknee + lankle
  dist = _euclidian(lknee, lankle)
  rad, deg = _calc_angle(lankle, lknee, np.array(lknee) + np.array([0, 100]))
  scaler = 23 / 18 - 1  # LUpperArm_h in 'norm_segm_coco_woman.csv' and 'contour.csv' for '83964_1' and 'impressionism'
  dx, dy = - dist * scaler * np.sin(abs(rad)), dist * scaler * np.cos(abs(rad))
  points.append((lankle[0], lankle[1], dx, dy))

  # debug
  cv2.circle(image, lankle, radius=3, color=src_color, thickness=-1)
  cv2.circle(image, (int(lankle[0] + dx), int(lankle[1] + dy)), radius=3, color=dst_color, thickness=-1)

  cv2.imshow('debug', image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

  return points


# Modigliani
image = Image.open("warp/83964.jpg")

points = get_points()

image = warp(image, points)
image.save("warped.png", "PNG")

# Lempicka
# image = Image.open("warp/83964.jpg")
# image = warp(image,[(356, 144, -10, 0), (376, 144, 10, 0), # RUpperArm
#                     (282, 141, -10, 0), (302, 141, 10, 0), # LUpperArm
#                     (339, 263, -10, 0), (359, 263, 10, 0), # RThigh
#                     (296, 264, -10, 0), (316, 264, 10, 0) # LThigh
#                     ])
# image.save("warped.png","PNG")