import os
import json
import cv2
import numpy as np
from matplotlib.path import Path

def load_image_from_dir(path):
  images = []
  for filename in os.listdir(path):
    image = cv2.imread(os.path.join(path, filename), 0)
    if image:
      images.append(image)

  return images

def fill_in_boundary(points, height, width):
    poly_path = Path(points)
    x, y = np.mgrid[:height, :width]
    coord = np.hstack((x.reshape(-1,1), y.reshape(-1,1)))
    mask = poly_path.contains_points(coord)

    return mask.reshape(height, width)



def output_groudtruth_image_from_json(json_obj):
    # create binary array where annotated worm pixels are 1s
    shapes = json_obj["shapes"]
    height = json_obj["imageHeight"]
    width = json_obj["imageWidth"]
    truth_image = np.zeros((height, width), dtype=np.int8)
    for shape in shapes:
        points = shape["points"]
        truth_image = np.logical_or(fill_in_boundary(points, height, width), truth_image)

    return truth_image.T

def read_json(path):
    truth_dict = dict()
    for filename in os.listdir(path):
        file_lst = filename.split('.')
        if file_lst[-1] == 'json':
            with open(os.path.join(path, filename)) as json_file:
                data = json.load(json_file)
                truth_dict[".".join(file_lst[:-1])] = output_groudtruth_image_from_json(data)

    return truth_dict
