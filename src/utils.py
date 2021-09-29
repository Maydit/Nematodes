import os
import json
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

def load_image_from_dir(path):
  images = []
  for filename in os.listdir(path):
    image = cv2.imread(os.path.join(path, filename), 0)
    if image:
      images.append(image)

  return images

def fill_in_boundary(points, array):
    n = len(points)
    for i in range(n):
        # draw lines
        x0, y0 = points[i]
        x0, y0 = int(x0), int(x1)
        x1, y1 = int(x1), int(y1)
        x1, y1 = points[(i+1)%n]
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                array[x, y] = 1
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                array[x, y] = 1
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy        
        array[x, y] = 1

    return binary_fill_holes(array)



def output_groudtruth_image_from_json(json_obj):
    # create binary array where annotated worm pixels are 1s
    shapes = json_obj["shapes"]
    height = json_obj["imageHeight"]
    width = json_obj["imageWidth"]
    truth_image = np.zeros((height, width), dtype=np.int8)
    for shape in shapes:
        points = shape["points"]
        truth_image = fill_in_boundary(points, truth_image)

    return truth_image

def read_json(path):
    truth_dict = dict()
    for filename in os.listdir(path):
        file_lst = filename.split('.')
        if file_lst[-1] == 'json':
            with open(os.path.join(path, filename)) as json_file:
                data = json.load(json_file)
                truth_dict[".".join(file_lst[:-1])] = output_groudtruth_image_from_json(data)

    return truth_dict
