import os
import json
import cv2
import numpy as np
from matplotlib.path import Path
import torch
import torchvision.transforms as transforms

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


def get_pred_mask(images_dataset, model):
    '''
        images_dataset: WormDataset
        model: trained unet
        image shape 576x576, 4 continuous pieces combine to original piece
        output image shape: 1x1x1152x1152
    '''
    pred_masks = []
    with torch.no_grad():
        for i in range(len(images_dataset) // 4):
        # get list of 4 cropped image
            image_lst = [images_dataset[i*4 + j][0] for j in range(4)]
            # output from model
            pred_lst = [model(torch.unsqueeze(img, 0)) for img in image_lst]
            # concat to one, each piece shape 1x1xHxW
            top_two = torch.cat(pred_lst[:2], dim=3)
            down_two = torch.cat(pred_lst[2:], dim=3)
            pred_masks.append(torch.cat((top_two, down_two), dim=2))

    return pred_masks


def equal_crop(feature, height, width):
    """crop to equal pieces"""
    feature = torch.from_numpy(feature)
    features = [transforms.functional.crop(feature, i, j, height, width) for i in range(0, feature.shape[0], height) for j in range(0, feature.shape[1], width)]
    return features

def equal_crop_rotation(feature, height, width):
    """crop to equal pieces, then rotate in four directions from each piece"""
    feature = torch.from_numpy(feature)
    features = [transforms.functional.crop(feature, i, j, height, width) for i in range(0, feature.shape[0], height) for j in range(0, feature.shape[1], width)]
    augmented_features = []
    for feature in features:
      feature = torch.unsqueeze(feature, 0)
      for i in range(4):
        augmented_features.append(transforms.functional.rotate(feature, 90. * i,))

      augmented_features.append(transforms.functional.hflip(feature))
      augmented_features.append(transforms.functional.vflip(feature))
    return augmented_features