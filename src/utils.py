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


def get_pred_mask(images_dataset, model, p=4, binary=True, device='cuda:0', output_device='cpu'):
    '''
        images_dataset: WormDataset
        model: trained unet
        p: crop size in 1d
        binary: if True output binary mask, else sigmoid score
        image shape 288x288, 16 continuous pieces combine to original piece if p = 4
        output image shape: 1x1x1152x1152
    '''
    pred_masks = []
    p_sq = p ** 2
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for i in range(len(images_dataset) // p_sq):
            # get list of p square cropped image
            image_lst = [images_dataset[i*p_sq + j][0] for j in range(p_sq)]
            # output from model
            if binary:
                pred_lst = [torch.sigmoid(model(torch.unsqueeze(img.to(device), 0))) > 0.5 for img in image_lst]
            else:
                pred_lst = [torch.sigmoid(model(torch.unsqueeze(img.to(device), 0))) for img in image_lst]
            # concat to one, each piece shape 1x1xHxW
            tensor_tuple = [torch.cat(pred_lst[i:(i+p)], dim=3) for i in range(0,p_sq,p)]
            pred_masks.append(torch.cat(tensor_tuple, dim=2).to(output_device))

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