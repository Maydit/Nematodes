from torch.utils.data import Dataset
import pickle
from itertools import chain
from utils import equal_crop, equal_crop_rotation
import torch
import torch.functional as F


class WormDataset(Dataset):
    '''
        torch dataset class used for inference to get prediction mask,
        use get_pred_mask in utils
    '''
    def __init__(self, input_path, label_path, crop_size, transform = None):
      with open(input_path, 'rb') as file:
        features = pickle.load(file)
      with open(label_path, 'rb') as file:
        labels = pickle.load(file)
      self.crop_size = crop_size
      self.transform = transform
      self.features = list(chain.from_iterable([equal_crop(feature, crop_size[0], crop_size[1]) for feature in self.filter(features)]))
      self.labels = list(chain.from_iterable([equal_crop(label, crop_size[0], crop_size[1]) for label in self.filter(labels)]))
      assert (len(self.features) == len(self.labels))

    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[0] >= self.crop_size[0] and
            img.shape[1] >= self.crop_size[1])]

    def __len__(self):
      return len(self.features)

    def __getitem__(self, idx):
      feature, label = self.features[idx], self.labels[idx] / 255.
      feature = F.threshold(255 - feature, 60., 0.)


      return torch.unsqueeze(feature / 255., 0), torch.unsqueeze(label, 0)

class AugmentedWormDataset(Dataset):
    '''
        torch dataset class for unet training
    '''
    def __init__(self, input_path, label_path, crop_size, transform = None):
      with open(input_path, 'rb') as file:
        features = pickle.load(file)
      with open(label_path, 'rb') as file:
        labels = pickle.load(file)
      self.crop_size = crop_size
      self.transform = transform
      self.features = list(chain.from_iterable([equal_crop_rotation(feature, crop_size[0], crop_size[1]) for feature in self.filter(features)]))
      self.labels = list(chain.from_iterable([equal_crop_rotation(label, crop_size[0], crop_size[1]) for label in self.filter(labels)]))
      assert (len(self.features) == len(self.labels))

    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[0] >= self.crop_size[0] and
            img.shape[1] >= self.crop_size[1])]

    def __len__(self):
      return len(self.features)

    def __getitem__(self, idx):
      feature, label = self.features[idx], self.labels[idx] / 255.
      feature = F.threshold(255 - feature, 60., 0.)


      return feature / 255., label
