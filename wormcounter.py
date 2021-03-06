# -*- coding: utf-8 -*-
"""wormcounter.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qU_Zg6TN8H708oVAp4yQ2h2eGmkY7NCY
"""

import pandas as pd
import os
import tarfile
import shutil
import cv2
import numpy as np
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import segmentation_models_pytorch as smp
import math
import csv
import re
import tempfile

from skimage.segmentation import watershed
from skimage.morphology import skeletonize
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from scipy.signal import convolve2d

progDesc = """Convert a .tiff file into a .csv with worm counts or a .tar.gz file into a .csv of worm counts.
Prints the number of worms counted for a .tiff or the number of files counted for a .tar.gz.
Expects 2 arguments: the file path to take in the form of .tiff or .tar.gz and the file path to put out the .csv
Optionally you can put --verbose or -v in the case of a .tar.gz to output the .csv for each individual .tiff file within. This will make a directory with the name of the out path."""

def PathToImageDecoder(path):
  """Converts filepath bitstring with tiff image 
  into a workable image for processing."""
  return cv2.imdecode(path, cv2.IMREAD_GRAYSCALE)

def PathToImage(path):
  """Converts filepath with tiff image into a workable image for processing."""
  return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def MaskToConnectAndLoc(mask):
  """Take a bitmask image and convert it to a list of bitmask images that each 
  contain one fully connected component and a list of locations corresponding
  to a point within the fully connected component."""
  bmaskList = []
  areas = []
  #connectedComponents expects uint8 array
  ret, labels, _, centroids = cv2.connectedComponentsWithStats(mask)
  for label in range(1, ret):
    submask = np.array(labels, dtype=np.uint8)
    submask[label == labels] = 255
    bmaskList.append(submask)
    areas.append(int(np.sum(submask)))
  return bmaskList, centroids[1:].astype(int).tolist(), areas

def watershed_count(image):
  distance = ndi.distance_transform_edt(image)
  coords = peak_local_max(distance, labels=image)
  mask = np.zeros(distance.shape, dtype=bool)
  mask[tuple(coords.T)] = True
  markers, _ = ndi.label(mask)
  labels = watershed(-distance, markers, mask=image)
  return len(np.unique(labels))

def thin_kernal_count(image):
  image = skeletonize(image)
  # apply 3x3 kernal sum
  kernalized_image = convolve2d(image, np.ones((3,3), dtype=np.int8), mode='same', boundary='fill') == 2
  head_tail = np.logical_and(image, kernalized_image)
  # entry with value 2 is head or tail
  count = np.ceil(head_tail.sum() / 2)
  return count

def thresholding(img):
  thresh = 4482
  area = np.sum(img)
  if area > thresh:
    return watershed_count(img)
  return thin_kernal_count(img)

def ConnectToCount(mask):
  """Take a bitmask image that has one fully connected component and determine
  the count of worms in it."""
  return thresholding(mask)

def ReconstructMask(masklist, origShape):
  """Turn a collection of mask subimages into the whole image
  based on the origShape."""
  #We know subimages are 288 by 288
  numcol = math.ceil(origShape[1] / 288)
  tensorTuple = [torch.cat(masklist[i:(i+numcol)], dim=2) for i in range(0,len(masklist),numcol)]
  mask = torch.cat(tensorTuple, dim=1).squeeze(0)
  return torchvision.transforms.functional.crop(mask, 0, 0, origShape[0], origShape[1])

def EqualCrop(feature, height, width):
    """Crop to equal pieces. Takes a feature image and splits it into a list of
    subimages each with size height * width."""
    feature = torch.from_numpy(feature)
    newh = math.ceil(feature.shape[0] / height) * height
    neww = math.ceil(feature.shape[1] / width) * width
    feature = torchvision.transforms.Pad(padding = (0, 0, neww - feature.shape[1], newh - feature.shape[0]), fill = 255)(feature)
    features = [torchvision.transforms.functional.crop(feature, i, j, height, width) for i in range(0, feature.shape[0], height) for j in range(0, feature.shape[1], width)]
    return features

class MaskDataset(Dataset):
  """Dataset for a single image."""
  def __init__(self, image):
    #we have image, we want list of (288, 288) subsections of the image.
    self.features = EqualCrop(image, 288, 288)
  def __len__(self):
    return len(self.features)
  def __getitem__(self, idx):
    return torch.unsqueeze(1. - self.features[idx] / 255., 0)

def ImageToDataset(image):
  """Take a worm image and convert it to a pytorch Dataset suitable for 
  the Model. Is actually a collection because the model expects a certain size
  so we split the image into subimages."""
  return MaskDataset(image)

def DatasetToMask(dataset, model, device):
  """Take the maskDataset datapoints and get the image bitmask that defines
  the worms in it. This bitmask is a list of the datapoints' results in the same order."""
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
  masklist = []
  with torch.no_grad():
    for item in dataloader:
      masklist.append((torch.sigmoid(model(item.to(device))) > 0.5).squeeze(0)) #the model returns a sigmoid but we want a bitmask
  return masklist

def ImageToMask(image, device):
  """Take a worm image and determine an image bitmask that defines the worms
  in it."""
  dataset = ImageToDataset(image)
  imgToMask = smp.Linknet(in_channels=1, encoder_weights=None)
  imgToMask.load_state_dict(torch.load('linknet_resnet34_288.pt', map_location=torch.device(device)))
  imgToMask.to(device)
  imgToMask.eval()
  results = DatasetToMask(dataset, imgToMask, device)
  return ReconstructMask(results, image.shape)

def ImageToCSV(image, device, verbose=0):
  """Take a worm image and convert it to a CSV with the following fields
  [centroid location, worm count]. The first row will have 
  [Total, the total count]."""
  imgBitmask = ImageToMask(image, device).cpu().numpy().astype(np.uint8)
  fccs, locs, areas = MaskToConnectAndLoc(imgBitmask)
  totalWorms = ConnectToCount(imgBitmask)
  locs.insert(0, 'Total')
  areas.insert(0, sum(areas))
  df = pd.DataFrame(data = {'Total worms': totalWorms, 'Location': locs, 'Area': areas})
  return df.to_csv(index=False, quoting=csv.QUOTE_NONNUMERIC), imgBitmask

def SinglePathToCSV(inPath, outPath, verbose=0):
  """Takes an inPath with a single tiff image and converts 
  to a CSV at outPath following standards in ImageToCSV.
  Returns the total count of worms for the inPath image.
  If verbose is 2 also save the mask image."""
  if isinstance(inPath, str):
    image = PathToImage(inPath)
  else:
    image = PathToImageDecoder(inPath)
  device = 'cpu'
  if torch.cuda.is_available():
    device = 'cuda'
  csv, imgBitmask = ImageToCSV(image, device, verbose)
  if verbose > 0:
    with open(outPath, 'w+', newline='\n') as file:
      file.write(csv)
    if verbose >= 2:
      imgBitmask = imgBitmask * 255
      cv2.imwrite(outPath[:-4] + '.png', imgBitmask)
  return int(re.search(r'\d+', csv).group()) #Get the total count, which is the first number

def PathToCSV(inFilepath, outFilepath, verbose=0):
  """Takes a .tar.gz inFilepath and converts to a CSV containing the counts
  of the worms in each tiff file contained within. If verbose is 1 a CSV
  will be created for each tiff that contains the counts for individual
  connected components detected. If verbose is 2 an image of the detected worms
  will also be created for each tiff.
  Returns total tiff files counted."""
  total_counted = 0
  df = pd.DataFrame(columns=['filename', 'count'])
  tar = tarfile.open(inFilepath, "r:gz")
  dir = outFilepath[:-4] #remove .csv
  if verbose > 0:
    if os.path.exists(dir):
      shutil.rmtree(dir)
    os.makedirs(dir)
  with tempfile.TemporaryDirectory() as tempPath:
    #extract into the temp directory
    tar.extractall(tempPath)
    tar.close()
    #loop over all files in the temp directory and convert them
    for subdir, _, filenames in os.walk(tempPath):
      for origFilename in filenames:
        if '._' not in origFilename and origFilename.endswith('tiff'):
          total_counted += 1
          filePath = os.path.join(tempPath, subdir, origFilename)
          origFilename = origFilename.split("/")[-1]
          newFilename = dir + '/' + origFilename[:-5] + '.csv' #remove tiff and make csv
          currCount = SinglePathToCSV(filePath, newFilename, verbose)
          print(currCount, newFilename)
          df = df.append({'filename': origFilename, 'count': currCount}, ignore_index=True)
          df.sort_values('filename', ascending=True, inplace=True)
          #print it to csv
          df.to_csv(outFilepath, index=False)
  return total_counted

def DynamicConvert(inFilepath, outFilepath, verbose=0):
  """Takes a tiff or a tar.gz filepath and accordingly deals with them
  using PathToCSV for tar.gz and SinglePathToCSV for tiff. Returns count
  of worms for single tiff or count of files for .tar.gz"""
  if outFilepath.endswith(".csv"):
    if inFilepath.endswith(".tiff"):
      return str(SinglePathToCSV(inFilepath, outFilepath, verbose)) + " worms counted."
    if inFilepath.endswith(".tar.gz"):
      return str(PathToCSV(inFilepath, outFilepath, verbose)) + " files processed."
  else:
    raise OSError(errno.EINVAL, "Expected .csv filetype for output file", outFilepath)
  raise OSError(errno.EINVAL, "Expected .tiff filetype or .tar.gz filetype for input file", inFilepath)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description = progDesc)
  parser.add_argument('inFile', help="Input file location, expects .tar.gz or .tiff")
  parser.add_argument('outFile', help="Output file location, expects .csv")
  parser.add_argument('-v', '--verbose', dest='verbose', action="count", default=0, help="Indicates to make a .csv file for each .tiff within the .tar.gz. Does nothing for a .tiff input.")
  args = parser.parse_args()
  print(DynamicConvert(args.inFile, args.outFile, args.verbose))