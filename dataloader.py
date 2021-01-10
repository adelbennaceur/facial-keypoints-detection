import os
import random
from math import *

import torch
import torch.nn as nn
from torchvision import transforms, utils
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt
import torchvision.transforms.functional as tf
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import albumentations as A
from PIL import Image
import cv2
import imutils


def get_train_val(dataset, batch_size=64):

    len_valid_set = int(0.1 * len(dataset))
    len_train_set = len(dataset) - len_valid_set

    train_set, validation_set = random_split(dataset, [len_train_set, len_valid_set])

    # shuffle and batch the datasets
    train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(validation_set, batch_size, shuffle=True, num_workers=4)

    return train_loader, valid_loader

class Dataset300W(Dataset):

    def __init__(self, transform=None):

        
        self.root_dir = './dataset/ibug_300W_large_face_landmark_dataset'
        
        tree = ET.parse(os.path.join(self.root_dir, "labels_ibug_300W_train.xml"))
        root = tree.getroot()

        self.image_filenames = []
        self.landmarks = []
        self.crops = []
        self.transform = transform

        for filename in root[2]:
            self.image_filenames.append(os.path.join(self.root_dir, filename.attrib['file']))

            self.crops.append(filename[0].attrib)

            landmark = []
            for num in range(68):
                x_coordinate = int(filename[0][num].attrib['x'])
                y_coordinate = int(filename[0][num].attrib['y'])
                xy = (x_coordinate, y_coordinate)
                landmark.append(xy)
            self.landmarks.append(landmark)

        self.landmarks = np.array(self.landmarks).astype('float32')     
        
        assert len(self.image_filenames) == len(self.landmarks)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image = cv2.imread(self.image_filenames[index])
        landmarks = self.landmarks[index]
        
        if image is None:
            print(self.image_filenames[index])

        if self.transform:
            image, landmarks = self.transform(image, landmarks, self.crops[index])
            #transformed = self.transform(image=image, keypoints=landmarks)
            #image , landmarks = transformed["image"] , transformed["keypoints"]

        landmarks = torch.tensor([landmarks])
        landmarks = landmarks.squeeze()
        
        sample = {"image": image, "landmarks" : landmarks}

        return sample



class Transforms:
    def __init__(self):
        pass

    def rotate(self, image, landmarks, angle):
        angle = random.uniform(-angle, +angle)

        transformation_matrix = torch.tensor(
            [
                [+cos(radians(angle)), -sin(radians(angle))],
                [+sin(radians(angle)), +cos(radians(angle))],
            ]
        )

        image = imutils.rotate(np.array(image), angle)

        landmarks = landmarks - 0.5
        new_landmarks = np.matmul(landmarks, transformation_matrix)
        new_landmarks = new_landmarks + 0.5
        return Image.fromarray(image), new_landmarks

    def resize(self, image, landmarks, img_size):
        image = tf.resize(image, img_size)
        return image, landmarks

    def color_jitter(self, image, landmarks):
        color_jitter = transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
        )
        image = color_jitter(image)
        return image, landmarks

    def crop_face(self, image, landmarks, crops):
        left = int(crops["left"])
        top = int(crops["top"])
        width = int(crops["width"])
        height = int(crops["height"])

        image = tf.crop(image, top, left, height, width)

        img_shape = np.array(image).shape
        landmarks = torch.tensor(landmarks) - torch.tensor([[left, top]])
        landmarks = landmarks / torch.tensor([img_shape[1], img_shape[0]])
        return image, landmarks

    def __call__(self, image, landmarks, crops):

        image = Image.fromarray(image)
        image, landmarks = self.crop_face(image, landmarks, crops)
        image, landmarks = self.resize(image, landmarks, (224, 224))
        image, landmarks = self.color_jitter(image, landmarks)
        image, landmarks = self.rotate(image, landmarks, angle=10)

        image = tf.to_tensor(image)
        image = tf.normalize(image, [0.5], [0.5])
        return image, landmarks
