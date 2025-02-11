import torch
import os
from utils.utils import get_images_labels_list, read_image

import sys
import os
import cv2

import matplotlib.pyplot as plt


SEED = 125

db_name = 'STL10'
limit = 10
subset = 'train'

image_paths, labels = get_images_labels_list(db_name, subset, limit=100)


plt.figure(figsize=(32, 32))
for i, (x, y) in enumerate(zip(image_paths[:36], labels[:36])):
    image = read_image(x)
    path = y
    plt.subplot(6, 6, i+1)
    plt.imshow(image)
    plt.axis('off')
    plt.title(y)
plt.show()



# train_image_paths = 
# print(len(train_image_paths))