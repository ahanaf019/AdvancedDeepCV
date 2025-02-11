from utils.utils import get_images_labels_list, read_image
from supervised.datasets import ImageClassificationDataset

import matplotlib.pyplot as plt


SEED = 125

db_name = 'STL10'
limit = 100
subset = 'train'

image_paths, labels = get_images_labels_list(db_name, subset, limit=limit)


# plt.figure(figsize=(32, 32))
# for i, (x, y) in enumerate(zip(image_paths[:36], labels[:36])):
#     image = read_image(x)
#     path = y
#     plt.subplot(6, 6, i+1)
#     plt.imshow(image)
#     plt.axis('off')
#     plt.title(y)
# plt.show()

train_db = ImageClassificationDataset(images=image_paths, labels=labels, transforms=None)

x, y = next(iter(train_db))
plt.imshow(x)
plt.title(y)
plt.show()



# train_image_paths = 
# print(len(train_image_paths))