import os
import cv2
from typing import List
from pathlib import Path



import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def load_state(path: str, model: nn.Module=None, optim: torch.optim.Optimizer=None)-> tuple[nn.Module, torch.optim.Optimizer]:
    """Loads the saved weights to the model and optimizer from the saved path. 

    Args:
        path (str): Full path of the saved state object
        model (nn.Module, optional): Model object to load to. Defaults to None.
        optim (torch.optim.Optimizer, optional): Optimizer to load to. Defaults to None.

    Returns:
        tuple[nn.Module, torch.optim.Optimizer]: The weight-loaded model and optimizer objects
    """
    obj = torch.load(path)
    if model is not None:
        model.load_state_dict(obj['model_state'])
        print('Model State Loaded')
    if optim is not None:
        optim.load_state_dict(obj['optim_state'])
        print('Optimizer State Loaded')
    print(f'Loaded state from Epoch {obj["epoch"]}.')
    return model, optim


def read_image(path: str, size: tuple[int, int]=(96, 96))-> np.ndarray:
    """Returns image as a numpy array from path

    Args:
        path (str): Whole path of image
        size (tuple[int, int], optional): Resize image to `size`. Defaults to (96, 96).

    Returns:
        np.ndarray: Numpy UINT8 Image
    """
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, size)
    return image.astype(np.uint8)



def get_images_labels(db_name:str, subset:str, start_idx_per_class=0, end_idx_per_class=-1, shuffle=True, seed=322)-> tuple[List, List]:
    """Get a list of images and labels. The database should be stored in `~/Datasets/{db_name}`

    Args:
        db_name (str): name of the dataset
        subset (str): Which subset (train, test or other)
        start_idx_per_class (int, optional): The index of the first element of each class to take. Defaults to 0.
        end_idx_per_class (int, optional): The index of the last element of each class to take. Defaults to -1.
        seed (int, optional): sets random state when shuffling the array

    Returns:
        tuple[List, List]: The subset of images and labels from the dataset
    """
    np.random.seed(seed)
    db_path = Path(os.environ['HOME']) / 'Datasets' / db_name

    if not os.path.exists(db_path):
        print('DB not found')
        return None
    print(f'DB found in {db_path}')

    classes = sorted((db_path / subset).glob('*'))
    class_wise_dict = {}

    for _class in classes:
        images = sorted(_class.glob("*"))
        # shuffle the images in each class
        if shuffle:
            images = np.random.choice(images, len(images), replace=False)
        class_wise_dict[_class.stem] = images[start_idx_per_class:end_idx_per_class]

    class_name_list = sorted(list(class_wise_dict.keys()))
    
    images_list = []
    for key in class_wise_dict.keys():
        images_list += class_wise_dict[key].tolist()
    
    # shuffle all the images
    if shuffle:
        images_list = np.random.choice(images_list, len(images_list), replace=False).tolist()
    
    labels = [class_name_list.index(x.parent.stem) for x in images_list]
    print(f'Total Classes: {len(classes)}')
    print(f'Total Images ({end_idx_per_class} per class): {len(images_list)}')
    return images_list, labels



def _plot(arr1: np.ndarray, arr2: np.ndarray, ylabel: str):
    """Helper function to plot train and validation curves.

    Args:
        arr1 (np.ndarray): The training curve
        arr2 (np.ndarray): The validation curve
        ylabel (str): The label on the Y-axis
    """
    plt.plot(arr1, '.-', color='blueviolet')
    plt.plot(arr2, 'x-', color='forestgreen')
    plt.grid('on')
    plt.ylim([0, 1.01])
    plt.xlim([0, len(arr1)])
    plt.xlabel('Epoch', fontdict={'size':16})
    plt.ylabel(f'{ylabel}', fontdict={'size':16})
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.legend(['Train', 'Val'])

def plot_curves(losses, val_losses, accs, val_accs, suptitle=f'Image Size: {-1}', save_path='fig.png'):
    plt.figure(figsize=(12, 6))
    plt.subplot(1,2,1)
    _plot(losses, val_losses, 'Loss')

    plt.subplot(1,2,2)
    _plot(accs, val_accs, 'Accuracy')
    
    plt.suptitle(suptitle)

    plt.savefig(save_path)
    plt.close()


if __name__ == '__main__':
    losses = [0.4371, 0.9556,6, 0.6388, 0.2404, 0.2404, 0.1523, 0.8796, 0.6410, 0.7373,
        0.1185, 0.9729, 0.8492, 0.2911, 0.2636, 0.2651, 0.3738, 0.5723, 0.4888, 0.3621,
        0.6507, 0.2255, 0.3629, 0.4297, 0.5105, 0.8067, 0.2797, 0.5628, 0.6332, 0.1418,
        0.6468, 0.2535, 0.1585, 0.9540, 0.9691, 0.8276, 0.3742, 0.1879, 0.7158, 0.4961,
        0.2098, 0.5457, 0.1309, 0.9184, 0.3329, 0.6963, 0.3805, 0.5681, 0.5920, 0.2664]
    
    val_losses = [0.5310, 1.0107, 0.8467, 0.7178, 0.2600, 0.3248, 0.0700, 0.8188, 0.5500, 0.7023,
        0.0963, 0.9272, 0.9149, 0.2625, 0.2198, 0.2736, 0.3020, 0.6327, 0.4037, 0.4595,
        0.7051, 0.1653, 0.2640, 0.4928, 0.5518, 0.8525, 0.3340, 0.4776, 0.6049, 0.0650,
        0.7194, 0.2781, 0.1247, 0.8667, 0.9313, 0.7926, 0.4201, 0.2154, 0.7933, 0.4906,
        0.1338, 0.5883, 0.1831, 0.9306, 0.3871, 0.6950, 0.3851, 0.5536, 0.4971, 0.1879]
    
    accs = [0.6126, 0.8546, 0.7257, 0.8034, 0.9630, 0.6997, 0.7642, 0.9022, 0.6915, 0.6308,
        0.7159, 0.6645, 0.9719, 0.9232, 0.8534, 0.9486, 0.9215, 0.6746, 0.9570, 0.8157,
        0.9230, 0.9584, 0.7272, 0.6440, 0.6912, 0.7708, 0.9272, 0.9443, 0.6028, 0.8043,
        0.7670, 0.6888, 0.6479, 0.7350, 0.9772, 0.7293, 0.8075, 0.8812, 0.7455, 0.9887,
        0.9850, 0.7007, 0.7989, 0.7204, 0.7139, 0.6148, 0.8438, 0.8011, 0.6206, 0.7115]

    val_accs = [0.6534, 0.8285, 0.6902, 0.8024, 1.0116, 0.6739, 0.7814, 0.9284, 0.6653, 0.6536,
        0.7027, 0.6777, 0.9852, 0.9268, 0.8124, 0.9821, 0.9035, 0.6433, 0.9111, 0.8248,
        0.9407, 0.9101, 0.7284, 0.6167, 0.7057, 0.7383, 0.9463, 0.9330, 0.6465, 0.7681,
        0.7511, 0.6502, 0.6904, 0.7728, 0.9530, 0.7453, 0.8392, 0.8867, 0.7484, 0.9629,
        0.9443, 0.7404, 0.8389, 0.7337, 0.6978, 0.5997, 0.8664, 0.8408, 0.6593, 0.7394]

    plot_curves(losses, val_losses, accs, val_accs, 512)