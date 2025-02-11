import torchmetrics.classification
from utils.utils import get_images_labels_list, read_image
from supervised.datasets import ImageClassificationDataset
from supervised.trainer import SupervisedTrainer
from supervised.models import CNNModel

import os
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.model_selection import train_test_split
import torchmetrics
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

SEED = 125
db_name = 'STL10'
train_subset = 'train'
test_subset = 'test'
NUM_CLASSES = 10
limit_per_class = 200
TEST_SIZE = 100 * NUM_CLASSES
IMAGE_SIZE = 96
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
image_paths, labels = get_images_labels_list(db_name, train_subset, limit=limit_per_class)
test_paths, test_labels = get_images_labels_list(db_name, test_subset, limit=100000)
X_train, X_val, y_train, y_val = train_test_split(image_paths, labels, test_size=TEST_SIZE, random_state=SEED)
print(len(X_train), len(X_val))


train_data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(size=IMAGE_SIZE, scale=(0.7, 1.02)),
    transforms.RandomAutocontrast(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomPerspective(),
    transforms.RandomGrayscale(),
    transforms.RandomInvert(),
])

test_data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

for pretrained in [False, True]:
    for train_cap in [100, 50, 10, 5, 1]:
        train_db = ImageClassificationDataset(images=X_train[: train_cap * NUM_CLASSES], labels=y_train[: train_cap * NUM_CLASSES], transforms=train_data_transforms)
        val_db = ImageClassificationDataset(images=X_val, labels=y_val, transforms=test_data_transforms)
        test_db = ImageClassificationDataset(images=test_paths, labels=test_labels, transforms=test_data_transforms)

        train_loader = DataLoader(train_db, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_db, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
        test_loader = DataLoader(test_db, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

        model = CNNModel(num_classes=NUM_CLASSES, hidd_dim=512, pretrained=pretrained).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=1e-6)
        loss_fn = nn.CrossEntropyLoss()

        save_filename = Path(f'checkpoints/{model.__class__.__name__}/checkpoint.pt')
        os.makedirs(save_filename.parent, exist_ok=True)

        metrics = [
            torchmetrics.classification.MulticlassAccuracy(num_classes=NUM_CLASSES),
            torchmetrics.classification.MulticlassAUROC(num_classes=NUM_CLASSES),
            torchmetrics.classification.MulticlassF1Score(num_classes=NUM_CLASSES)
        ]
        trainer = SupervisedTrainer(model, train_loader, val_loader, optim, loss_fn, NUM_CLASSES, save_filename=save_filename, device=device, metrics=metrics)

        history = trainer.train_model(NUM_EPOCHS, early_stop_patience=10, lr_reduce_patience=5, reset_lr_after_training=False)

        trainer.load_state(save_filename)
        with open(f'logs/{Path(__file__).parent.stem}.{model.__class__.__name__}.log', 'a+') as f:
            f.write('='*30)
            f.write(f'\nPretraied: {pretrained}\nTrain Cap: {train_cap} per class\n')
            result = trainer.evaluate(test_loader)
            for key in result.keys():
                f.write(f'{key}: {result[key]:0.4f}\n')
            f.write('='*30)
            f.write('\n\n\n')
