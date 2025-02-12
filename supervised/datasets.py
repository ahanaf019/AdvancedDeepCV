from torch.utils.data import Dataset
from utils.utils import read_image

class ImageClassificationDataset(Dataset):
    def __init__(self, images, labels, image_size=(96, 96), transforms=None):
        super().__init__()
        self.images = images
        self.labels = labels
        self.image_size = image_size
        self.transforms = transforms

    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = read_image(self.images[index], self.image_size)
        label = self.labels[index]

        if self.transforms is not None:
            image = self.transforms(image)
        return image, label


class SiameseDataset(Dataset):
    def __init__(self, images, labels, image_size=(96, 96), transforms=None):
        super().__init__()
        self.images = images
        self.labels = labels
        self.image_size = image_size
        self.transforms = transforms
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Incomplete
        pass