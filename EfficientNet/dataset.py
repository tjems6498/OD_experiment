import os
import numpy as np
import cv2
import torch
import torch.nn

from torch.utils.data import Dataset
import pdb


class Person(Dataset):
    def __init__(self, data_folder, classes, transform=None):
        super(Person, self).__init__()
        self.datafolder = data_folder
        self.id = os.listdir(data_folder)
        self.classes = classes
        self.transform = transform

    def __len__(self):
        return len(self.id)

    def __getitem__(self, idx):
        # image = np.array(Image.open(os.path.join(self.datafolder, self.id[idx])))
        image = cv2.cvtColor(cv2.imread(os.path.join(self.datafolder, self.id[idx])), cv2.COLOR_BGR2RGB)
        label = self.classes[self.id[idx].split('_')[0]]

        if self.transform is not None:
            image = self.transform(image=image)['image']

        return image, label



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from utils import get_augmentation

    data_folder = 'E:\\Computer Vision\\data\\project\\person'
    classes = {'kid': 0, 'man': 1, 'woman': 2}
    classes_inverse = {v: k for k, v in classes.items()}
    transform = get_augmentation(phase='train')

    dataset = Person(data_folder, classes, transform=transform)

    for i in np.random.randint(0, 4000, 100):
        image, label = dataset[i]
        image = np.array(image.permute(1, 2, 0))

        plt.imshow(image)
        plt.title(classes_inverse[label])
        plt.show()

