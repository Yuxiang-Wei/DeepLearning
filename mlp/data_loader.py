import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib.pyplot as plt


class Fer2013(torch.utils.data.Dataset):
    def __init__(self, csv_file, train, transform=None, target_transform=None):
        super(Fer2013, self).__init__()
        data = pd.read_csv(csv_file)
        # get images and extract features
        images = []
        if train:
            category = 'Training'

        else:
            category = 'PublicTest'
        # get samples and labels of the actual category
        category_data = data[data['Usage'] == category]
        samples = category_data['pixels'].values
        labels = category_data['emotion'].values

        for i in range(len(samples)):
            try:
                image = np.fromstring(samples[i], dtype=float, sep=" ").reshape((48, 48))
                images.append((image, np.int(labels[i])))
            except Exception as e:
                print("error in image: " + str(i) + " - " + str(e))

        self.imgs = images
        plt.imshow(self.imgs[0][0])
        plt.show()
        self.transform = transform
        self.target_transform = target_transform
        print(len(self.imgs))

    def __getitem__(self, index):
        img, label = self.imgs[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode='L')
        label = torch.tensor(label)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        # img = torch.from_numpy(img).float()
        return img, label

    def __len__(self):
        return len(self.imgs)
