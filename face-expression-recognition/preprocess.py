import numpy as np
import pandas as pd

data = pd.read_csv('fer2013.csv')

selected_labels = [0, 2, 3, 4, 6]
images_num = np.zeros(7)

training = []
test = []

for category in data['Usage'].unique():
    if category[0] != 'T':
        continue
    category_data = data[data['Usage'] == category]
    samples = category_data['pixels'].values
    labels = category_data['emotion'].values

    for i in range(len(samples)):
        if labels[i] in selected_labels and images_num[labels[i]] < 3000:
            image = np.fromstring(samples[i], dtype=int, sep=' ').reshape(48, 48)
            training.append((image, labels[i]))
            images_num[labels[i]] += 1
        elif labels[i] in selected_labels and images_num[labels[i]] < 3300:
            image = np.fromstring(samples[i], dtype=int, sep=' ').reshape(48, 48)
            test.append((image, labels[i]))
            images_num[labels[i]] += 1


np.save('data/training.npy', training)
np.save('data/test.npy', test)
