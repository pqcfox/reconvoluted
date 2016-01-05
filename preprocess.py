#!/usr/bin/env python3
import os
import pickle
import random
import re

import numpy as np
from scipy import misc
from keras.utils import np_utils

IMAGE_SIZE = 48
RELOAD_IMAGES = False

datasets = [
    {
        'name': 'full_dataset',
        'classes': None,
        'items_per_class': None,
        'train_test_same': False,
    },
    {
        'name': 'micro_dataset',
        'classes': 5,
        'items_per_class': 5,
        'train_test_same': True,
    },
]

image_dir = os.path.join('data', 'raw', 'JPEGImages')
split_dir = os.path.join('data', 'raw', 'ImageSplits')
processed_dir = os.path.join('data', 'processed')
datasets_dir = os.path.join('data', 'datasets')


def naive_square(array, target_size):
    """Scales and crops an array to a centered square"""
    shorter = min(array.shape)
    scale = float(target_size / shorter)
    scaled = misc.imresize(array, scale)
    landscape = (array.shape[0] == shorter)
    diff = max(scaled.shape) - min(scaled.shape)
    if diff == 0:
        return scaled
    margin_one = diff // 2
    margin_two = margin_one if diff % 2 == 0 else margin_one + 1
    if landscape:
        return scaled[:, margin_one:-margin_two]
    else:
        return scaled[margin_one:-margin_two, :]


def get_label(name):
    """Retrieves the label from an image filename"""
    label_regex = '(\w+)_\d+.jpg'
    match = re.match(label_regex, name)
    return match.group(1)


def prune_set(split_set, set_classes, items_per_class):
    """Prunes a set of images and labels into a train_set or test_set"""
    pruned_split_set = ([], [])
    pairs = list(zip(*split_set))
    for set_class in set_classes:
        all_class_pairs = [pair for pair in pairs if pair[1] == set_class]
        if items_per_class is None:
            class_pairs = all_class_pairs
        else:
            class_pairs = random.sample(all_class_pairs, items_per_class)
        new_images, new_labels = list(zip(*class_pairs))
        pruned_split_set[0].extend(new_images)
        pruned_split_set[1].extend(new_labels)
    return pruned_split_set


def convert_set(set_values, all_labels):
    """Converts train_set or test_set values into numpy arrays"""
    image_arrays, labels = set_values
    extended_image_arrays = [array[np.newaxis, ...] for array in image_arrays]
    image_array = np.concatenate(extended_image_arrays, axis=0)
    image_array /= 255
    label_indexes = [all_labels.index(label) for label in labels]
    label_array = np.array(label_indexes)
    categorical_array = np_utils.to_categorical(label_array, len(all_labels))
    return (image_array, categorical_array)


if RELOAD_IMAGES:
    image_names = os.listdir(image_dir)
    image_arrays, names = [], []
    print('Loading images...')
    for image_name in image_names:
        input_path = os.path.join(image_dir, image_name)
        output_path = os.path.join(processed_dir, image_name)
        image_array = misc.imread(input_path, flatten=True)
        square_array = naive_square(image_array, IMAGE_SIZE)
        image_arrays.append(square_array)
        names.append(image_name)
        misc.imsave(output_path, square_array)

print('Loading image splits...')
train_set, test_set = ([], []), ([], [])
set_names = {'train.txt': train_set, 'test.txt': test_set}
for split_name, current_set in set_names.items():
    split_path = os.path.join(split_dir, split_name)
    with open(split_path, 'r') as f:
        lines = f.read().splitlines()
    for image_name in lines:
        image_path = os.path.join(processed_dir, image_name)
        image_array = misc.imread(image_path, flatten=True)
        reshaped_array = image_array[np.newaxis, ...]
        current_set[0].append(reshaped_array)
        label = get_label(image_name)
        current_set[1].append(label)

all_classes = list(set(train_set[1] + test_set[1]))
for dataset in datasets:
    print('Forming {}...'.format(dataset['name']))
    if dataset['classes'] is None:
        classes = all_classes
    else:
        classes = random.sample(all_classes, dataset['classes'])
    dataset_train = prune_set(train_set, classes, dataset['items_per_class'])
    if dataset['train_test_same']:
        dataset_test = tuple(list(l) for l in dataset_train)
    else:
        dataset_test = prune_set(test_set, classes, dataset['items_per_class'])
    output_train = convert_set(dataset_train, all_classes)
    output_test = convert_set(dataset_test, all_classes)
    output = (output_train, output_test)
    output_name = '{}.pkl'.format(dataset['name'])
    output_path = os.path.join(datasets_dir, output_name)
    with open(output_path, 'wb') as f:
        pickle.dump(output, f)
