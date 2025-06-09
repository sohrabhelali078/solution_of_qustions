import csv
import numpy as np
from typing import Set, Tuple, List
import torch
import torch.utils
import torch.utils.data
import torch.nn as nn
import torchvision
NoneType = type(None)
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.models import vgg11
from torchvision.models import mobilenet_v2
import torchvision.transforms as transforms
import time

# Exercise 1
def id_to_fruit(fruit_id: int, fruits: Set[str]) -> str:
    """
    This method returns the fruit name by getting the string at a specific index of the set.

    :param fruit_id: The id of the fruit to get
    :param fruits: The set of fruits to choose the id from
    :return: The string corresponding to the index ``fruit_id``
    """
    sorted_fruits = sorted(fruits)
    if fruit_id < 0 or fruit_id >= len(sorted_fruits):
        raise RuntimeError(f"Fruit with id {fruit_id} does not exist")
    return sorted_fruits[fruit_id]

name1 = id_to_fruit(1, {"apple", "orange", "melon", "kiwi", "strawberry"})
name3 = id_to_fruit(3, {"apple", "orange", "melon", "kiwi", "strawberry"})
name4 = id_to_fruit(4, {"apple", "orange", "melon", "kiwi", "strawberry"})


# Exercise 2
def swap(coords: np.ndarray):
    """
    This method will flip the x and y coordinates in the coords array.

    :param coords: A numpy array of bounding box coordinates with shape [n,5]
    :return: The new numpy array where the x and y coordinates are flipped.
    """
    swapped = coords.copy()
    swapped[:, 0], swapped[:, 1] = coords[:, 1], coords[:, 0]
    swapped[:, 2], swapped[:, 3] = coords[:, 3], coords[:, 2]
    return swapped

coords = np.array([[10, 5, 15, 6, 0],
                   [11, 3, 13, 6, 0],
                   [5, 3, 13, 6, 1],
                   [4, 4, 13, 6, 1],
                   [6, 5, 13, 16, 1]])
swapped_coords = swap(coords)


# Exercise 3
def plot_data(csv_file_path: str):
    """
    This code plots the precision-recall curve based on data from a .csv file,
    where precision is on the x-axis and recall is on the y-axis.
    """
    results = []
    with open(csv_file_path) as result_csv:
        csv_reader = csv.reader(result_csv, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            results.append([float(row[0]), float(row[1])])
        results = np.stack(results)

    plt.plot(results[:, 0], results[:, 1])  # precision on x-axis, recall on y-axis
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.show()

# Generate test CSV for plotting
with open("data_file.csv", "w", newline='') as f:
    w = csv.writer(f)
    w.writerow(["precision", "recall"])
    w.writerows([
        [0.013, 0.951],
        [0.376, 0.851],
        [0.441, 0.839],
        [0.570, 0.758],
        [0.635, 0.674],
        [0.721, 0.604],
        [0.837, 0.531],
        [0.860, 0.453],
        [0.962, 0.348],
        [0.982, 0.273],
        [1.0, 0.0]
    ])

plot_data('data_file.csv')
