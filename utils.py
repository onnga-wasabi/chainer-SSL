import subprocess
from pathlib import Path
import numpy as np
import pickle
import random

SCRIPTS_DIR = Path(__file__).parent.resolve()


def load_row_cifar10():
    CIFAR10 = SCRIPTS_DIR / "cifar-10-batches-py"
    if not CIFAR10.exists():
        download = f"wget -O {CIFAR10}.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        extract = f"tar -xzvf {CIFAR10}.tar.gz"
        remove = f"rm {CIFAR10}.tar.gz"
        subprocess.call(download, shell=True)
        subprocess.call(extract, shell=True)
        subprocess.call(remove, shell=True)
    train_x = []
    train_y = []
    val_x = []
    val_y = []
    fnames = [f"{CIFAR10}/data_batch_{i}" for i in range(1, 6)]
    for fname in fnames:
        with open(fname, 'rb') as f:
            dict = pickle.load(f, encoding='bytes')
        [train_x.append(datum) for datum in dict[b'data']]
        [train_y.append(label) for label in dict[b'labels']]
    fname = f"{CIFAR10}/test_batch"
    with open(fname, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
        [val_x.append(datum) for datum in dict[b'data']]
        [val_y.append(label) for label in dict[b'labels']]
    train_x = np.array(train_x).astype('f')
    train_x = train_x.reshape(-1, 3, 32, 32)
    train_y = np.array(train_y)
    val_x = np.array(val_x).astype('f')
    val_x = val_x.reshape(-1, 3, 32, 32)
    val_y = np.array(val_y)

    return train_x, train_y, val_x, val_y


def pickup_sample(images, labels, num=10):

    classes = [list(np.where(labels == label)[0]) for label in sorted(set(labels))]
    sample_idxs = [random.sample(classes[i], num) for i in range(len(classes))]
    sample_images = []
    sample_labels = []
    for sample_idx in sample_idxs:
        [sample_images.append(images[idx]) for idx in sample_idx]
        [sample_labels.append(labels[idx]) for idx in sample_idx]

    return np.array(sample_images), np.array(sample_labels)
