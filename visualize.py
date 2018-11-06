import chainer
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import argparse
from utils import load_row_cifar10
from models import Cifar10
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, default=-1)
    parser.add_argument('-b', '--batch', type=int, default=32)
    return parser.parse_args()


def show_embedding_distribution(embeddings, labels, method='TSNE', file_name='tmp'):

    if isinstance(embeddings, chainer.Variable):
        embeddings = embeddings.data
    elif isinstance(embeddings, list):
        embeddings = np.array(embeddings)

    if len(embeddings.shape) > 2:
        num_of_data = len(embeddings)
        embeddings = embeddings.reshape(num_of_data, -1)

    if method == 'TSNE':
        decompd_embeddings = TSNE(n_components=2).fit_transform(embeddings)
    elif method == 'PCA':
        decompd_embeddings = PCA(n_components=2).fit_transform(embeddings)
    else:
        print('method ha TSNE ka PCA desu.')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for c, label in enumerate(np.unique(labels)):
        idx = [i for i, x in enumerate(labels) if x == label]
        ax.scatter(decompd_embeddings[idx, 0], decompd_embeddings[idx, 1], label=label)
    plt.legend()
    plt.savefig(f'{file_name}.png')


def main():
    args = parse()
    model = Cifar10()
    chainer.serializers.load_npz('hoffer.npz', model)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    _, _, vx, vy = load_row_cifar10()
    images = vx
    labels = vy
    embeddings = []
    for i in range(0, len(images), args.batch):
        batch = chainer.cuda.to_gpu(images[i:i + args.batch])
        with chainer.using_config('train', False):
            [embeddings.append(b) for b in chainer.cuda.to_cpu(model(batch).data)]
    model.to_cpu()
    show_embedding_distribution(embeddings, labels, method='TSNE', file_name='hoffer')


if __name__ == '__main__':
    main()
