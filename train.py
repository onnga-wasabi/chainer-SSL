import argparse
import chainer
import chainer.functions as F
from chainer.datasets import (
    get_cifar10,
    get_mnist,
)
from updater import MetricStandardUpdater
from models import CAE


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, default=-1)
    parser.add_argument('-b', '--batch', type=int, default=300)
    parser.add_argument('-e', '--epoch', type=int, default=100)
    return parser.parse_args()


def main():
    args = parse()
    train, test = get_mnist(ndim=3)
    train, test = get_cifar10()
    model = CAE()
    if args.gpu > -1:
        model.to_gpu()
    train_iter = chainer.iterators.SerialIterator(train, args.batch)
    # test_iter = chainer.iterators.SerialIterator(test, args.batch, repeat=False, shuffle=False)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    updater = MetricStandardUpdater(train_iter, optimizer, device=args.gpu)

    trainer = chainer.training.Trainer(updater, stop_trigger=(args.epoch, 'epoch'))
    # trainer.extend(chainer.training.extensions.Evaluator(test_iter, target=model, device=args.gpu, eval_func=F.mean_squared_error))
    trainer.extend(chainer.training.extensions.LogReport())
    trainer.extend(chainer.training.extensions.PrintReport([
        'epoch', 'main/loss', 'main/metric', 'validation/main/loss'
    ]))
    trainer.extend(chainer.training.extensions.ProgressBar())
    trainer.run()


if __name__ == '__main__':
    main()

