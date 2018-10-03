import argparse
import chainer
from updater import (
    IyatomiMetricStandardUpdater,
    HofferMetricStandardUpdater,
)
from models import Cifar10
from utils import (
    load_row_cifar10,
    pickup_sample,
)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, default=-1)
    parser.add_argument('-b', '--batch', type=int, default=32)
    parser.add_argument('-e', '--epoch', type=int, default=90)
    parser.add_argument('-u', '--updater', default='hoffer')
    return parser.parse_args()


def main():
    args = parse()
    tx, ty, vx, vy = load_row_cifar10()
    refs_points = pickup_sample(tx, ty)
    model = Cifar10()
    if args.gpu > -1:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
    train = chainer.datasets.TupleDataset(tx, ty)
    train_iter = chainer.iterators.SerialIterator(train, args.batch)
    optimizer = chainer.optimizers.NesterovAG(lr=0.1)
    optimizer.setup(model)
    if args.updater == 'iyatomi':
        updater = IyatomiMetricStandardUpdater(train_iter, optimizer, device=args.gpu)
    elif args.updater == 'hoffer':
        updater = HofferMetricStandardUpdater(refs_points, train_iter, optimizer, device=args.gpu)

    trainer = chainer.training.Trainer(updater, stop_trigger=(args.epoch, 'epoch'))
    trainer.extend(chainer.training.extensions.ExponentialShift('lr', 0.1), trigger=(30, 'epoch'))
    trainer.extend(chainer.training.extensions.LogReport())
    trainer.extend(chainer.training.extensions.PrintReport([
        'epoch',
        'main/metric',
        'validation/main/loss'
    ]))
    trainer.extend(chainer.training.extensions.ProgressBar())
    trainer.run()


if __name__ == '__main__':
    main()
