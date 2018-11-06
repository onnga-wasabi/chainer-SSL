import chainer
import chainer.functions as F
from chainer.backends.cuda import cupy
import random


class MetricStandardUpdater(chainer.training.updaters.StandardUpdater):

    def update_core(self):
        batch = self._iterators['main'].next()
        images, labels = self.converter(batch, self.device)
        optimizer = self._optimizers['main']
        model = optimizer.target
        optimizer.update(self.metric, model, images, labels)


class IyatomiMetricStandardUpdater(MetricStandardUpdater):

    def metric(self, model, images, labels):
        xp = cupy.get_array_module(images)
        batchsize = len(images)
        embeddings = model(images)

        embeddings = F.reshape(embeddings, ((batchsize, -1)))
        shape = embeddings.shape
        metric = 0
        for embedding, label in zip(embeddings, labels):
            eculideans = F.sum((embeddings - F.broadcast_to(embedding, (batchsize, shape[1])))**2, axis=1)
            ratios = -F.log_softmax(F.expand_dims(-eculideans, axis=0))[0]
            metric += F.sum(ratios[xp.where(labels == label)])
        chainer.report({'metric': metric}, model)
        return metric


class Iyatomi2MetricStandardUpdater(MetricStandardUpdater):

    def metric(self, model, images, labels):
        batchsize = len(images)
        embeddings = model(images)

        embeddings = F.reshape(embeddings, ((batchsize, -1)))
        shape = embeddings.shape
        metric = 0
        for embedding in embeddings:
            eculideans = F.sum((embeddings - F.broadcast_to(embedding, (batchsize, shape[1])))**2, axis=1)
            ratios = -F.log_softmax(F.expand_dims(-eculideans, axis=0))[0]
            weights = F.softmax(F.expand_dims(-eculideans, axis=0))[0]
            metric += F.sum(ratios * weights)
        chainer.report({'metric': metric}, model)
        return metric


class HofferMetricStandardUpdater(MetricStandardUpdater):

    def __init__(self, refs_points, train_iter, optimizer, device=None):
        super(HofferMetricStandardUpdater, self).__init__(train_iter, optimizer, device=device)
        self.refs_images, self.refs_labels = refs_points

    def get_sample_points(self):
        xp = cupy.get_array_module(self.refs_images)
        labels = self.refs_labels
        refs_images = \
            xp.array([random.choice(self.refs_images[xp.where(labels == label)]) for label in sorted(set(labels))])
        refs_images = chainer.dataset.convert.to_device(self.device, refs_images)
        return refs_images

    def metric(self, model, images, labels):
        batchsize = len(images)
        train_embeds = model(images)
        train_embeds = F.reshape(train_embeds, ((batchsize, -1)))

        sample_points = model(self.get_sample_points())
        sample_points = F.reshape(sample_points, ((len(set(self.refs_labels)), -1)))

        shape = train_embeds.shape

        eculideans = []
        append = eculideans.append
        for embedding in train_embeds:
            append(-F.sum((sample_points - F.broadcast_to(embedding, (len(sample_points), shape[1])))**2, axis=1))
        eculideans = F.reshape(F.concat(eculideans, axis=0), (-1, len(sample_points)))
        metric = F.softmax_cross_entropy(eculideans, labels)
        chainer.report({'metric': metric}, model)
        return metric
