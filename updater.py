import chainer
import chainer.functions as F
from chainer.backends.cuda import cupy


class MetricStandardUpdater(chainer.training.updaters.StandardUpdater):
    def metric(self, model, images, labels):
        xp = cupy.get_array_module(images)
        batchsize = len(images)
        embeddings = model.encode(images)
        # mse = F.mean_squared_error(images, model.decode(embeddings))

        embeddings = F.reshape(embeddings, ((batchsize, -1)))
        shape = embeddings.shape
        metric = 0
        for embedding, label in zip(embeddings, labels):
            eculideans = F.sum((embeddings - F.broadcast_to(embedding, (batchsize, shape[1]))), axis=1)**2
            ratios = -F.log_softmax(F.expand_dims(-eculideans, axis=0))[0]
            metric += F.sum(ratios[xp.where(labels == label)])
        # chainer.report({'mse': mse, 'metric': metric}, model)
        chainer.report({'metric': metric}, model)
        # loss = mse + metric
        return metric

    def update_core(self):
        batch = self._iterators['main'].next()
        images, labels = self.converter(batch, self.device)
        optimizer = self._optimizers['main']
        model = optimizer.target
        optimizer.update(self.metric, model, images, labels)
