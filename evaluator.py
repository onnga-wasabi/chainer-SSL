import chainer
import chainer.functions as F
import chainer.reporter as reporter_module
import copy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


class KNNEvaluator(chainer.training.extensions.Evaluator):
    def __init__(self, iterator, target, train_dataset, batch, device=None):
        super(KNNEvaluator, self).__init__(iterator, target, device=device)
        self.batch = batch
        self.X, self.y = self.converter(train_dataset, device)

    def knn(self, X, y):
        batch = len(X)
        X = F.reshape((self._targets['main'](X)), (batch, -1))
        X = chainer.dataset.convert.to_device(-1, X.array)
        y = chainer.dataset.convert.to_device(-1, y)
        pred = self.cls.predict(X)
        accuracy = accuracy_score(pred, y)
        chainer.report({'accuracy': accuracy}, self._targets['main'])

    def evaluate(self):
        iterator = self._iterators['main']
        eval_func = self.knn

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = reporter_module.DictSummary()

        self.cls = KNeighborsClassifier(n_neighbors=3)
        embeddings = []
        append = embeddings.append
        for idx in range(0, len(self.X), self.batch):
            [append(chainer.dataset.convert.to_device(-1, x.array))
             for x in self._targets['main'](self.X[idx:idx + self.batch])]
        embeddings = F.reshape(F.concat(embeddings, axis=0), (len(self.X), -1))
        y = chainer.dataset.convert.to_device(-1, self.y)
        self.cls.fit(embeddings.array, y)

        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                in_arrays = self.converter(batch, self.device)
                with chainer.function.no_backprop_mode():
                    if isinstance(in_arrays, tuple):
                        eval_func(*in_arrays)
                    elif isinstance(in_arrays, dict):
                        eval_func(**in_arrays)
                    else:
                        eval_func(in_arrays)

            summary.add(observation)

        return summary.compute_mean()
