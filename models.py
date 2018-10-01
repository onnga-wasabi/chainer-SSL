import chainer
import chainer.links as L
import chainer.functions as F


class CAE(chainer.Chain):
    def __init__(self):
        super(CAE, self).__init__()
        with self.init_scope():
            self.encoder = Encoder()
            self.decoder = Decoder()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def __call__(self, x):
        return self.decoder(self.encoder(x))


class Encoder(chainer.Chain):
    def __init__(self):
        super(Encoder, self).__init__()
        w = chainer.initializers.HeNormal()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 16, 5, 1, 1, initialW=w)
            self.conv2 = L.Convolution2D(None, 32, 3, 1, 1, initialW=w)
            self.conv3 = L.Convolution2D(None, 64, 3, 1, 1, initialW=w)
            self.conv4 = L.Convolution2D(None, 64, 3, 1, 1, initialW=w)
            self.conv5 = L.Convolution2D(None, 128, 3, 1, 1, initialW=w)

    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2)
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(F.relu(self.conv4(h)), 2)
        h = F.average_pooling_2d(F.relu(self.conv5(h)), 6)
        return h


class Decoder(chainer.Chain):
    def __init__(self):
        super(Decoder, self).__init__()
        w = chainer.initializers.HeNormal()
        with self.init_scope():
            self.deconv1 = L.Deconvolution2D(None, 128, 3, 1, 1, initialW=w)
            self.deconv2 = L.Deconvolution2D(None, 64, 3, 1, 1, initialW=w)
            self.deconv3 = L.Deconvolution2D(None, 32, 3, 1, 1, initialW=w)
            self.deconv4 = L.Deconvolution2D(None, 3, 3, 1, 1, initialW=w)

    def __call__(self, x):
        h = F.unpooling_2d(F.relu(self.deconv1(x)), 2, cover_all=False)
        h = F.unpooling_2d(F.relu(self.deconv2(h)), 2, cover_all=False)
        h = F.unpooling_2d(F.relu(self.deconv3(h)), 2, cover_all=False)
        h = F.unpooling_2d(F.relu(self.deconv4(h)), 2, cover_all=False)
        return h
