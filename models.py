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


class Cifar10(chainer.Chain):
    def __init__(self):
        super(Cifar10, self).__init__()
        w = chainer.initializers.HeNormal()
        with self.init_scope():
            self.cv1_1 = L.Convolution2D(None, 192, 5, 1, 1, initialW=w)
            self.bn1_1 = L.BatchNormalization(192)
            self.cv1_2 = L.Convolution2D(None, 160, 1, 1, initialW=w)
            self.bn1_2 = L.BatchNormalization(160)
            self.cv1_3 = L.Convolution2D(None, 96, 1, 1, initialW=w)
            self.bn1_3 = L.BatchNormalization(96)
            self.cv2_1 = L.Convolution2D(None, 96, 5, 1, 1, initialW=w)
            self.bn2_1 = L.BatchNormalization(96)
            self.cv2_2 = L.Convolution2D(None, 192, 1, 1, initialW=w)
            self.bn2_2 = L.BatchNormalization(192)
            self.cv2_3 = L.Convolution2D(None, 192, 1, 1, initialW=w)
            self.bn2_3 = L.BatchNormalization(192)
            self.cv3_1 = L.Convolution2D(None, 192, 3, 1, 1, initialW=w)
            self.bn3_1 = L.BatchNormalization(192)
            self.cv3_2 = L.Convolution2D(None, 192, 1, 1, initialW=w)
            self.bn3_2 = L.BatchNormalization(192)

    def __call__(self, x):
        h = self.bn1_1(F.relu(self.cv1_1(x)))
        h = self.bn1_2(F.relu(self.cv1_2(h)))
        h = self.bn1_3(F.relu(self.cv1_3(h)))
        h = F.max_pooling_2d(h, 3, stride=2, pad=1)
        h = self.bn2_1(F.relu(self.cv2_1(h)))
        h = self.bn2_2(F.relu(self.cv2_2(h)))
        h = self.bn2_3(F.relu(self.cv2_3(h)))
        h = F.max_pooling_2d(h, 3, stride=2, pad=1)
        h = self.bn3_1(F.relu(self.cv3_1(h)))
        h = self.bn3_2(F.relu(self.cv3_2(h)))
        return F.average_pooling_2d(h, 7)
