from keras.layers import Conv2D, Add, Activation

from retinanet.layers.upsamplelike import UpsampleLike


class FPN():
    """
    """

    def __init__(self, C3, C4, C5, feature_size=256):
        # upsample C5 to get P5 from the FPN paper
        self.P5 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C5_reduced')(C5)
        self.P5_upsampled = UpsampleLike(name='P5_upsampled')([self.P5, C4])
        self.P5 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P5')(self.P5)

        # add P5 elementwise to C4
        self.P4 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
        self.P4 = Add(name='P4_merged')([self.P5_upsampled, self.P4])
        self.P4_upsampled = UpsampleLike(name='P4_upsampled')([self.P4, C3])
        self.P4 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4')(self.P4)
        # add P4 elementwise to C3
        self.P3 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
        self.P3 = Add(name='P3_merged')([self.P4_upsampled, self.P3])
        self.P3 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(self.P3)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6')(C5)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7 = Activation('relu', name='C6_relu')(self.P6)
        self.P7 = Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P7')(self.P7)

    def get_layers(self):
        return [self.P3, self.P4, self.P5, self.P6, self.P7]
