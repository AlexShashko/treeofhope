import keras
import numpy as np
import tensorflow

from retinanet.anchors.anchor_parameters import AnchorParameters


def shift(shape, stride, anchors):
    """ Produce shifted anchors based on shape of the map and stride size.
    Args
        shape  : Shape to shift the anchors over.
        stride : Stride to shift the anchors with over the shape.
        anchors: The anchors to apply at each location.
    """
    shift_x = (keras.backend.arange(0, shape[1], dtype=keras.backend.floatx()) + keras.backend.constant(0.5,
                                                                                                        dtype=keras.backend.floatx())) * stride
    shift_y = (keras.backend.arange(0, shape[0], dtype=keras.backend.floatx()) + keras.backend.constant(0.5,
                                                                                                        dtype=keras.backend.floatx())) * stride

    shift_x, shift_y = tensorflow.meshgrid(shift_x, shift_y)
    shift_x = keras.backend.reshape(shift_x, [-1])
    shift_y = keras.backend.reshape(shift_y, [-1])

    shifts = keras.backend.stack([
        shift_x,
        shift_y,
        shift_x,
        shift_y
    ], axis=0)

    shifts = keras.backend.transpose(shifts)
    number_of_anchors = keras.backend.shape(anchors)[0]

    k = keras.backend.shape(shifts)[0]  # number of base points = feat_h * feat_w

    shifted_anchors = keras.backend.reshape(anchors, [1, number_of_anchors, 4]) + keras.backend.cast(
        keras.backend.reshape(shifts, [k, 1, 4]), keras.backend.floatx())
    shifted_anchors = keras.backend.reshape(shifted_anchors, [k * number_of_anchors, 4])

    return shifted_anchors


class Anchors(keras.layers.Layer):
    """ Keras layer for generating achors for a given shape.
    """

    def __init__(self, size, stride, ratios=None, scales=None, *args, **kwargs):
        """ Initializer for an Anchors layer.
        Args
            size: The base size of the anchors to generate.
            stride: The stride of the anchors to generate.
            ratios: The ratios of the anchors to generate (defaults to AnchorParameters.default.ratios).
            scales: The scales of the anchors to generate (defaults to AnchorParameters.default.scales).
        """
        self.size = size
        self.stride = stride
        self.ratios = ratios
        self.scales = scales

        if ratios is None:
            self.ratios = AnchorParameters.default.ratios
        elif isinstance(ratios, list):
            self.ratios = np.array(ratios)
        if scales is None:
            self.scales = AnchorParameters.default.scales
        elif isinstance(scales, list):
            self.scales = np.array(scales)


        self.num_anchors = len(self.ratios) * len(self.scales)
        self.anchors = keras.backend.variable(self.generate_anchors(
            base_size=self.size,
            ratios=self.ratios,
            scales=self.scales,
        ))

        super(Anchors, self).__init__(*args, **kwargs)


    def call(self, inputs, **kwargs):
        features = inputs
        features_shape = keras.backend.shape(features)
        # generate proposals from bbox deltas and shifted anchors
        if keras.backend.image_data_format() == 'channels_first':
            anchors = shift(features_shape[2:4], self.stride, self.anchors)
        else:
            anchors = shift(features_shape[1:3], self.stride, self.anchors)
        anchors = keras.backend.tile(keras.backend.expand_dims(anchors, axis=0), (features_shape[0], 1, 1))

        return anchors

    def generate_anchors(self, base_size=16, ratios=None, scales=None):
        """
        Generate anchor (reference) windows by enumerating aspect ratios X
        scales w.r.t. a reference window.
        """

        if ratios is None:
            ratios = AnchorParameters.default.ratios

        if scales is None:
            scales = AnchorParameters.default.scales

        num_anchors = len(ratios) * len(scales)

        # initialize output anchors
        anchors = np.zeros((num_anchors, 4))

        # scale base_size
        anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

        # compute areas of anchors
        areas = anchors[:, 2] * anchors[:, 3]

        # correct for ratios
        anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
        anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

        # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
        anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
        anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

        return anchors

    def compute_output_shape(self, input_shape):
        if None not in input_shape[1:]:
            if keras.backend.image_data_format() == 'channels_first':
                total = np.prod(input_shape[2:4]) * self.num_anchors
            else:
                total = np.prod(input_shape[1:3]) * self.num_anchors

            return (input_shape[0], total, 4)
        else:
            return (input_shape[0], None, 4)

    def get_config(self):
        config = super(Anchors, self).get_config()
        config.update({
            'size': self.size,
            'stride': self.stride,
            'ratios': self.ratios.tolist(),
            'scales': self.scales.tolist(),
        })

        return config
