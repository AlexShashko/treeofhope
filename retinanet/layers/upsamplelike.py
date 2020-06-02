import keras
import tensorflow

def resize_images(images, size, method='bilinear', align_corners=False):
    """ See https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/image/resize_images .
    Args
        method: The method used for interpolation. One of ('bilinear', 'nearest', 'bicubic', 'area').
    """
    methods = {
        'bilinear': tensorflow.image.ResizeMethod.BILINEAR,
        'nearest' : tensorflow.image.ResizeMethod.NEAREST_NEIGHBOR,
        'bicubic' : tensorflow.image.ResizeMethod.BICUBIC,
        'area'    : tensorflow.image.ResizeMethod.AREA,
    }
    return tensorflow.compat.v1.image.resize_images(images, size, methods[method], align_corners)

class UpsampleLike(keras.layers.Layer):
    """ Keras layer for upsampling a Tensor to be the same shape as another Tensor.
    """

    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = keras.backend.shape(target)
        if keras.backend.image_data_format() == 'channels_first':
            source = tensorflow.transpose(source, (0, 2, 3, 1))
            output = resize_images(source, (target_shape[2], target_shape[3]), method='nearest')
            output = tensorflow.transpose(output, (0, 3, 1, 2))
            return output
        else:
            return resize_images(source, (target_shape[1], target_shape[2]), method='nearest')

    def compute_output_shape(self, input_shape):
        if keras.backend.image_data_format() == 'channels_first':
            return (input_shape[0][0], input_shape[0][1]) + input_shape[1][2:4]
        else:
            return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)
