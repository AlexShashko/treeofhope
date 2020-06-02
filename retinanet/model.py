import keras
import keras_resnet.models
from keras.layers import Input

from retinanet.anchors.anchor_parameters import AnchorParameters
from retinanet.anchors.anchors import Anchors
from retinanet.feature_pyramid_network import FPN
from retinanet.layers.bbox_layers import RegressBoxes, ClipBoxes
from retinanet.filter_detection import FilterDetections
from retinanet.initializers.prior_probability import PriorProbability
from retinanet.utils.check_training_model import check_training_model


class RetinaNet():
    """
    """

    def __init__(self,
                 num_classes,
                 input_shape=(800, 800, 3),
                 num_anchors=None,
                 submodels=None,
                 name=None):
        # self.inputs = Input(shape=input_shape)
        self.inputs = Input(shape=input_shape)

        # self.backbone = keras.applications.resnet.ResNet101(include_top=False, weights='imagenet', input_tensor=self.inputs, input_shape=input_shape, classes=num_classes)
        # print(self.backbone.output)
        self.backbone = keras_resnet.models.ResNet50(self.inputs, include_top=False, freeze_bn=True)
        # print(self.backbone.summary())
        # print(self.backbone.layers[-4].outputs)

        if num_anchors is None:
            num_anchors = AnchorParameters.default.num_anchors()

        if submodels is None:
            submodels = self.default_submodels(num_classes, num_anchors)

        # self.backbone_outputs = Dense(3)(self.backbone.outputs[0])
        # print(1)

        # x = self.backbone.output
        # x = GlobalAveragePooling2D()(x)
        # x = Dropout(0.7)(x)
        # self.backbone_outputs = Dense(num_classes, activation= 'softmax')(self.backbone.output)


        C3, C4, C5 = self.backbone.output[1:]


        self.features = FPN(C3, C4, C5).get_layers()

        self.pyramids = self.__build_pyramid(submodels, self.features)

        self.model = keras.models.Model(inputs=self.inputs, outputs=self.pyramids, name=name)
        # ...

    def __build_model_pyramid(self, name, model, features):
        """ Applies a single submodel to each FPN level.
        Args
            name     : Name of the submodel.
            model    : The submodel to evaluate.
            features : The FPN features.
        Returns
            A tensor containing the response from the submodel on the FPN features.
        """
        return keras.layers.Concatenate(axis=1, name=name)([model(f) for f in features])

    def __build_pyramid(self, models, features):
        """ Applies all submodels to each FPN level.
        Args
            models   : List of submodels to run on each pyramid level (by default only regression, classifcation).
            features : The FPN features.
        Returns
            A list of tensors, one for each submodel.
        """
        return [self.__build_model_pyramid(n, m, features) for n, m in models]

    def default_classification_model(
            self,
            num_classes,
            num_anchors,
            pyramid_feature_size=256,
            prior_probability=0.01,
            classification_feature_size=256,
            name='classification_submodel'
    ):
        """ Creates the default classification submodel.
        Args
            num_classes                 : Number of classes to predict a score for at each feature level.
            num_anchors                 : Number of anchors to predict classification scores for at each feature level.
            pyramid_feature_size        : The number of filters to expect from the feature pyramid levels.
            classification_feature_size : The number of filters to use in the layers in the classification submodel.
            name                        : The name of the submodel.
        Returns
            A keras.models.Model that predicts classes for each anchor.
        """
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
        }

        if keras.backend.image_data_format() == 'channels_first':
            inputs = keras.layers.Input(shape=(pyramid_feature_size, None, None))
        else:
            inputs = keras.layers.Input(shape=(None, None, pyramid_feature_size))
        outputs = inputs
        for i in range(4):
            outputs = keras.layers.Conv2D(
                filters=classification_feature_size,
                activation='relu',
                name='pyramid_classification_{}'.format(i),
                kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
                bias_initializer='zeros',
                **options
            )(outputs)

        outputs = keras.layers.Conv2D(
            filters=num_classes * num_anchors,
            kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer=PriorProbability(probability=prior_probability),
            name='pyramid_classification',
            **options
        )(outputs)

        # reshape output and apply sigmoid
        if keras.backend.image_data_format() == 'channels_first':
            outputs = keras.layers.Permute((2, 3, 1), name='pyramid_classification_permute')(outputs)
        outputs = keras.layers.Reshape((-1, num_classes), name='pyramid_classification_reshape')(outputs)
        outputs = keras.layers.Activation('sigmoid', name='pyramid_classification_sigmoid')(outputs)

        return keras.models.Model(inputs=inputs, outputs=outputs, name=name)

    def default_regression_model(self, num_values, num_anchors,
                                 pyramid_feature_size=256,
                                 regression_feature_size=256,
                                 name='regression_submodel'):
        """ Creates the default regression submodel.
        Args
            num_values              : Number of values to regress.
            num_anchors             : Number of anchors to regress for each feature level.
            pyramid_feature_size    : The number of filters to expect from the feature pyramid levels.
            regression_feature_size : The number of filters to use in the layers in the regression submodel.
            name                    : The name of the submodel.
        Returns
            A keras.models.Model that predicts regression values for each anchor.
        """
        # All new conv layers except the final one in the
        # RetinaNet (classification) subnets are initialized
        # with bias b = 0 and a Gaussian weight fill with stddev = 0.01.
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'kernel_initializer': keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
            'bias_initializer': 'zeros'
        }

        if keras.backend.image_data_format() == 'channels_first':
            inputs = keras.layers.Input(shape=(pyramid_feature_size, None, None))
        else:
            inputs = keras.layers.Input(shape=(None, None, pyramid_feature_size))
        outputs = inputs
        for i in range(4):
            outputs = keras.layers.Conv2D(
                filters=regression_feature_size,
                activation='relu',
                name='pyramid_regression_{}'.format(i),
                **options
            )(outputs)

        outputs = keras.layers.Conv2D(num_anchors * num_values, name='pyramid_regression', **options)(outputs)
        if keras.backend.image_data_format() == 'channels_first':
            outputs = keras.layers.Permute((2, 3, 1), name='pyramid_regression_permute')(outputs)
        outputs = keras.layers.Reshape((-1, num_values), name='pyramid_regression_reshape')(outputs)

        return keras.models.Model(inputs=inputs, outputs=outputs, name=name)

    def default_submodels(self, num_classes, num_anchors):
        """ Create a list of default submodels used for object detection.
        The default submodels contains a regression submodel and a classification submodel.
        Args
            num_classes : Number of classes to use.
            num_anchors : Number of base anchors.
        Returns
            A list of tuple, where the first element is the name of the submodel and the second element is the submodel itself.
        """
        return [
            ('regression', self.default_regression_model(4, num_anchors)),
            ('classification', self.default_classification_model(num_classes, num_anchors))
        ]


def __build_anchors(anchor_parameters, features):
    """ Builds anchors for the shape of the features from FPN.
    Args
        anchor_parameters : Parameteres that determine how anchors are generated.
        features          : The FPN features.
    Returns
        A tensor containing the anchors for the FPN features.
        The shape is:
        ```
        (batch_size, num_anchors, 4)
        ```
    """
    anchors = [
        Anchors(
            size=anchor_parameters.sizes[i],
            stride=anchor_parameters.strides[i],
            ratios=anchor_parameters.ratios,
            scales=anchor_parameters.scales,
            name='anchors_{}'.format(i)
        )(f) for i, f in enumerate(features)
    ]
    return keras.layers.Concatenate(axis=1, name='anchors')(anchors)


def add_bboxes_to_model(model,
                        non_max_suppress=True,
                        class_specific_filter=True,
                        name='retinanet-bbox',
                        anchor_params=None,
                        nms_threshold=0.5,
                        score_threshold=0.05,
                        max_detections=300,
                        parallel_iterations=32,
                        ):
    if anchor_params is None:
        anchor_params = AnchorParameters.default

    check_training_model(model)

    features = [model.get_layer(p_name).output for p_name in ['P3', 'P4', 'P5', 'P6', 'P7']]
    anchors = __build_anchors(anchor_params, features)

    regression = model.outputs[0]

    classification = model.outputs[1]

    # print(model.outputs[2])

    boxes = RegressBoxes(name='boxes')([anchors, regression])
    boxes = ClipBoxes(name='clipped_boxes')([model.inputs[0], boxes])

    detections = FilterDetections(
        nms=non_max_suppress,
        class_specific_filter=class_specific_filter,
        name='filtered_detections',
        nms_threshold=nms_threshold,
        score_threshold=score_threshold,
        max_detections=max_detections,
        parallel_iterations=parallel_iterations
    )([boxes, classification] + [])

    # construct the model
    return keras.models.Model(inputs=model.inputs, outputs=detections, name=name)
