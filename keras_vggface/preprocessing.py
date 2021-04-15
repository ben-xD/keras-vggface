import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from tensorflow.keras.models import Model
from tensorflow.python.keras import backend as K, Input
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops import math_ops


class ChannelReversal(Layer):
    """Image color channel reversal layer (e.g. RGB -> BGR)."""

    def __init__(self):
        super(ChannelReversal, self).__init__()

    def call(self, inputs, **kwargs):
        return tf.reverse(inputs, axis=tf.constant([3]), name="channel_reversal")


class DepthwiseNormalization(Layer):
    """Channel specific normalisation (aka. depthwise normalization, because the number
     of channels is the "depth" of the image)"""

    def __init__(self, mean=None, stddev=None):
        super(DepthwiseNormalization, self).__init__()
        if mean is None:
            mean = [0., 0., 0.]
        if stddev is None:
            stddev = [1., 1., 1.]
        self.mean = tf.broadcast_to(mean, [224,224,3])
        self.stddev = tf.broadcast_to(stddev, [224,224,3])   

    def call(self, inputs, **kwargs):
        if inputs.dtype != K.floatx():
            inputs = math_ops.cast(inputs, K.floatx())

        return (inputs - self.mean) / self.stddev


def create_preprocessing_model(output_shape=(224, 224, 3), model_variant="senet50"):
    """Preprocessing model. Use this as the first model to preprocess images before using the original models.
    Alternatively, preprocessing can be done using numpy/ PIL and on Android, Android.graphics.bitmap.createBitmap, but
    they're are not consistent.

    The values used as arguments to [DepthwiseNormalization] were taken from [keras_vggface.utils.preprocess_input],
    specifically, the version 2 parameters. This means this function/ preprocessing model only supports
    RESNET50 and SENET50, since version 1 is used for VGG16.

    Args:
        model_variant: either vgg16, resnet50 or senet50.
        output_shape: The output shape for the processing model, which should match the input
        shape of the subsequent model, formatted with channels-last

    Returns:
        model: Keras model containing layers to preprocess images as per the original keras-vggface/utils.preprocess_inputs
    """
    input_shape = (None, None, 3)
    input = Input(shape=input_shape, batch_size=1, name="input_image")

    x = ChannelReversal()(input)
    x = Resizing(output_shape[0], output_shape[1], interpolation='bilinear', name="Resize")(x)
    if model_variant == "senet50" or model_variant == "resnet50":
        output = DepthwiseNormalization([91.4953, 103.8827, 131.0912])(x)
    elif model_variant == "vgg16":
        output = DepthwiseNormalization([93.5940, 104.7624, 129.1863])(x)
    else:
        raise ValueError(f"Unsupported model_variant: {model_variant}")

    model = Model(input, output, name='preprocessing')
    return model
