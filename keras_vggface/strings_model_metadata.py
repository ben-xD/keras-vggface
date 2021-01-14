"""
Create another class for another model.
"""


class VggFaceMetadata:
    NAME = "VGGFace SeNet50"
    SHORT_DESCRIPTION = "This Face model is a modified form of VGGFace SeNet50 (Taken from the `keras_vggface` " \
                        "python library), available at https://github.com/popsa-hq/keras-vggface. This model takes in a face image " \
                        "and outputs a feature vector representing the face. Similar faces should get similar " \
                        "(but often not identical) feature vectors. Feature vectors from many images can be compared and grouped by unique person."
    AUTHOR = "Q. Cao, L. Shen, W. Xie, O. M. Parkhi, A. Zisserman"
    LICENSE = "For more information, see https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/, " \
              "https://github.com/WeidiXie/Keras-VGGFace2-ResNet50, https://pypi.org/project/keras-vggface/ and " \
              "https://github.com/rcmalli/keras-vggface"
    VERSION = "1.0.0"

    class Layers:
        IOS_INPUT = "Input face (224x224) to be used to calculate feature vector. Apart from resizing the image to fit the model, the image does not need to be preprocessed, as the CoreML performs channel reversal and normalization."
        ANDROID_INPUT = "Input image to be classified. The expected image is {0} x {1}, with "
        "three channels in BGR (blue, green, red) per pixel (NOT RGB). Each value in the "
        "tensor is four bytes (floating point).".format(224, 224)
        OUTPUT = "A vector of 2048 floats, representing the face."


class AnotherModelMetadata:
    NAME = ""
    SHORT_DESCRIPTION = ""
    AUTHOR = ""
    LICENSE = ""
    VERSION = ""

    class Layers:
        INPUT = ""
        OUTPUT = ""