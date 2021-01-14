import coremltools as ct

from keras_vggface import VGGFace
from keras_vggface.strings_model_metadata import VggFaceMetadata
import numpy as np
from tensorflow.keras.preprocessing import image
from keras_vggface import utils

COREML_FILE_FORMAT = ".mlmodel"


def create_core_ml_model_file(keras_model, filename):
    if COREML_FILE_FORMAT not in filename:
        filename += COREML_FILE_FORMAT

    image_input = ct.ImageType(shape=(1, 224, 224, 3,), bias=[91.4953, 103.8827, 131.0912], color_layout="BGR")
    coreml_model = ct.convert(keras_model, inputs=[image_input])
    write_metadata(coreml_model)
    coreml_model.save(filename)


def write_metadata(model):
    print(type(model))
    """When replacing VggFace with another model, why not make a new class to go alongside `VggFaceMetadata`, e.g. `PoseNetMetadata` 
    
    Args:
        model: CoreML model
    """
    model.short_description = VggFaceMetadata.SHORT_DESCRIPTION
    model.author = VggFaceMetadata.AUTHOR
    model.license = VggFaceMetadata.LICENSE
    model.version = VggFaceMetadata.VERSION

    model.input_description["input_1"] = VggFaceMetadata.Layers.IOS_INPUT
    model.output_description["Identity"] = VggFaceMetadata.Layers.OUTPUT


def core_ml_example():
    """Example usage to get face embeddings from cropped image of human face"""
    coreml_model = ct.models.MLModel("Face.mlmodel")
    img = image.load_img('../image/ajb.jpg', target_size=(224, 224))
    output_dictionary = coreml_model.predict({"input_1": img})
    embeddings = output_dictionary["Identity"][0]
    print("CoreML embeddings: ", embeddings)


def create_core_ml_for_tensorflow_preprocessing():
    input = ct.TensorType(shape=(1, 224, 224, 3))
    keras_model = VGGFace(model="senet50", pooling="avg", include_top=False, input_shape=(224, 224, 3))
    coreml_model = ct.convert(keras_model, inputs=[input])
    write_metadata(coreml_model)
    coreml_model.save("Face-without-preprocessing.mlmodel")

def core_ml_with_tensorflow_preprocessing():
    """
    Used to isolate why numbers are significantly different from Android and Python output.
    """
    img = image.load_img('../image/ajb.jpg', target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = utils.preprocess_input(x, version=2)

    coreml_model = ct.models.MLModel("Face-without-preprocessing.mlmodel")
    output_dictionary = coreml_model.predict({"input_1": x})
    embeddings = output_dictionary["Identity"][0]
    print("CoreML embeddings (tensorflow preprocessing): ", embeddings)


def tensorflow_example():
    """This example uses TensorFlow instead of CoreML, and was found to give consistent numbers to CoreML"""
    model = VGGFace(model="senet50", pooling="avg", include_top=False)
    img = image.load_img('../image/ajb.jpg', target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = utils.preprocess_input(x, version=2)
    embeddings = model.predict(x)
    print("TensorFlow embeddings: ", embeddings)


if __name__ == "__main__":
    # This code is used in https://github.com/popsa-hq/prototype.face-similarity/tree/benb/mobilisation/mobile

    # Example, to test the output
    # core_ml_example()
    # tensorflow_example()

    # Removing Preprocessing from CoreML
    # create_core_ml_for_tensorflow_preprocessing()
    core_ml_with_tensorflow_preprocessing()

    # First attempt: Model does not do resizing and preprocessing is done
    # face_model = VGGFace(model="senet50", pooling="avg", include_top=False, input_shape=(224, 224, 3))
    # create_core_ml_model_file(face_model, "Face.mlmodel")

    # Second attempt (Not yet attempted): Add preprocessing to model.
    # embeddings_with_preprocessing_model = create_vggface_with_preprocessing_model()
    # create_core_ml_model_file(embeddings_with_preprocessing_model, "VggFace2SeNetWithPreprocessing.mlmodel")
