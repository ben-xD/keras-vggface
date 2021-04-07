from keras_vggface import VGGFace
import tensorflow as tf
from tflite_support import flatbuffers
from tflite_support import metadata as _metadata
from tflite_support import metadata_schema_py_generated as _metadata_fb
from keras_vggface.models import create_preprocessing_model
from keras_vggface.strings_model_metadata import VggFaceMetadata

TFLITE_FILE_FORMAT = ".tflite"


def create_tflite_model_file(keras_model, filename):
    """Converts keras model into TensorFlow lite model, and
    saves it as `filename.tflite` file in working directory
    # Arguments
      keras_model: a Keras model to be converted
      filename: Filename (with or without `.tflite` extension)
    """
    if TFLITE_FILE_FORMAT not in filename:
        filename += TFLITE_FILE_FORMAT

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()
    with open(filename, 'wb') as f:
        f.write(tflite_model)


def write_metadata(model_filename):
    metadata = _metadata_fb.ModelMetadataT()
    metadata.description = VggFaceMetadata.SHORT_DESCRIPTION
    metadata.name = VggFaceMetadata.NAME
    metadata.author = VggFaceMetadata.AUTHOR
    metadata.license = VggFaceMetadata.LICENSE
    metadata.version = VggFaceMetadata.VERSION

    subgraph = _metadata_fb.SubGraphMetadataT()
    subgraph.inputTensorMetadata = [create_input_metadata()]
    subgraph.outputTensorMetadata = [create_output_metadata()]
    metadata.subgraphMetadata = [subgraph]

    save_metadata_to_model_file(metadata, model_filename)


def create_input_metadata():
    input_metadata = _metadata_fb.TensorMetadataT()
    input_metadata.description = VggFaceMetadata.Layers.ANDROID_INPUT
    input_metadata.content = _metadata_fb.ContentT()
    input_metadata.content.contentProperties = _metadata_fb.ImagePropertiesT()
    input_metadata.content.contentProperties.colorSpace = (
        _metadata_fb.ColorSpaceType.RGB)
    input_metadata.content.contentPropertiesType = (
        _metadata_fb.ContentProperties.ImageProperties)
    input_normalization = _metadata_fb.ProcessUnitT()
    input_normalization.optionsType = (
        _metadata_fb.ProcessUnitOptions.NormalizationOptions)
    input_normalization.options = _metadata_fb.NormalizationOptionsT()
    input_normalization.options.mean = [91.4953, 103.8827,
                                        131.0912]  # Normalize here, and reverse the channels in the Android app (when image is loaded)
    input_normalization.options.std = [1, 1, 1]
    input_metadata.processUnits = [input_normalization]
    input_stats = _metadata_fb.StatsT()
    # input_stats.max = [255]
    # input_stats.min = [0]
    input_metadata.stats = input_stats
    return input_metadata


def create_output_metadata():
    output_metadata = _metadata_fb.TensorMetadataT()
    output_metadata.name = "embeddings"
    output_metadata.description = "Embedding vector with 2048 values per face."
    output_metadata.content = _metadata_fb.ContentT()
    output_metadata.content.content_properties = _metadata_fb.FeaturePropertiesT()
    output_metadata.content.contentPropertiesType = (
        _metadata_fb.ContentProperties.FeatureProperties)
    output_stats = _metadata_fb.StatsT()
    # output_stats.max = [1.0]
    # output_stats.min = [0.0]
    output_metadata.stats = output_stats
    return output_metadata


def save_metadata_to_model_file(metadata, model_filename):
    b = flatbuffers.Builder(0)
    b.Finish(metadata.Pack(b),
             _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
    metadata_buffer = b.Output()
    populator = _metadata.MetadataPopulator.with_model_file(model_filename)
    populator.load_metadata_buffer(metadata_buffer)
    # populator.load_associated_files(["your_path_to_label_file"]) # No associated files for this (e.g. No labels files)
    populator.populate()


# Examples
def tensorflow_lite_example():
    """Example usage to get face embeddings from cropped image of human face"""
    import numpy as np
    from tensorflow.keras.preprocessing import image

    img = image.load_img('../image/ajb.jpg', target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    interpreter = tf.lite.Interpreter(model_path="Face.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # An option to compare the 2. They produce the same results.
    USE_TENSORFLOW_PREPROCESSOR = True
    if USE_TENSORFLOW_PREPROCESSOR:
        preprocessed = use_tensorflow_preprocessor(x)
    else:
        preprocessed = use_tensorflow_lite_preprocessor(x)

    interpreter.set_tensor(input_details[0]['index'], preprocessed)
    interpreter.invoke()
    tflite_interpreter_output = interpreter.get_tensor(output_details[0]['index'])
    embeddings = tflite_interpreter_output[0]
    print("TensorFlow Lite embeddings: ", embeddings)


def use_tensorflow_preprocessor(x):
    image_preprocessor = create_preprocessing_model()
    return image_preprocessor.predict(x)


def use_tensorflow_lite_preprocessor(x):
    image_preprocessor_interpreter = tf.lite.Interpreter(model_path="Face-preprocessing.tflite")
    image_preprocessor_interpreter.resize_tensor_input(0, [1, 224, 224, 3])
    image_preprocessor_interpreter.allocate_tensors()
    preprocessor_input_details = image_preprocessor_interpreter.get_input_details()
    preprocessor_output_details = image_preprocessor_interpreter.get_output_details()
    image_preprocessor_interpreter.set_tensor(preprocessor_input_details[0]['index'], x)
    image_preprocessor_interpreter.invoke()
    preprocessor_interpreter_output = image_preprocessor_interpreter.get_tensor(preprocessor_output_details[0]['index'])
    preprocessed = preprocessor_interpreter_output[0]
    preprocessed = np.expand_dims(preprocessed, axis=0)
    return preprocessed


def tensorflow_custom_preprocessing_example():
    """Example usage to get face embeddings from cropped image of human face"""
    import numpy as np
    from tensorflow.keras.preprocessing import image

    image_preprocessor = create_preprocessing_model()
    embeddings_model = VGGFace(model="senet50", pooling="avg", include_top=False, input_shape=(224, 224, 3))

    img = image.load_img('../image/ajb.jpg', target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preprocessed = image_preprocessor.predict(x)
    embeddings = embeddings_model.predict(preprocessed)
    print("TensorFlow embeddings: ", embeddings)


def get_predictions_from_png_image_example():
    """Example usage to get predictions (human identity) from image"""
    from tensorflow.keras.preprocessing import image
    import numpy as np
    import keras_vggface.utils as libutils

    image_preprocessor = create_preprocessing_model()
    model = VGGFace(model='senet50')
    img = image.load_img('image/ajb-resized.jpg', target_size=(224, 224), interpolation="bilinear")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preprocessed = image_preprocessor.predict(x)
    predictions = model.predict(preprocessed)
    print('Predicted:', libutils.decode_predictions(predictions))
    # Output of normal:                                [[["b' A._J._Buckley'", 0.91385096], ["b' Guy_Garvey'", 0.009176245], ["b' Jeff_Corwin'", 0.008781389], ["b' Michael_Voltaggio'", 0.0073467665], ["b' Nick_Frost'", 0.0065856054]]]
    # Output of custom preprocessing (1 model):        [[["b' A._J._Buckley'", 0.91558367], ["b' Guy_Garvey'", 0.009039231], ["b' Jeff_Corwin'", 0.008346532], ["b' Michael_Voltaggio'", 0.0071733994], ["b' Nick_Frost'", 0.006603726]]]
    # (this) output of custom preprocessing (2 model): [[["b' A._J._Buckley'", 0.91385096], ["b' Guy_Garvey'", 0.009176245], ["b' Jeff_Corwin'", 0.008781389], ["b' Michael_Voltaggio'", 0.0073467665], ["b' Nick_Frost'", 0.0065856054]]]


if __name__ == "__main__":
    # This code is used in https://github.com/popsa-hq/prototype.face-similarity/tree/benb/mobilisation/mobile
    tensorflow_lite_example()
    tensorflow_custom_preprocessing_example()

    # # First stage: Image preprocessing model
    # image_preprocessor = create_preprocessing_model()
    # create_tflite_model_file(image_preprocessor, 'FaceEmbeddingsPreprocessing.tflite')
    #
    # # Second stage: Face vector calculation
    # embeddings_model = VGGFace(model="senet50", pooling="avg", include_top=False, input_shape=(224, 224, 3))
    # model_filename = 'FaceEmbeddings.tflite'
    # create_tflite_model_file(embeddings_model, model_filename)
    # write_metadata(model_filename)
