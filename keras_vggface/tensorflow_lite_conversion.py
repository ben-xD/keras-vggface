from keras_vggface import VGGFace
import tensorflow as tf
from keras_vggface.models import create_vggface_preprocessing_model

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


def get_embeddings_from_png_image_example():
    """Example usage to get face embeddings from cropped image of human face"""
    import numpy as np
    from tensorflow.keras.preprocessing import image

    image_preprocessor = create_vggface_preprocessing_model()
    embeddings_model = VGGFace(model="senet50", pooling="avg", include_top=False, input_shape=(224, 224, 3))

    img = image.load_img('../image/ajb.jpg', target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preprocessed = image_preprocessor.predict(x)
    embeddings = embeddings_model.predict(preprocessed)
    print("embeddings: ", embeddings)


def get_predictions_from_png_image_example():
    """Example usage to get predictions (human identity) from image"""
    from tensorflow.keras.preprocessing import image
    import numpy as np
    import keras_vggface.utils as libutils

    image_preprocessor = create_vggface_preprocessing_model()
    model = VGGFace(model='senet50')
    img = image.load_img('image/ajb-resized.jpg', target_size=(224, 224), interpolation="bilinear")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preprocessed = image_preprocessor.predict(x)
    predictions = model.predict(preprocessed)
    print('Predicted:', libutils.decode_predictions(predictions))
    # # Output of normal:                                [[["b' A._J._Buckley'", 0.91385096], ["b' Guy_Garvey'", 0.009176245], ["b' Jeff_Corwin'", 0.008781389], ["b' Michael_Voltaggio'", 0.0073467665], ["b' Nick_Frost'", 0.0065856054]]]
    # # Output of custom preprocessing:                  [[["b' A._J._Buckley'", 0.91558367], ["b' Guy_Garvey'", 0.009039231], ["b' Jeff_Corwin'", 0.008346532], ["b' Michael_Voltaggio'", 0.0071733994], ["b' Nick_Frost'", 0.006603726]]]
    # # Output of custom preprocessing (separate model): [[["b' A._J._Buckley'", 0.91385096], ["b' Guy_Garvey'", 0.009176245], ["b' Jeff_Corwin'", 0.008781389], ["b' Michael_Voltaggio'", 0.0073467665], ["b' Nick_Frost'", 0.0065856054]]]


if __name__ == "__main__":
    print(
        "Find more descriptive conversion from Keras to TensorFlow lite models in popsa-hq/prototype.face-similarity/mobile")
    image_preprocessor = create_vggface_preprocessing_model()
    create_tflite_model_file(image_preprocessor, 'preprocessing.tflite')

    model = VGGFace(model="senet50", pooling="avg", include_top=False, input_shape=(224, 224, 3))
    create_tflite_model_file(model, 'VggFace2SeNet.tflite')

    print("Remember to add model metadata before using on devices.")
