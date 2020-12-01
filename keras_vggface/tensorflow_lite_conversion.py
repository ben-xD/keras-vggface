from keras_vggface import VGGFace
import tensorflow as tf
from models import preprocessing


def create_tflite_model_file(keras_model, filename):
    """Converts keras model into TensorFlow lite model, and
    saves it as `filename.tflite` file in working directory
    # Arguments
      keras_model: a Keras model to be converted
      filename: Filename (with or without `.tflite` extension)
    """
    if ".tflite" not in filename:
        filename += ".tflite"

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()
    with open(filename, 'wb') as f:
        f.write(tflite_model)


def get_embeddings_from_png_image_example():
    """Example usage"""
    import numpy as np
    from tensorflow.keras.preprocessing import image

    preprocessing_model = preprocessing(output_shape=(224, 224, 3))
    model = VGGFace(model="senet50", pooling="avg", include_top=False, input_shape=(224, 224, 3))

    img = image.load_img('../image/ajb.jpg', target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preprocessed = preprocessing_model.predict(x)
    embeddings = model.predict(preprocessed)
    print("embeddings: ", embeddings)


if __name__ == "__main__":
    print(
        "Find more descriptive conversion from Keras to TensorFlow lite models in popsa-hq/prototype.face-similarity/mobile")
    preprocessing_model = preprocessing()
    create_tflite_model_file(preprocessing_model, 'preprocessing.tflite')

    model = VGGFace(model="senet50", pooling="avg", include_top=False, input_shape=(224, 224, 3))
    create_tflite_model_file(model, 'VggFace2SeNet.tflite')

    print("Remember to add model metadata before using on devices.")
