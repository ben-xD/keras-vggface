from keras_vggface import VGGFace
import tensorflow as tf
from keras_vggface.models import preprocessing

def create_tflite_model_file(model, filename):
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()
  with open(filename, 'wb') as f:
    f.write(tflite_model)

model = preprocessing()
create_tflite_model_file(model, 'preprocessing.tflite')

model = VGGFace(model="senet50", pooling="avg", include_top=False)
create_tflite_model_file(model, 'VggFace2SeNet.tflite')