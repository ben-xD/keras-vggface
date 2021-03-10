import numpy as np
from keras_vggface import VGGFace
from keras.preprocessing import image
from keras_vggface import utils
import keras
import unittest
import tensorflow as tf

# # First attempt, to see outputs relatively correct. It works (not fully the same, as the image was downsized to test this.)
# keras.backend.image_data_format()
# model = VGGFace(model='senet50')
# # img = image.load_img('image/ajb.jpg', target_size=(224, 224))
# img = image.load_img('image/ajb-resized.jpg')
# print(img.height)
# print(img.width)
# x = image.img_to_array(img, dtype="uint8")
# x = np.expand_dims(x, axis=0)
# # x = utils.preprocess_input(x, version=2)
# preds = model.predict(x)
# print ('\n', "SENET50")
# print('\n',preds)
# print('\n','Predicted:', utils.decode_predictions(preds))
# # print(utils.decode_predictions(preds)[0][0][0])
# # self.assertIn('A._J._Buckley', utils.decode_predictions(preds)[0][0][0])
# # self.assertAlmostEqual(utils.decode_predictions(preds)[0][0][1], 0.9993529, places=3)


# Conversion
model = VGGFace(model="senet50", pooling="avg", include_top=False)
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
#                                        tf.lite.OpsSet.SELECT_TF_OPS]

tflite_model = converter.convert()
print("Converted model:")
print(tflite_model)
model_filename = 'VggFace2SeNet.tflite'
with open(model_filename, 'wb') as f:
  f.write(tflite_model)