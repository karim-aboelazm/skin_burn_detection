import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

CLASSES = ['1st Degree Burn', '2nd Degree Burn', '3rd Degree Burn']
def load_my_model():
    model = tf.keras.models.load_model('my_model.h5')
    return model


def process_image(image_file):
    img = image.load_img(image_file, target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    return x

model = load_my_model()
img = process_image("1st.PNG")
# Get prediction
prediction = model.predict(img)

pred = np.argmax(prediction)
disease = CLASSES[pred]
accuracy = round(prediction[0][pred]*100,2)
print(disease,accuracy)