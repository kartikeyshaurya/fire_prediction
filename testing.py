from tensorflow import keras
import tensorflow as tf
from keras import layers
from keras.preprocessing.image import load_img
import streamlit as st
from tempfile import NamedTemporaryFile



from keras.preprocessing.image import load_img

import streamlit as st

from tempfile import NamedTemporaryFile

st.set_option('deprecation.showfileUploaderEncoding', False)

buffer = st.file_uploader("Image here pl0x")
temp_file = NamedTemporaryFile(delete=False)

def load_model(image_address):
    model = keras.models.load_model("model/sample_classifier_model/save_at_21.h5")
    image_size = (180,180)
    img = keras.preprocessing.image.load_img(
        c, target_size=image_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    predictions = model.predict(img_array)
    score = predictions[0]
    final_score = str(score)
    return(final_score)


    
if buffer:
    temp_file.write(buffer.getvalue())
    st.write(load_img(temp_file.name))
    c = (temp_file.name)
    v = load_model(temp_file.name)
    st.write("this is kartikey")
    st.write(v)


st.header("If you wanna take a look in the code here is the initial code ")








