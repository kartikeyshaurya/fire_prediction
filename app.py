import streamlit as st
import numpy as np
from PIL import Image

from tensorflow import keras
import tensorflow as tf
from keras import layers
from tempfile import NamedTemporaryFile
from keras.preprocessing.image import load_img

   
   


# main title 
st.title('Fire Prediction App ')

# always remeber take only numpy image
image = Image.open("back.jpg")
st.image(image, caption= 'Amazon forest Fire 2019',  use_column_width=True)


#paragraph 1 
st.markdown("Forest Fires cause severe hazard in endangering vegetation , Animal and human life across the world , apart from this it can be a factor for  Airborne hazards ,  Water pollution and other Post- fire risks. Also after Post-fire  is very hard to control and even wildland fire fighters face several life-threatening hazards including heat stress, fatigue, smoke and dust, as well as the risk of other injuries such as burns, cuts and scrapes, animal bites, and even rhabdomyolysis. Between 2000–2016, more than **350** wildland firefighters died on-duty. Only Amazon forest fire cost Brazil US$957 billion to **US$3.5_trillion** over a 30-year period.")
st.markdown("Fast and efficient detection is a key factor in wildfire fighting. Early detection efforts were being made by the different goverenmet from old time, such as Fire lookout towers , Aerial and land photography.")
st.markdown("Computer vision has been quite popular in recent years.It enables computer to understand the content of images and videos.The objective of this project is to develop a model which makes a decision whether it is fire or not.Through this we can automate the process of detecting wildfire")


# HEADER 1 
st.header("Model Architechture")
st.markdown("it will be updated soon")




# MAIN WORKING AREA 
st.header("Model Application and testing ")
st.markdown("Here i have designed two seperate model for classifying the forest fire , the first one is trained on solely on two types of dataset.")
st.markdown("where as the second model has already extrated some knowledge from vgg model developed by **K. Simonyan**  and only last layer is developed by me .")
genre = st.radio(
         "select a model that you want to test on ",
             ('Model from scratch', 'Model from transfer learning'))


st.set_option('deprecation.showfileUploaderEncoding', False)
## loading the image 
buffer = st.file_uploader("Choose the appropriate Image")
temp_file = NamedTemporaryFile(delete=False)


def load_model(image_address):
    model = keras.models.load_model("model/sample_classifier_model/save_at_21.h5")
    image_size = (180,180)
    img = keras.preprocessing.image.load_img(
        image_address, target_size=image_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    predictions = model.predict(img_array)
    score = predictions[0]
    final_score = str(score)
    return(final_score)


if buffer:
    temp_file.write(buffer.getvalue())
    st.write(load_img(temp_file.name))

    v = load_model(temp_file.name)
    if genre == 'Model from scratch':
        st.subheader("the probablility that it is fire is ")
        st.write(v)
    if genre == "Model from transfer learning":
        st.write('b')
    
st.header("If you wanna take a look in the code here is the initial code ")


code = '''
import tensorflow as tf 
from tensorflow import keras 
from keras import layers

import os 
import shutil

# finding out the wrong images

num_skippped = 0
image_size = (180,180)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")

data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1),
    ]
)

plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")

train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    x = data_augmentation(inputs)

    # Entry block
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=image_size + (3,), num_classes=2)
keras.utils.plot_model(model, show_shapes=True)



epochs = 50

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
)

'''
st.code(code, language= 'python')







