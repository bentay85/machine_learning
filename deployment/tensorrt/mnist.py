import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score

import numpy as np
 
import os
import random
 
#load and normalise train and test sets
(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
 
train_images = train_images / 255.0
test_images = test_images / 255.0
 
# Define the model.
def create_model():
  model = keras.models.Sequential([
      keras.layers.Flatten(input_shape=(28, 28)),
      keras.layers.Dense(32, activation='relu'),
      keras.layers.Dropout(0.2),
      keras.layers.Dense(10, activation='softmax')
  ])
 
  model.compile(
      optimizer='adam',
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'])
  
  return model
 
model = create_model()
 
model.summary()

#Callback function for early stopping
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=1)

history = model.fit(
    train_images,
    train_labels, 
    batch_size=64,
    epochs=10,
    validation_data=(test_images, test_labels),
    callbacks=[callback])

#tf.keras.models.save_model(model,'tf_model/')
model.save('tf_model/')