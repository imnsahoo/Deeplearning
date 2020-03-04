# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import cv2 as cv

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255


#-----------------------------------------Training Starts-------------------------------------

model = tf.keras.models.Sequential()#Feed foeward 
model.add(tf.keras.layers.Flatten())#input layer
model.add(tf.keras.layers.Dense(128,activation = tf.nn.relu))#hidden
model.add(tf.keras.layers.Dense(128,activation = tf.nn.relu))#hidden
model.add(tf.keras.layers.Dense(10,activation = tf.nn.softmax))#Output classification for 10 digit 

model.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy',metrics= ['accuracy'])
model.fit(x_train,y_train, epochs = 3)

#-----------------------------------------Training Ends-------------------------------------

model.save('num_rec_model1.model')
new_model = tf.keras.models.load_model('C:/Users/NSahoo/.spyder-py3/num_rec_model1.model')

file = "C:/Users/NSahoo/.spyder-py3/1.jpg"
new_model.evaluate(x_test, y_test)


image = cv.imread(file, cv.IMREAD_GRAYSCALE)
image = cv.resize(image, (28, 28))
image = image.astype('float32')
image = image.reshape(1, 28, 28, 1)
image = 255-image
image /= 255


plt.imshow(image.reshape(28, 28),cmap='Greys')
plt.show()
pred = new_model.predict(image.reshape(1, 28, 28, 1), batch_size=1)
print(pred.argmax())