
import tensorflow as tf
tf.__version__
mnist = tf.keras.datasets.mnist #28 *28 hand written image from 0-9
(x_train, y_train),(x_test,y_test) = mnist.load_data() # Traing Data and tesing data will be downloaded and stored in array

x_train = tf.keras.utils.normalize(x_train, axis =1)  # normalize this data between 0 to 1 
x_test = tf.keras.utils.normalize(x_test, axis =1)    # normalize this data between 0 to 1 

#-----------------------------------------Training Starts-------------------------------------
model = tf.keras.models.Sequential() #Feed foeward 
model.add(tf.keras.layers.Flatten()) #input layer

model.add(tf.keras.layers.Dense(128,activation = tf.nn.relu))#hidden
model.add(tf.keras.layers.Dense(128,activation = tf.nn.relu))#hidden
model.add(tf.keras.layers.Dense(10,activation = tf.nn.softmax))#Output classification for 10 digit 

opt = tf.keras.optimizers.SGD(learning_rate=0.1)
#tf.keras.optimizers.SGD
model.compile(opt,loss = 'sparse_categorical_crossentropy',metrics= ['accuracy'])
model.fit(x_train,y_train, epochs = 3)
#mean_squared_error
#-----------------------------------------Training Ends-------------------------------------
val_loss, val_acc = model.evaluate(x_test,y_test)
print(val_loss,val_acc )


import matplotlib.pyplot as plt
#print(x_train[1])
plt.imshow(x_train[1], cmap = plt.cm.binary)

model.save('num_rec_model.model')
new_model = tf.keras.models.load_model('num_rec_model.model')
print(new_model)

predictions = new_model.predict(x_test)
print(predictions[0])

import numpy as np 
print(np.argmax(predictions[0]))
plt.imshow(x_test[0],cmap = plt.cm.binary)
plt.show()





