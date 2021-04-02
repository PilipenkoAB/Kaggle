import numpy as np
import pandas as pd

import tensorflow as tf
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

# Add the dataset
train_data = pd.read_csv('train.csv')

#let's pull information to understand what we have 
#train_data.head()
#train_data.info
#train_data.describe

# Devide data to X - to train and Y - labels
train_y = train_data["label"]
train_data.drop(["label"], axis=1, inplace=True)
train_X=train_data


#Check data for missing values
train_data.isnull().any().describe()


# from DataFrame(pandas) to list(int64) then reshape to 28x28 array
train_X = train_X.values.reshape(-1, 28, 28, 1)
# from DataFrame(pandas) to list(int64)
train_y = train_y.values   

## encode data from number values to categorical data
train_y = to_categorical(train_y)

##Scaling:
train_X = train_X/255.00

# Creating a model
model = tf.keras.Sequential([
tf.keras.layers.Conv2D(32, kernel_size = (5,5), padding = 'same', activation ='relu', input_shape = (28,28,1)),
tf.keras.layers.Conv2D(32, kernel_size = (5,5), padding = 'same', activation ='relu'),
tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(512, activation = "relu"),
tf.keras.layers.Dense(10, activation = "softmax")
])

#Set the optimizer 
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

# make variations to avoid overfit
datagen = ImageDataGenerator(rotation_range=5, zoom_range=0.09)

# fit the model
datagen.fit(train_X)
batch =512
model.fit_generator(datagen.flow(train_X, train_y, batch_size=batch), epochs=30)






#----------------
# Make predictions

#train_X.shape

test_X = pd.read_csv('test.csv')
test_X.head()
test_X = test_X.values.reshape(-1,28,28,1)
test_X = test_X / 255.0

# Predictions

predictions = model.predict(test_X)
predictions[354]
pred = np.argmax(predictions, axis=1)

plt.imshow(test_X[354][:,:,0],cmap='gray')
plt.show()

pred[354]

pred_digits = pd.DataFrame({'ImageId': range(1,len(test_X)+1) ,'Label':pred })
pred_digits.to_csv("pre_digits_1.csv",index=False)
pred_digits.head()