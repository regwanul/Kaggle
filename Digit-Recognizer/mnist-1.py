from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.layers.core import Dense, Activation, Dropout

import pandas as pd
import numpy as np

# Read data
train = pd.read_csv('input/train.csv')
labels = train.ix[:,0].values.astype('int32')
X_train = (train.ix[:,1:].values).astype('float32')
X_test = (pd.read_csv('input/test.csv').values).astype('float32')

# One-hot encoding the labels 
y_train = to_categorical(labels) 

# Feature Scaling and Mean Normalization
mean = X_train.mean().astype(np.float32)
std = X_train.std().astype(np.float32)
X_train = (X_train - mean)/std
X_test = (X_test - mean)/std

input_dim = X_train.shape[1]
num_classes = y_train.shape[1]

# 3-layer network
model = Sequential()
model.add(Dense(784, activation='sigmoid', input_dim=input_dim))
model.add(Dense(784, activation='sigmoid'))
model.add(Dense(num_classes, activation='sigmoid'))

model.summary()

model.compile(loss='mean_squared_logarithmic_error', optimizer='adam', metrics=['accuracy'])

print("Training...")
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.3, verbose=2)

print("Generating test predictions...")
preds = model.predict_classes(X_test, verbose=2)

submission=pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds})
submission.to_csv("mnist-predictions.csv", index=False, header=True)
