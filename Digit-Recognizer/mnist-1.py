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
scale = np.max(X_train)
X_train /= scale
X_test /= scale

mean = np.std(X_train)
X_train -= mean
X_test -= mean

input_dim = X_train.shape[1]
num_classes = y_train.shape[1]

# 3-layer network
model = Sequential()
model.add(Dense(784, activation='sigmoid', input_dim=input_dim))
model.add(Dropout(0.2))
model.add(Dense(784, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='sigmoid'))

model.summary()

model.compile(loss='mean_squared_logarithmic_error', optimizer='adam', metrics=['accuracy'])

print("Training...")
model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.3, verbose=1)

print("Generating test predictions...")
preds = model.predict_classes(X_test, verbose=1)

submission=pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds})
submission.to_csv("mnist-predictions.csv", index=False, header=True)
