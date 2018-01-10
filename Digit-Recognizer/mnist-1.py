from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.layers.core import Dense, Activation, Dropout
from keras.callbacks import ModelCheckpoint

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
model.add(Dense(784, init='uniform', activation='relu', input_dim=input_dim))
model.add(Dropout(0.2))
model.add(Dense(784, init='uniform', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, init='uniform', activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='weights.hdf5', verbose=1, save_best_only=True)
print("Training...")
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.3, verbose=1, callbacks=[checkpointer])

print("Generating test predictions...")
preds = model.predict_classes(X_test, verbose=1)

submission=pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds})
submission.to_csv("mnist-predictions.csv", index=False, header=True)
