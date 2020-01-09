# DenseNet practice
import os
import sys
import matplotlib.pyplot as plt
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Conv3D, DepthwiseConv2D, SeparableConv2D, Conv3DTranspose
from keras.layers import Flatten, MaxPool2D, AvgPool2D, GlobalAvgPool2D, UpSampling2D, BatchNormalization
from keras.layers import Concatenate, Add, Dropout, ReLU, Lambda, Activation, LeakyReLU, PReLU
from keras.optimizers import Adam, SGD
from keras.callbacks import ReduceLROnPlateau
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score

from time import time
import numpy as np
import datetime
from data import getData, getTrainData, getTestData
from sklearn.model_selection import train_test_split

day = (datetime.datetime.now()+datetime.timedelta(hours=+0)).strftime("%Y-%m-%d %H:%M")
startTime = (datetime.datetime.now()+datetime.timedelta(hours=+0)).strftime("%H:%M")

def densenet(img_shape, n_classes, f=12):
  repetitions = 6, 12, 24, 16
  
  def bn_rl_conv(x, f, k=1, s=1, p='same'):
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(f, k, strides=s, padding=p)(x)
    return x
  
  
  def dense_block(tensor, r):
    for _ in range(r):
      #x = bn_rl_conv(tensor, 4*f)
      x = bn_rl_conv(tensor, f, 3)
      tensor = Concatenate()([tensor, x])
    return tensor
  
  
  def transition_block(x):
    x = bn_rl_conv(x, K.int_shape(x)[-1] // 2)
    x = AvgPool2D(2, strides=2, padding='same')(x)
    return x
  
  
  input = Input(img_shape)
  
  x = Conv2D(64, 7, strides=2, padding='same')(input)
  x = MaxPool2D(3, strides=2, padding='same')(x)
  
  for r in repetitions:
    d = dense_block(x, r)
    x = transition_block(d)
  
  x = GlobalAvgPool2D()(d)
  
  output = Dense(n_classes, activation='softmax')(x)
  
  model = Model(input, output)
  return model

input_shape = 224, 224, 3
n_classes = 4

model = densenet(input_shape, n_classes)
model.summary()
#opt = Adam(lr=0.3)
opt = SGD(lr=0.1)#, decay=1e-4, momentum=0.9, nesterov=True)
#opt = SGD(lr=0.1, momentum=0.9, decay=0.0001, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["categorical_accuracy"])

# Parameter
if(len(sys.argv)<=1):
    EP=200
else:
    EP = int(sys.argv[1])

if(len(sys.argv)<=2):
    BS=20
else:
    BS = int(sys.argv[2])

if(len(sys.argv)<=3):
    LOC = "dataset/original"
else:
    LOC = "dataset/"+sys.argv[3]

print("=========Experiment===========")
print(f"> Location:  {LOC}")
print(f"> Epochs:    {EP}")
print(f"> Batch size:{BS}")
print("==============================")

width = 224
height = 224


x_train, y_train = getTrainData(LOC, width, height)
# Split the data
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle= True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.5, patience=10, mode='auto', cooldown=3, min_lr=0.00001)
startTrain = (datetime.datetime.now()+datetime.timedelta(hours=+0)).strftime("%m-%d %H:%M")
history = model.fit(x_train, y_train, 
                    batch_size=BS, 
                    epochs=EP,
                    #validation_split=0.2,
                    validation_data = (x_val, y_val),
                    callbacks=[reduce_lr],
                    verbose=1)

dataset = LOC.split("/")[1]

model.save(f"model/{dataset}densenet.h5")
endTrain = (datetime.datetime.now()+datetime.timedelta(hours=+0)).strftime("%H:%M")

print("> FINISH TRAINING")
print("> VALIDATION TESTING...")
#x_val, y_val = getTestData(LOC, width, height)
y_pred = model.predict(x_val)
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
label = np.argmax(y_val, axis=1)
acc = accuracy_score(label,pred)
print('Accuracy:  ',acc)
labels = ["black", "mg", "moth", "oil"]
print(classification_report(label, pred, target_names=labels))
print(confusion_matrix(label, pred))

endTest = (datetime.datetime.now()+datetime.timedelta(hours=+0)).strftime("%H:%M")
#trainTime=endTrain-startTime
#testTime=endTest-endTrain
print("----------------------------------\n")
print(f"< Epoch:{EP}, Batch size:{BS} >")
print(f"  Loading image start at {startTime}, end at {startTrain}")
print(f"  Training start at {startTrain}, end at {endTrain}")
print(f"  Testing start at {endTrain}, end at {endTest}")
print("----------------------------------\n")



f = open('record/Training Record.txt', 'a+')
f.write(day)
f.write("\n")
f.write(f"Epochs:     {EP}\nBatch_size: {BS}\nModel:      DenseNet\nAccuracy:   {acc}\n")
f.write(f"Training:   start at {startTime}, end at {endTrain}\nTesting:    start at {endTrain}, end at {endTest}")
f.write("\n======================================================\n")

# Plot learning curve 

# Show figure
acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
#Train and validation accuracy
plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.plot(epochs, val_acc, 'g', label='Validation accurarcy')
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'y', label='Validation loss')
plt.title('Training and Validation')
plt.show()