from glob import glob
from os.path import splitext
import numpy as np
import cv2
import datetime
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, confusion_matrix
from keras.applications.vgg16 import VGG16
from keras import optimizers

### 限制gpu使用率30%
import tensorflow as tf

# 只使用 30% 的 GPU 記憶體
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# 設定 Keras 使用的 TensorFlow Session
tf.keras.backend.set_session(sess)
###

# 5 classes contain healthy
DATASET = "4class"
classes = 4
chooseModel = "CNN"
width = 224
height= 224

if(DATASET=="5class"):
    classes=5
if(DATASET=="6class"):
    classes=6

if(chooseModel == "CNN"):
    width=192
    height=256

LR = 1e-3
EPOCHS = 100
BS = 16

startTime = (datetime.datetime.now()+datetime.timedelta(hours=+0)).strftime("%m-%d %H:%M")

def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            if (image.shape[0]<image.shape[1]):
                image = np.rot90(image)
            image = cv2.resize(image, (width,height)) #size:(w,h)
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None

def readimg(folder):
    imgList, labelList = [],[]
    print("> Loading "+folder)
    for name in glob(f'./dataset/{DATASET}/'+folder+'[a-z]*'):
        imgList.append(convert_image_to_array(name))
        labelList.append(name.split("_")[0][10:])
    return imgList, labelList

List, Label = readimg("trainB/")
LE = LabelEncoder()
trueLabel = LE.fit_transform(Label)
x_train = np.array(List, dtype=np.float16) / 255.0
y_train = to_categorical(trueLabel)

List, Label = readimg("valB/")
LE = LabelEncoder()
trueLabel = LE.fit_transform(Label)
x_val = np.array(List, dtype=np.float16) / 255.0
y_val = to_categorical(trueLabel)

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

def buildModel(inputShape, chanDim, lr, epochs, classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(classes))
    model.add(Activation("softmax"))
    #model.summary()
    opt = Adam(lr=lr)
    model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
    return model

def buildVGG(classes):
    vgg16_model = VGG16(weights="imagenet")
    vgg16_model.layers.pop()
    model = Sequential()
    for layer in vgg16_model.layers:
        model.add(layer)
    #model.layers.pop()
    for layer in model.layers:
        layer.trainable = False
    model.add(Dense(classes))
    model.add(Activation("softmax"))
    opt = Adam(lr=LR)
    model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
    return model

def buildMyVGG(classes):
    vgg16_model = VGG16()
    vgg16_model.layers.pop()
    model = Sequential()
    for layer in vgg16_model.layers:
        model.add(layer)
    for layer in model.layers:
        layer.trainable = False
    model.add(Dense(classes))
    model.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    return model

def buildAlex(classes):
    # Initializing the CNN
    classifier = Sequential()

    # Convolution Step 1
    classifier.add(Conv2D(96, 11, strides = (4, 4), padding = 'valid', input_shape=(224, 224, 3), activation = 'relu'))

    # Max Pooling Step 1
    classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))
    classifier.add(BatchNormalization())

    # Convolution Step 2
    classifier.add(Conv2D(256, 11, strides = (1, 1), padding='valid', activation = 'relu'))

    # Max Pooling Step 2
    classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding='valid'))
    classifier.add(BatchNormalization())

    # Convolution Step 3
    classifier.add(Conv2D(384, 3, strides = (1, 1), padding='valid', activation = 'relu'))
    classifier.add(BatchNormalization())

    # Convolution Step 4
    classifier.add(Conv2D(384, 3, strides = (1, 1), padding='valid', activation = 'relu'))
    classifier.add(BatchNormalization())

    # Convolution Step 5
    classifier.add(Conv2D(256, 3, strides=(1,1), padding='valid', activation = 'relu'))

    # Max Pooling Step 3
    classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))
    classifier.add(BatchNormalization())

    # Flattening Step
    classifier.add(Flatten())

    # Full Connection Step
    classifier.add(Dense(units = 4096, activation = 'relu'))
    classifier.add(Dropout(0.4))
    classifier.add(BatchNormalization())
    classifier.add(Dense(units = 4096, activation = 'relu'))
    classifier.add(Dropout(0.4))
    classifier.add(BatchNormalization())
    classifier.add(Dense(units = 1000, activation = 'relu'))
    classifier.add(Dropout(0.2))
    classifier.add(BatchNormalization())
    classifier.add(Dense(units = classes, activation = 'softmax'))
    classifier.summary()


    # Compiling the CNN
    classifier.compile(optimizer=optimizers.Adam(lr=0.001),
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
    
    return classifier



depth=3

inputShape = (height, width, depth)
chanDim = -1
if K.image_data_format() == "channels_first":
    inputShape = (depth, height, width)
    chanDim = 1

aug = ImageDataGenerator(
    rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, 
    zoom_range=0.2,horizontal_flip=True, 
    fill_mode="nearest")


if(chooseModel=="CNN"):
    model = buildModel(inputShape, chanDim, LR, EPOCHS, classes)
elif(chooseModel=="VGG"):
    model = buildVGG(classes)
elif(chooseModel=="Alex"):
    model = buildAlex(classes)

#model = buildVGG(classes)
#x_train, x_test, y_train, y_test = train_test_split(List, Label, test_size=0.2, random_state = 42)
#model.summary()
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.2, patience=5, mode='auto', cooldown=1, min_lr=1e-6)
#history = model.fit(x_train, y_train, batch_size=BS, epochs=EPOCHS, verbose=1, validation_data=(x_val, y_val))#, callbacks=[reduce_lr])

history = model.fit_generator(
    aug.flow(x_train, y_train, batch_size=BS),
    validation_data=(x_val, y_val),
    steps_per_epoch=np.ceil(len(x_train)/BS)*3,
    epochs=EPOCHS, verbose=1, callbacks=[reduce_lr]
    )

#from sklearn.metrics import accuracy_score
y_pred = model.predict(x_val)
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
test = list()
for i in range(len(y_val)):
    test.append(np.argmax(y_val[i]))


print(classification_report(test, pred))
score = accuracy_score(test,pred)
print('Accuracy:  ', score)
print('Micro acc: ',precision_score(test,pred,average='micro'))
print(confusion_matrix(test, pred))

endTime = (datetime.datetime.now()+datetime.timedelta(hours=+0)).strftime("%m-%d %H:%M")

print(f"Start at {startTime}, end at {endTime}")
if(chooseModel=="CNN"):
    model.save(f"./dataset/{DATASET}/model/CNN.h5")
elif (chooseModel=="VGG"):
    model.save_weights(f"./dataset/{DATASET}/model/VGG.weight")
elif (chooseModel=="Alex"):
    model.save(f"./dataset/{DATASET}/model/Alex.h5")

import seaborn as sns
sns.set()

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

#accuracy plot
plt.plot(epochs, acc, color='green', label='Training Accuracy')
plt.plot(epochs, val_acc, color='blue', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

#loss plot
plt.plot(epochs, loss, color='pink', label='Training Loss')
plt.plot(epochs, val_loss, color='red', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()