from glob import glob
from os.path import splitext
import datetime
import numpy as np
import cv2
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from keras.models import load_model, Sequential
from keras.layers.core import Activation, Flatten, Dropout, Dense
import matplotlib.pyplot as plt
from keras import backend as K
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
import pandas as pd
#5 classes contain healthy
DATASET = "4class"
classes = 4
chooseModel = "CNN"
width = 224
height= 224
testResult = []
if(DATASET=="5class"):
    classes=5
if(DATASET=="6class"):
    classes=6

if(chooseModel == "CNN"):
    width=192
    height=256

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
        testResult.append((name.split("/")[4][:]).split(".")[0])
    return imgList, labelList

#testList, testLabel = readimg("inTest/")
testList, testLabel = readimg("testB/")
#print(testLabel)
label_binarizer = LabelBinarizer()
testLabel = label_binarizer.fit_transform(testLabel)
testList = np.array(testList, dtype=np.float16) / 225.0

if (chooseModel=="CNN"):
    model = load_model(f"./dataset/{DATASET}/model/CNN.h5")
elif (chooseModel=="Alex"):
    model = load_model(f"./dataset/{DATASET}/model/Alex.h5")
elif (chooseModel=="VGG"):
    LR = 1e-3

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
    model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["categorical_accuracy"])
    model.load_weights(f"./dataset/{DATASET}/model/VGG.weight")


#model = load_model(f"./{DATASET}/model/CNN.h5")
model.summary()
print("[Testing]")
scores = model.evaluate(testList, testLabel)
print(f"Test Accuracy: {scores[1]*100}")
test = np.argmax(testLabel, axis=1)
pred = model.predict_classes(testList)

dict = {"Image": testResult,  
        "True": test,
        "Prediciton": pred
       }
record = pd.DataFrame(dict)
record = record.sort_values(by="Image")
print(record)
record.to_csv("Result.csv")
#print(pred)
#print(test)
#print(testResult)
#print(np.vstack((pred, test)))


#print(y_test)
#print(y_pred)
labels = ["black","mg","moth","multi","oil"]
labels4 = ["black","mg","moth","oil"]

if(classes==4):
    labels = labels4
print(classification_report(test, pred, target_names=labels))
print('Accuracy:  ',accuracy_score(test,pred))
#print('Macro acc: ',precision_score(test,pred,average='macro'))
#print('Micro acc: ',precision_score(test,pred,average='micro'))
print(confusion_matrix(test, pred))
