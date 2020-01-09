from glob import glob
from os.path import splitext
import cv2
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import numpy as np



def convert_image_to_array(image_dir, width, height):
    print(image_dir)
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            if (image.shape[0]<image.shape[1]):
                image = np.rot90(image)
            image = cv2.resize(image, (width,height)) #size:(w,h)
            #print(image/255)
            return img_to_array(image)
        else :
            print("wtgf")
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None

def readimg(folder, width, height):
    imgList, labelList = [],[]
    print("> Loading "+folder)
    for name in glob(f'{folder}/[a-z]*'):
        imgList.append(convert_image_to_array(name, width, height))
        labelList.append((name.split("_")[0]).split("/")[4])
        #./4class/test/moth_222.JPG
        #print((name.split("_")[0]).split("/"))
    print("> Done")
    return imgList, labelList

def readTestImg(folder, width, height):
    imgList, labelList, imgName, imgLink = [],[],[],[]
    print("> Loading "+folder)
    for name in glob(f'{folder}/[a-z]*'):
        imgList.append(convert_image_to_array(name, width, height))
        labelList.append((name.split("_")[0]).split("/")[3])
        img = name.split("/")[3].split(".")[0]
        loc = name.split("/")[2]
        imgName.append(img)
        imgLink.append("["+img+"](http://134.208.3.54/plant/"+loc+"/"+img+".jpg)")
        #./4class/test/moth_222.JPG
        #print(name)
    print("> Done")
    return imgList, labelList, imgName, imgLink

def getData(loc, width, height, train, bal=False):
    if(train==True):
        print("[Reading Training Data]")
        x_train, y_train = readimg(f"./{loc}/train", width, height)
        x_train = np.array(x_train, dtype=np.float16) / 255.0
        y_train = to_categorical(LabelEncoder().fit_transform(y_train))
        print("[Reading Val Data]")
        x_val, y_val = readimg(f"./{loc}/val", width, height)
        x_val = np.array(x_val, dtype=np.float16) / 255.0
        y_val = to_categorical(LabelEncoder().fit_transform(y_val))

        return x_train, y_train, x_val, y_val
    elif(train==False):
        print("[Reading Testing Data]")
        if(bal == True):
            print("> Balance data")
            x_train, y_train, imgName, imgLink = readTestImg(f"./{loc}/test", width, height)
        else:
            x_train, y_train, imgName, imgLink = readTestImg(f"./{loc}/test", width, height)
        
        x_train = np.array(x_train, dtype=np.float16)# / 255.0
        y_train = to_categorical(LabelEncoder().fit_transform(y_train))

        return x_train, y_train, imgName, imgLink

def getTrainData(loc, width, height):
    x_train, y_train = readimg(f"./{loc}/train", width, height)
    x_train = np.array(x_train, dtype=np.float16) / 255.0
    y_train = to_categorical(LabelEncoder().fit_transform(y_train))

    return x_train, y_train

def getTestData(loc, width, height):
    x_test, y_test = readimg(f"./{loc}/test", width, height)
    x_test = np.array(x_test, dtype=np.float16) / 255.0
    y_test = to_categorical(LabelEncoder().fit_transform(y_test))

    return x_test, y_test

def get3TypeData(loc, width, height, folder):
    x_test, y_test, imgName, imgLink = readTestImg(f"./{loc}/{folder}", width, height)
    x_test = np.array(x_test, dtype=np.float16) / 255.0
    y_test = to_categorical(LabelEncoder().fit_transform(y_test))

    return x_test, y_test, imgName, imgLink