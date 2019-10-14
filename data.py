from glob import glob
from os.path import splitext
import cv2
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import numpy as np



def convert_image_to_array(image_dir, width, height):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            #if (image.shape[0]<image.shape[1]):
            #    image = np.rot90(image)
            image = cv2.resize(image, (width,height)) #size:(w,h)
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None

def readimg(folder, width, height):
    imgList, labelList = [],[]
    print("> Loading "+folder)
    for name in glob(f'{folder}/[a-z]*'):
        imgList.append(convert_image_to_array(name, width, height))
        labelList.append((name.split("_")[0]).split("/")[3])
        #./4class/test/moth_222.JPG
        #print(name)
    print("> Done")
    return imgList, labelList

def readTestImg(folder, width, height):
    imgList, labelList, imgName = [],[],[]
    print("> Loading "+folder)
    for name in glob(f'{folder}/[a-z]*'):
        imgList.append(convert_image_to_array(name, width, height))
        labelList.append((name.split("_")[0]).split("/")[3])
        imgName.append(name.split("/")[3].split(".")[0])
        #./4class/test/moth_222.JPG
        #print(name)
    print("> Done")
    return imgList, labelList, imgName

def getData(loc, width, height, train):
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
        x_train, y_train, imgName = readTestImg(f"./{loc}/test", width, height)
        x_train = np.array(x_train, dtype=np.float16) / 255.0
        y_train = to_categorical(LabelEncoder().fit_transform(y_train))

        return x_train, y_train, imgName