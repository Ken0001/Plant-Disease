# 2019/11/27 program on server
import argparse
import numpy as np
import pandas as pd
from keras.models import load_model

from data import convert_image_to_array
# Get argument
parser = argparse.ArgumentParser(description="This is a predict program")

parser.add_argument('-d', help='Data location')
parser.add_argument('-m', help='Model')
parser.add_argument('-f', help='testing f')
 
args = parser.parse_args()

if args.d:
    print('Data  :', args.d)
if args.m:
    print('Model :', args.m)
if args.f:
    print('flag f :', args.f)

print("Start predict")
label_list = ["黑點病","健康","缺鎂","潛葉蛾","油胞病"]
width, height = 224, 224
img = []
img.append(convert_image_to_array(args.d, width, height))
img = np.array(img, dtype=np.float16) / 255.0

model = load_model(f"model/{args.m}.h5")
#model.summary()
y_pred = model.predict(img)
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
#np.set_printoptions(suppress=True)

def toPanda(y_pred):
    result = np.hsplit(y_pred, 5)
    Black = result[0].flatten()
    Healthy = result[1].flatten()
    Mg = result[2].flatten()
    Moth = result[3].flatten()
    Oil = result[4].flatten()
    dict = {
        "Image": args.d,
        "Prediction": label_list[pred[0]],
        "Black": Black,
        "Healthy": Healthy,
        "Mg": Mg,
        "Moth": Moth,
        "Oil": Oil
    }
    record = pd.DataFrame(dict)
    record.set_index("Image", inplace=True)
    record = record.sort_values(by="Image")
    np.set_printoptions(suppress=True)
    print(record)

toPanda(y_pred)

print("Done")