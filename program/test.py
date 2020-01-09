from keras.models import load_model
from data import getData, get3TypeData, getTestData
import numpy as np
import pandas as pd
from keras.utils import plot_model
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score
import sys
### Choose Data Location
if(len(sys.argv)<=1):
    LOC = "dataset/original"
else:
    LOC = "dataset/"+sys.argv[1]
#folder = "part"
width = 224
height = 224

def toPanda(y_pred):
    result = np.hsplit(y_pred, 5)
    Black = result[0].flatten()
    Healthy = result[1].flatten()
    Mg = result[2].flatten()
    Moth = result[3].flatten()
    Oil = result[4].flatten()
    dict = {
        "Image": testImgLink,
        "True": trueLabel,
        "Prediction": pred,
        "Black": Black,
        "Healthy": Healthy,
        "Mg": Mg,
        "Moth": Moth,
        "Oil": Oil
    }
    record = pd.DataFrame(dict)
    record.set_index("Image", inplace=True)
    #record.round({"Black":6,"Mg":6,"Moth":6,"Oil":6})
    record = record.sort_values(by="Image")
    print(record)

    record.to_csv(f"1127Result({folder}).csv", float_format='%.6f')

#model = load_model(f"{LOC}cnn.h5")
dataset = LOC.split("/")[1]
model = load_model(f"model/densenet.h5")
#model.summary()
plot_model(model, to_file='model.png', show_shapes=True)
# getData(LOC, W, H, Train, Balance)
x_test, y_test = getTestData(LOC, width, height)
#x_test, y_test, testImgName, testImgLink = get3TypeData(LOC, width, height, folder)
#labels = ["black", "mg", "moth", "oil"]
#if LOC == "clean":
#    labels = ["black", "healthy", "mg", "moth", "oil"]

print(x_test.shape)

# 4 class: 
labels = ["black", "mg", "moth", "oil"]
# 5 class: 
# labels = ["black", "healthy", "mg", "moth", "oil"]
y_pred = model.predict(x_test)
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
np.set_printoptions(suppress=True)

trueLabel = np.argmax(y_test, axis=1)

print(classification_report(trueLabel, pred, target_names=labels))
print('Accuracy:  ',accuracy_score(trueLabel,pred))
print(confusion_matrix(trueLabel, pred))

#toPanda(y_pred)


"""
result = np.hsplit(y_pred, 4)

Black = result[0].flatten()
Healthy = result[1].flatten()
Mg = result[2].flatten()
Moth = result[3].flatten()
Oil = result[4].flatten()

Black = result[0].flatten()
#Healthy = result[1].flatten()
Mg = result[1].flatten()
Moth = result[2].flatten()
Oil = result[3].flatten()

#print(np.hsplit(y_pred, 4))
#print(np.around(y_pred, decimals=6)[:10])
#print(pred)


dict = {"Image": testImgLink,
        "True": trueLabel,
        "Prediction": pred,
        "Black": Black,
        #"Healthy": Healthy,
        "Mg": Mg,
        "Moth": Moth,
        "Oil": Oil
        #"Link": testImgLink
}

record = pd.DataFrame(dict)
record.set_index("Image", inplace=True)
#record.round({"Black":6,"Mg":6,"Moth":6,"Oil":6})
record = record.sort_values(by="Image")
print(record)

record.to_csv("Result.csv", float_format='%.6f')

"""
