from keras.models import load_model
from data import getData
import numpy as np
import pandas as pd

LOC = "4class"
width = 224
height = 224


model = load_model("cnn.h5")
model.summary()

x_test, y_test, testImgName = getData(LOC, width, height, False)
labels = ["black","mg","moth","oil"]

y_pred = model.predict(x_test)
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
np.set_printoptions(suppress=True)

result = np.hsplit(y_pred, 4)
Black = result[0].flatten()
Mg = result[1].flatten()
Moth = result[2].flatten()
Oil = result[3].flatten()
#print(np.hsplit(y_pred, 4))
#print(np.around(y_pred, decimals=6)[:10])
#print(pred)
trueLabel = np.argmax(y_test, axis=1)

dict = {"Image": testImgName,
        "True": trueLabel,
        "Prediction": pred,
        "Black": Black,
        "Mg": Mg,
        "Moth": Moth,
        "Oil": Oil
}

record = pd.DataFrame(dict)
record.set_index("Image", inplace=True)
record = record.sort_values(by="Image")
print(record)

