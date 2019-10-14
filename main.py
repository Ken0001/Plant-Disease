from cnn import model
from data import getData
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from sklearn.metrics import accuracy_score, classification_report, precision_score, confusion_matrix
import numpy as np
# Parameter
EP = 75
BS = 10
LR = 1e-3
width = 224
height = 224
# Data location
LOC = "4class"
# Get Training and Val Data (train = True or False)
x_train, y_train, x_val, y_val = getData(LOC, width, height, True)

# Reduce learning rate
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.2, patience=10, mode='auto', cooldown=1, min_lr=0.00001)

tbCallBack = TensorBoard(log_dir='./logs',  # log 目录
                histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                #batch_size=32,    # 用多大量的数据计算直方图
                write_graph=True,  # 是否存储网络结构图
                write_grads=True,  # 是否可视化梯度直方图
                write_images=True, # 是否可视化参数
                embeddings_freq=0, 
                embeddings_layer_names=None, 
                embeddings_metadata=None)

history = model.fit(x_train, y_train, 
                    batch_size=BS, 
                    epochs=EP,
                    validation_data=(x_val, y_val),
                    callbacks=[reduce_lr, tbCallBack],
                    verbose=1)

model.save("cnn.h5")

y_pred = model.predict(x_val)
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
test = list()
for i in range(len(y_val)):
    test.append(np.argmax(y_val[i]))


print(classification_report(test, pred))
print('Accuracy:  ',accuracy_score(test,pred))
#print('Macro acc: ',precision_score(test,pred,average='macro'))
print('Micro acc: ',precision_score(test,pred,average='micro'))
print(confusion_matrix(test, pred))