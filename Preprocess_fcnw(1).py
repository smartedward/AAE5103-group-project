import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from keras import regularizers
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


#数据预处理函数
def data_preprocess(dataset):
    dataset.drop(["Unnamed: 0","id"],axis=1,inplace=True)
    dataset["Gender"] = dataset["Gender"].map({"Male":1,"Female":0})
    dataset["Customer Type"] = dataset["Customer Type"].map({"Loyal Customer":1,"disloyal Customer":0})
    dataset["Type of Travel"] = dataset["Type of Travel"].map({"Personal Travel": 1, "Business travel": 0})
    dataset["Class"] = dataset["Class"].map({"Eco": 0, "Eco Plus": 1, "Business": 2})
    dataset["satisfaction"] = dataset["satisfaction"].map({"satisfied": 1, "neutral or dissatisfied": 0})

# 保存新csv
def csv_conserve(train,test):
    outputpath1 = 'train_pre.csv'
    outputpath2 = 'test_pre.csv'
    train.to_csv(outputpath1, sep=',', index=False, header=True)
    test.to_csv(outputpath2, sep=',', index=False, header=True)


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

data_preprocess(train)
data_preprocess(test)

#如有少量缺失值则用0填入
train = train.replace(np.nan, 0)
test = test.replace(np.nan, 0)
# print(np.isnan(train).any())
# print(np.isnan(train).any())

# csv_conserve(train,test)

#生成热力图
# plt.figure(figsize = (15,15))
# heatpic = sns.heatmap(train.corr(), annot = True, cmap = "OrRd")
# heatpic.figure.savefig('heatpic.png',dpi=1000)


X=train.drop("satisfaction",axis=1)
y_train=train["satisfaction"]
scale=StandardScaler()
train_s=scale.fit_transform(X)
X_train=pd.DataFrame(train_s,columns=X.columns)

#根据热力图选出选出相关性>=0.25的特征
feature_selection=pd.Series(index=['Cleanliness', 'On-board service',
       'Flight Distance', 'Leg room service', 'Seat comfort',
       'Inflight entertainment',  'Class',
       'Inflight wifi service', 'Online boarding', 'Baggage handling']).index
# print(feature_selection)

X_train=X_train[feature_selection]
# print(X_train.shape)
# print(test.head())

X_test=test.drop("satisfaction",axis=1)
y_test=test["satisfaction"]
test_s=scale.transform(X_test)
X_test=pd.DataFrame(test_s,columns=X_test.columns)
X_test=X_test[feature_selection]
# print(X_test.shape)

X_train=X_train.values
y_train=y_train.values
X_test=X_test.values
y_test=y_test.values

stopping=EarlyStopping(monitor="val_loss",patience=15)
model=Sequential()

model.add(Dense(25,activation="relu",input_shape=(X_train.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(12,activation="relu",kernel_regularizer=regularizers.l1(0.01)))
model.add(Dense(12,activation="relu",kernel_regularizer=regularizers.l1(0.01)))
model.add(Dense(1,activation="relu",kernel_regularizer=regularizers.l1(0.01)))

model.compile(optimizer="adam",loss="binary_crossentropy")
print(model.summary())
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# plot_model(model)

history=model.fit(X_train,y_train,validation_data=(X_test,y_test),callbacks=[stopping],epochs=100,batch_size=50,verbose=1)
pd.DataFrame(history.history).plot()

result_fig= pd.DataFrame(history.history).plot()
result_fig.figure.savefig('result_fig.png',dpi=1000)

predictions= model.predict(X_train)
prediction_classes = [1 if prob > 0.5 else 0 for prob in np.ravel(predictions)]
print("Matrix:\n{}".format(confusion_matrix(y_train,prediction_classes)))
print("Classification:\n{}".format(classification_report(y_train,prediction_classes)))

predictions= model.predict(X_test)
prediction_classes = [1 if prob > 0.5 else 0 for prob in np.ravel(predictions)]
print("Matrix:\n{}".format(confusion_matrix(y_test,prediction_classes)))
print("Classification:\n{}".format(classification_report(y_test,prediction_classes)))