import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from PIL import Image




dataset = pd.read_csv("Dataset/dataset.csv")

Y = dataset["label"]
X = dataset.drop(columns=["label"])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = SVC(kernel="sigmoid", C=2)
model.fit(X_train, Y_train)
score = model.score(X_test, Y_test)
predictions = model.predict(X_test)
print(f"The accuaracy is {score*100}%")
print("The true values are ",Y_test.values)
print("The predicted values are ",predictions)


