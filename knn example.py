import sklearn
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing
import pickle

data = pd.read_csv("car.data")

encode = preprocessing.LabelEncoder()
buying = encode.fit_transform(list(data["buying"]))
maint = encode.fit_transform(list(data["maint"]))
door = encode.fit_transform(list(data["door"]))
persons = encode.fit_transform(list(data["persons"]))
lug_boot = encode.fit_transform(list(data["lug_boot"]))
safety = encode.fit_transform(list(data["safety"]))
cls = encode.fit_transform(list(data["class"]))

x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# this is the loop to achieve an higher accuracy for this model
"""high_acc = 0 
for _ in range(2000):

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    model = KNeighborsClassifier(n_neighbors=9)
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    print(acc)
    if acc > high_acc:
        high_acc = acc
        with open("car_model", "wb")as f:
            pickle.dump(model, f)"""


pickle_open = open("student_model.pickle", "rb")
model = pickle.load(pickle_open)
predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]
for x in range(len(predicted)):
    print("predicted: ", names[predicted[x]], "data: ", x_test[x], "actual: ", names[y_test[x]])
    n = model.kneighbors([x_test[x]], 9, True)
    print("N: ", n)
    print("-----------------------------------------------------------------------------------------------------------------------")

