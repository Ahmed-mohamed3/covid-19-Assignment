#IDS
#20198054 Abdullah Gamal
#20198007 Ahmed Mohamed

import numpy as np
import pandas as pd
import sklearn as sk
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# def binarize():
#     for i in range(1,21):
#         path_positive = (f"Dataset/Positive/P{i}.jpg")
#         path_negative = (f"Dataset/Negative/N{i}.jpg")
#          # to_save_path_positive = ()
#          # to_save_path_negative = ()
#         img_gray_positive = np.array(Image.open(path_positive).convert('L'))
#         img_gray_negative = np.array(Image.open(path_negative).convert('L'))
#         thresh = 120
#         maxval = 255
#         im_bin_positive = (img_gray_positive > thresh) * maxval
#         im_bin_negative = (img_gray_negative > thresh) * maxval
#         im_bin_positive = im_bin_positive.flatten().tolist()
#         im_bin_negative= im_bin_negative.flatten().tolist()
#         positives.append(im_bin_positive)
#         negatives.append(im_bin_negative)
#         positives[i-1].append(True)
#         negatives[i-1].append(False)
#        Image.fromarray(np.uint8(im_bin_positive)).save(f"Dataset/Positive/bin/P{i}_bin.jpg")
#        Image.fromarray(np.uint8(im_bin_negative)).save(f"Dataset/Negative/bin/N{i}_bin.jpg")

# positives = []
# negatives = []
# df_positives = pd.DataFrame(positives)
# df_negatives = pd.DataFrame(negatives)
# dataset = pd.concat([df_positives, df_negatives])
# pd.DataFrame(dataset).to_csv("Dataset/dataset.csv", index=False)

dataset = pd.read_csv("Dataset/dataset.csv")
Y = dataset["label"]
X = dataset.drop(columns=["label"])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

#number_of_hidden_nodes = np.arange(1,10)

model = MLPClassifier(hidden_layer_sizes=(15,), max_iter=1000, alpha=0.001,activation = "identity")
model.fit(X_train, Y_train)
score = model.score(X_test, Y_test)
print(score * 100)



