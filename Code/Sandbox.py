from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import layers, models
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np

#[ Data cifar10 ]#@>3
(cifar10_train_images, cifar10_training_labels), (cifar10_validation_images, cifar10_test_labels) = cifar10.load_data()

cifar10_classToLabel = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}

cifar10_training_labels = to_categorical(cifar10_training_labels)
cifar10_test_labels = to_categorical(cifar10_test_labels)#<3

colors = {}
classCount = {}
for class_label in cifar10_classToLabel.values():
    colors[class_label] = {
        "red": 0,
        "green": 0,
        "blue": 0,
    }
    classCount[class_label] = 0

for index, image in enumerate(cifar10_train_images):
    class_label = cifar10_classToLabel[np.argmax(cifar10_training_labels[index])]

    average = image.mean(axis=0).mean(axis=0)

    colors[class_label]['red'] += average[0]
    colors[class_label]['green'] += average[1]
    colors[class_label]['blue'] += average[2]


classCount[class_label] += 1

for class_label in cifar10_classToLabel.values():
    colors[class_label]['red'] /= classCount[class_label]
    colors[class_label]['green'] /= classCount[class_label]
    colors[class_label]['blue'] /= classCount[class_label]


N = len(colors)
ind = np.arange(N)  # the x locations for the groups
width = 0.27       # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)

red = [colors[label]["red"] for label in colors]
rects0 = ax.bar(ind+width*0, red, width, color='red')
green = [colors[label]["green"] for label in colors]
rects1 = ax.bar(ind+width*1, green, width, color='green')
blue = [colors[label]["blue"] for label in colors]
rects2 = ax.bar(ind+width*2, blue, width, color='blue')

ax.set_ylabel('Scores')
ax.set_xticks(ind+width)
ax.set_xticklabels([label for label in cifar10_classToLabel.values()])
ax.legend( [rects0[0], rects1[0], rects2[0]], ['red', 'green', 'blue'])
plt.xticks(rotation=90)

def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
                ha='center', va='bottom')

autolabel(rects0)
autolabel(rects1)
autolabel(rects2)

plt.show()
