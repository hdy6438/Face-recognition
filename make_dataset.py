import os

import numpy as np

dataset = []
label = []
for data in os.listdir("data/faces"):
    dataset.append(np.load("data/faces/{}".format(data)))
    label.append(data.split(".")[0])


np.save("data/dataset",np.array(dataset))
np.save("data/label",np.array(label))

