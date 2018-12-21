import pandas as pd
import numpy as np
import urllib.request
import os
import os.path

df = pd.read_csv("../data/How-beautiful-animals-DFE.csv")
df = df.append(pd.read_csv("../data/How-beautiful-buildings-DFE.csv"))


print("Downloading files...")


def load_img(row):
    filename = "../data/img/{}.jpg".format(row._unit_id)
    if not os.path.isfile(filename):
        try:
            urllib.request.urlretrieve(row.url, filename)
            print("dowloaded {}.".format(filename))
        except:
            print("warning: could download {}".format(row.url))


df.apply(load_img, axis=1)

print("Downloading files succeeded.")
