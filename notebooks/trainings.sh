#!/bin/sh

python train.py -t /data/models/model0 -n 10 -b keras.applications.mobilenet.MobileNet -o 0.75 -ld 0.001 -la 0.00003 -ed 2 -ea 2 -dd 0.0 -da 0.0 -d ""

python train.py -t /data/models/model1 -n 10 -b keras.applications.mobilenet.MobileNet -o 0.75 -ld 0.001 -la 0.00003 -ed 10 -ea 20 -dd 0.0 -da 0.0 -d ""

python train.py -t /data/models/model2 -n 100 -b keras.applications.mobilenet.MobileNet -o 0.75 -ld 0.001 -la 0.00003 -ed 10 -ea 20 -dd 0.0 -da 0.0 -d ""

python train.py -t /data/models/model3 -n 100 -b keras.applications.mobilenet.MobileNet -o 0.75 -ld 0.001 -la 0.00003 -ed 5 -ea 9 -dd 0.0 -da 0.0 -d ""

python train.py -t /data/models/model4 -n 100 -b keras.applications.mobilenet.MobileNet -o 0.75 -ld 0.001 -la 0.00003 -ed 5 -ea 9 -dd 0.0 -da 0.0 -d ""

python train.py -t /data/models/model5 -n 1000 -b keras.applications.mobilenet.MobileNet -o 0.75 -ld 0.001 -la 0.00003 -ed 5 -ea 9 -dd 0.0 -da 0.000023 -d ""

python train.py -t /data/models/model6 -n 1000 -b keras.applications.mobilenet.MobileNet -o 0.75 -ld 0.001 -la 0.00003 -ed 7 -ea 11 -dd 0.0 -da 0.000023 -d ""

python train.py -t /data/models/model7 -n 1500 -b keras.applications.mobilenet.MobileNet -o 0.75 -ld 0.001 -la 0.00003 -ed 5 -ea 9 -dd 0.0 -da 0.000023 -d ""

python train.py -t /data/models/model8 -n 2500 -b keras.applications.mobilenet.MobileNet -o 0.75 -ld 0.001 -la 0.00003 -ed 5 -ea 9 -dd 0.0 -da 0.000023 -d ""

python train.py -t /data/models/model9 -n 5000 -b keras.applications.mobilenet.MobileNet -o 0.75 -ld 0.001 -la 0.00003 -ed 5 -ea 9 -dd 0.0 -da 0.000023 -d ""

python train.py -t /data/models/model9 -b keras.applications.mobilenet.MobileNet -o 0.75 -ld 0.001 -la 0.00003 -ed 5 -ea 9 -dd 0.0 -da 0.000023 -d ""