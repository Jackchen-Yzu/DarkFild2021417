import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import matplotlib
matplotlib.use("Agg")
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from Res.Model import RES
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
from tensorflow.keras.callbacks import TensorBoard

#Shell操作
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-l", "--label", required=True,
	help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())
#定义超参
EPOCHS = 200
INIT_LR = 1e-2
BS = 40
IMAGE_DIMS = (66, 64, 3)

#数据和标签
data = []
labels = []

imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
 image = cv2.imread(imagePath)
 image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
 image = img_to_array(image)
 data.append(image)
 label = imagePath.split(os.path.sep)[-2]
 labels.append(label)

#像素归一化
data = np.array(data, dtype="float") / 255.0

#二值化标签
labels = np.array(labels)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

#交叉验证
(trainX, testX, trainY, testY) = train_test_split(data,labels,test_size=0.2,random_state=8)

#数据增强
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,height_shift_range=0.1, shear_range=0.2, 
zoom_range=0.2,horizontal_flip=True, fill_mode="nearest")





#模型编译
model=RES.Build(input_shape = (66, 64, 3), classes = 2)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

#tensorboard
logdir = os.path.join("cnn_selu_callbacks")

print(logdir)
if not os.path.exists(logdir):
    os.mkdir(logdir)
output_model_file = os.path.join(logdir,
                                 "fashion_mnist_model.h5")

tensorboard = TensorBoard(log_dir=logdir)

#训练模型

H = model.fit(
	x=aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1,
	callbacks=[tensorboard])

#保存模型
model.save(args["model"], save_format="h5")

#保存标签
f = open(args["label"], "wb")
f.write(pickle.dumps(lb))
f.close()

#画出图像
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])

#python train.py --dataset dataset --model  Resnet.model --label lb.ResNet
#python classify.py --model Resnet.model --labelbin lb.ResNet  --image 2
