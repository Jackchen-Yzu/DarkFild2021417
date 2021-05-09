from interval import Interval
import cv2
from imutils import paths
import matplotlib.pyplot as plt
import  numpy as np
####################################

#遍历一张HSV格式图片，计数Mask范围内的像素点
def pop_H(image, low, up):
    count = 0
    mask = Interval(low, up)
    b = image[:,:,0]
    a = b.flatten()
    for j in a:
        if j in mask:
            count = count + 1
    return count



#循环图片集，计算每一张图片chount
def all_pop(Paths,low,up):
    list1 = []


    SetPaths = sorted(list(paths.list_images(Paths)))
    for imagePath in SetPaths:
        image = cv2.imread(imagePath)
        image = cv2.GaussianBlur(image, (3, 3), 0)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        list1.append(pop_H(image, low, up)/600)
        list_new =  np.array(list1)

    return  list_new


b = all_pop('E:/Cluster/2', 160, 180)
a = all_pop('E:/Cluster/1',160,180)

listc = []
listx = []


for q in range(len(a)):
    listc.append(a[q]+0.1)


for j in range(0,141):
    listx.append(j)


label = 'Distribution of hue values of gold nanoprobes'
loc='left'
font_dict={'fontsize': 14,\
         'fontweight' : 8.2,\
         'verticalalignment': 'baseline',\
         'horizontalalignment': loc}


plt.figure()
plt.scatter(x = listx  ,y =  listc  ,marker= 'o'  ,c = 'r',label='1')
plt.scatter(x = listx  ,y =  b ,marker= 'o'  ,c = 'y',label='1')


plt.legend(["no-noise","noise"])
plt.ylim(0, 1.5)
plt.xlim(0,150)
plt.title(label,fontdict=font_dict,loc=loc)
plt.xlabel("Num") # x轴名称
plt.ylabel("Hue_Red_ratio") # y 轴名称
plt.show()

