from interval import Interval
import cv2
from imutils import paths
import matplotlib.pyplot as plt
import  numpy as np
#########################
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
def all_pop(Paths,low,up,low1,up1):
    list1 = []
    list2 = []

    SetPaths = sorted(list(paths.list_images(Paths)))
    for imagePath in SetPaths:
        image =cv2.cvtColor(cv2.imread(imagePath),cv2.COLOR_BGR2HSV)
        list1.append(pop_H(image, low, up))
        list2.append(pop_H(image, low1, up1)-200)

        list_new =  np.array(list1)
        list_new_1 = np.array(list2)

    return  list_new,list_new_1


a,b = all_pop('E:/Cluster/1',160,180,100,124)
c,d = all_pop('E:/Cluster/2',160,180,100,124)
e,f = all_pop('E:/Cluster/3',160,180,100,124)

label = 'Distribution of hue values of gold nanoprobes'
loc='left'
font_dict={'fontsize': 14,\
         'fontweight' : 8.2,\
         'verticalalignment': 'baseline',\
         'horizontalalignment': loc}


plt.figure()
plt.scatter(x = a  ,y = b  ,marker= 'o'  ,c = 'r',label='1')
plt.scatter(x = c  ,y = d  ,marker= 'o'  ,c = 'b',label='1')
plt.scatter(x = e  ,y = f  ,marker= 'o'  ,c = 'y',label='1')


plt.legend(["sandwich","singlegold","noise"])
plt.ylim(0, 1000)
plt.xlim(0,500)
plt.title(label,fontdict=font_dict,loc=loc)
plt.xlabel("Hue_Red") # x轴名称
plt.ylabel("Hue_Blue") # y 轴名称
plt.show()

