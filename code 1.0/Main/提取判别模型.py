import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import pickle
import cv2
from interval import Interval

'''实现目标：
①在原图中找到符合要求的金银纳米探针夹心结构、金纳米探针和极少数带有红色特征的杂质
②并将其批量导入到训练过的模型中判断
③读取预测结果为bind的图片的坐标信息
④将步骤③中的坐标信息在原图中画圆，并将夹心结构导出到制定文件夹'''


def extract(dark_image_path,low_array,up_array):  # nparray形式 np.array([156, 43,46]) /////   np.array([180, 255, 255])
 #定义常量
    roi_list = [] #储存每个图片的坐标信息
    singleimg_list = []

#主体部分
    kernel = np.ones((5, 5), np.uint8)
    image = cv2.imread(dark_image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv, low_array, up_array)
    red = cv2.bitwise_and(image,image, mask=red_mask)
    im_in = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)
    ret, im_th = cv2.threshold(im_in, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    im_out = cv2.dilate(im_th, kernel, iterations=5)
    a, b, c, d = cv2.connectedComponentsWithStats(im_out, connectivity=8, ltype=cv2.CV_32S)
    for t in range(1, a, 1):
        #输出c[(1,2,3,4),(2,3,4,5).......]
        #C包含坐标信息，a是联通组件的编号，由于connectedComponentsWithStats统计里全图的联通组件，所以需要循环遍历
        x, y, w, h, area = c[t]
        roi_list.append((x, y, w, h))
    for i in range(len(roi_list)):
        x = roi_list[i][0]
        y = roi_list[i][1]
        w = roi_list[i][2]
        h = roi_list[i][3]
        a_cor = (x, y, w)
        b_cor = (h, 0, 0)
        roi = image[y+1:y+h, x+1:x+w]
        roi = roi.astype(np.float32)
        #HSV且64*64后的矩阵
        roi_mid = np.insert(cv2.resize(roi,(20,20)),0,a_cor,axis=0)
        roi_hsv_new = np.insert(roi_mid,0,b_cor,axis=0)  # 带有坐标的每个小图
        singleimg_list.append(roi_hsv_new)
  #输出结果是64*64带坐标的矩阵
    return  singleimg_list

model = load_model('E:/cmy/CNN/pokedex.model')
lb = pickle.loads(open('E:/cmy/CNN/LB.pickle', "rb").read())

def pop_H(image, low, up):
    count = 0
    mask = Interval(low, up)
    b = image[:,:,0]
    a = b.flatten()
    for j in a:
        if j in mask:
            count = count + 1
    return count

##模型判别
def load_modellabel(singleimg_list):
    count = 0
    list888 = []
    for k in range(len(singleimg_list)):
        input_img = np.delete(singleimg_list[k],[0,1],axis=0)
        image1 = cv2.resize(input_img, (20, 20))
        image = image1.astype("float") / 255.0
        image2 = img_to_array(image)
        image3 = np.expand_dims(image2, axis=0)
        proba = model.predict(image3)[0]  # (输入测试数据,输出预测结果)
        idx = np.argmax(proba)  ##返回一个numpy数组中最大值的索引值
        label = lb.classes_[idx]  # 分类结果label标签

        if label == 'bind' and pop_H(image,160,180) > 50:
            count = count + 1

            x1 = singleimg_list[k][1][0][0]
            y1 = singleimg_list[k][1][0][1]
            w1 = singleimg_list[k][1][0][2]
            h1 = singleimg_list[k][0][0][0]
            list888.append((x1,y1,w1,h1))

    return  list888

ppp = extract('E:/4.tif',np.array([156, 43,46]),np.array([180, 255, 255]))
pppp = load_modellabel(ppp)
img = cv2.imread('E:/4.tif')


############################  count
list_cor = []
for m in range(len(pppp)):
    x = pppp[m][0]
    y = pppp[m][1]
    w = pppp[m][2]
    h = pppp[m][3]
    q = cv2.rectangle(img,(x,y),(x+w, y+h),(0, 255, 0), 1, 8, 0)
    list_cor.append(x)
    print(len(list_cor))
    cv2.imwrite('lll.png',q,[int(cv2.IMWRITE_JPEG_QUALITY), 95])








