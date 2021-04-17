import cv2
import numpy as np
from interval import Interval
from imutils import paths
#定义填充函数
#########################################################################
imagePaths = sorted(list(paths.list_images('E:/cmy/1')))
list1 = []
kernel = np.ones((5, 5), np.uint8)

for imagePath in imagePaths:
 img = cv2.imread(imagePath)
 hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
 lower_red = np.array([150, 43,46])
 upper_red = np.array([180, 255, 255])
 red_mask = cv2.inRange(hsv, lower_red, upper_red)
 red = cv2.bitwise_and(img, img, mask=red_mask)
 im_in = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)
 ret, im_th = cv2.threshold(im_in, 0, 255, cv2.THRESH_BINARY)
 im_out = cv2.dilate(im_th, kernel, iterations=5)
 list1.append(im_out)
###########################################################################

###########################################################################
#填充二值化后的图片，锚定ROI坐标后画圆圈出
roi_list = []
for j in range(len(list1)):
    a, b, c, d = cv2.connectedComponentsWithStats(list1[j],connectivity=4)#ltype=cv2.CV_32S
    for t in range(1, a, 1):
        x, y, w, h, area = c[t]
        roi_list.append((x, y, w, h))
    #q=cv2.rectangle(img, (x, y), (x+w,y+h), (0, 255, 0), 1, 8, 0)
##############################################################################
def pop_H(image, low, up):
    count = 0
    mask = Interval(low, up)
    b = image[:,:,0]
    a = b.flatten()
    for j in a:
        if j in mask:
            count = count + 1
    return count
###############################################################################
#单独保存ROI区域
def saveRoi(src, roi_list):
    list_roi = []
    list_roi_target = []
    list_roi_noise = []
    for i in range(len(roi_list)):
        x, y, w, h = roi_list[i]
        roi = src[y:y+h, x:x+w]
        list_roi.append(roi)
    for j in range(len(list_roi)):
        if pop_H(list_roi[j],150,180) > 50 :
            list_roi_target.append(list_roi[j])
        else:
            list_roi_noise.append(list_roi[j])
    return list_roi_target

for k in range(len(saveRoi(img,roi_list))):
   cv2.imwrite("roi_%d.png"%k, saveRoi(img,roi_list)[k],[int(cv2.IMWRITE_JPEG_QUALITY), 95])

