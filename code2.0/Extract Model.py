import cv2
import numpy as np
from interval import Interval

###Extract



list1 = []
kernel = np.ones((3, 3), np.uint8)
img_0 = cv2.imread('E:/3.tif')
img = np.pad(img_0,((10,10),(10,10),(0,0)),'constant',constant_values=(0,0))
hsv_l = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


lower_red = np.array([150, 43,46])
upper_red = np.array([180, 255, 255])
red_mask = cv2.inRange(hsv_l, lower_red, upper_red)
red = cv2.bitwise_and(img,img, mask=red_mask)
im_in = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)

ret, im_th = cv2.threshold(im_in, 0, 255, cv2.THRESH_BINARY)
im_out = cv2.dilate(im_th, kernel, iterations=3)
list1.append(im_out)


roi_list = []
for j in range(len(list1)):
    a, b, c, d = cv2.connectedComponentsWithStats(list1[j],connectivity=4)#ltype=cv2.CV_32S
    for t in range(1, a, 1):
        x, y, w, h, area = c[t]
        roi_list.append((x, y, w, h))

def pop_H(image, low, up):
    count = 0
    mask = Interval(low, up)
    b = image[:,:,0]
    a = b.flatten()
    for j in a:
        if j in mask:
            count = count + 1
    return count

def saveRoi(src, roi_list):
    list_roi = []
    list_roi_hsv = []
    list_roi_target = []
    list_roi_noise = []

    for i in range(len(roi_list)):
        x, y, w, h = roi_list[i]
        roi = src[y-5:y+h+2, x-5:x+w+2]
        list_roi.append(roi)
    for p in range(1,len(list_roi),1):
        roi_hsv = cv2.cvtColor(list_roi[p], cv2.COLOR_BGR2HSV)
        list_roi_hsv.append(roi_hsv)

    for j in range(len(list_roi_hsv)):
        if pop_H(list_roi_hsv[j],160,180)>30 :
            list_roi_target.append(cv2.cvtColor(list_roi_hsv[j], cv2.COLOR_HSV2BGR))

        else:
            list_roi_noise.append(list_roi[j])
    return list_roi_target,list_roi_noise

a,b = saveRoi(img,roi_list)



for k in range(len(a)):
  cv2.imwrite("roi_%d.png"%k, a[k],[int(cv2.IMWRITE_JPEG_QUALITY), 95])



