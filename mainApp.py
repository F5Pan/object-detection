import cv2
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
from keras.utils import np_utils
from keras.models import load_model
import drugVerify
import ImageCut


m1 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
m2 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
imgs = []
result = []
folderName = 'BBB'
grayfolderName= 'BBB'
picnum =11
namestr = 'b'

# 第一步：高斯
# oranginalimg = cv2.imread("D:\\master\\"+folderName+"\\"+namestr+str(picnum)+".jpg")
# oranginalimg = cv2.imread("D:\\master\\images\\manyImages\\S__6619259.jpg")
oranginalimg = cv2.imread("D:\\master\\images\\test2.jpg")


# oranginalimg = cv2.resize(oranginalimg, (380, 380))
# img = cv2.resize(oranginalimg, (380, 380))
img = oranginalimg.copy()

sobel = cv2.Canny(img, 50, 100)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (5, 5), 2)



h, w = img.shape[:2]                        #获取图像的高和宽
mask = np.zeros((h+2, w+2), np.uint8)       #掩码长和宽都比输入图像多两个像素点，泛洪填充不会超出掩码的非零边缘
#进行泛洪填充
cv2.floodFill(img, mask, (w-1,h-1), (255,255,255), (2,2,2),(3,3,3),8)
cv2.imshow("floodfill", img)

img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imshow("threshold", img)

# 第二步：計算每一點的梯度與方向找到邊緣強度
img1 = np.zeros(img.shape, dtype="uint8")  # 與原圖大小相同
theta = np.zeros(img.shape, dtype="float")  # 方向矩陣原圖像大小
rows, cols = img.shape
for i in range(1, rows - 1):
    for j in range(1, cols - 1):
        Gy = [np.sum(m2 * img[i - 1:i + 2, j - 1:j + 2])]
        Gx = [np.sum(m1 * img[i - 1:i + 2, j - 1:j + 2])]

        # 计算角度
        if Gx[0] == 0:
            theta[i - 1, j - 1] = 90
            continue
        else:
            temp = ((np.arctan2(Gy[0], Gx[0])) * 180 / np.pi) + 90

        if Gx[0] * Gy[0] > 0:
            if Gx[0] > 0:
                # 第一象限
                theta[i - 1, j - 1] = np.abs(temp)
            else:
                # 第三象限
                theta[i - 1, j - 1] = (np.abs(temp) - 180)
        if Gx[0] * Gy[0] < 0:
            if Gx[0] > 0:
                # 第四象限
                theta[i - 1, j - 1] = (-1) * np.abs(temp)
            else:
                # 第二象限
                theta[i - 1, j - 1] = 180 - np.abs(temp)
        # 圖像梯度的幅值
        img1[i - 1, j - 1] = (np.sqrt(Gx[0] ** 2 + Gy[0] ** 2))

# 計算梯度方向：四個角度進行量化
for i in range(1, rows - 2):
    for j in range(1, cols - 2):
        if (((theta[i, j] >= -22.5) and (theta[i, j] < 22.5)) or
                ((theta[i, j] <= -157.5) and (theta[i, j] >= -180)) or
                ((theta[i, j] >= 157.5) and (theta[i, j] < 180))):
            theta[i, j] = 0.0
        elif (((theta[i, j] >= 22.5) and (theta[i, j] < 67.5)) or
              ((theta[i, j] <= -112.5) and (theta[i, j] >= -157.5))):
            theta[i, j] = -45.0
        elif (((theta[i, j] >= 67.5) and (theta[i, j] < 112.5)) or
              ((theta[i, j] <= -67.5) and (theta[i, j] >= -112.5))):
            theta[i, j] = 90.0
        elif (((theta[i, j] >= 112.5) and (theta[i, j] < 157.5)) or
              ((theta[i, j] <= -22.5) and (theta[i, j] >= -67.5))):
            theta[i, j] = 45.0

# 第三步：非極大值抑制計算
img2 = np.zeros(img1.shape)  # 非極大值抑制圖片矩陣
# 非极大值抑制即为沿着上述4种类型的梯度方向，比较3*3邻域内对应邻域值的大小：
for i in range(1, img2.shape[0] - 1):
    for j in range(1, img2.shape[1] - 1):
        # 0度为水平边缘
        if (theta[i, j] == 0.0) and (img1[i, j] == np.max([img1[i, j], img1[i + 1, j], img1[i - 1, j]])):
            img2[i, j] = img1[i, j]
        # -45度边缘
        if (theta[i, j] == -45.0) and img1[i, j] == np.max([img1[i, j], img1[i - 1, j - 1], img1[i + 1, j + 1]]):
            img2[i, j] = img1[i, j]
        # 90度垂直边缘
        if (theta[i, j] == 90.0) and img1[i, j] == np.max([img1[i, j], img1[i, j + 1], img1[i, j - 1]]):
            img2[i, j] = img1[i, j]
        # 45度边缘
        if (theta[i, j] == 45.0) and img1[i, j] == np.max([img1[i, j], img1[i - 1, j + 1], img1[i + 1, j - 1]]):
            img2[i, j] = img1[i, j]

# 第四步：双阈值检测和边缘连接
img3 = np.zeros(img2.shape)  # 定义双阈值图像
TL = 26
TH = 130  # 关键在这两个阈值的选择
for i in range(1, img3.shape[0] - 1):
    for j in range(1, img3.shape[1] - 1):
        if img2[i, j] < TL:   
            img3[i, j] = 0
        elif img2[i, j] > TH:
            img3[i, j] = 255
        # 将小于高阈值，大于低阈值的点使用8连通区域确定（即：只有与TH像素连接时才会被接受，成为边缘点，赋255）
        elif ((img2[i + 1, j] < TH) or (img2[i - 1, j] < TH) or (img2[i, j + 1] < TH) or
              (img2[i, j - 1] < TH) or (img2[i - 1, j - 1] < TH) or (img2[i - 1, j + 1] < TH) or
              (img2[i + 1, j + 1] < TH) or (img2[i + 1, j - 1] < TH)):
            img3[i, j] = 255



cv2.imshow("original_img", oranginalimg)  # 原始图像
cv2.imshow("gray_img", img)  # 灰階高斯
cv2.imshow("Opencv_canny", sobel)  # 角度值灰度图
cv2.imshow("grad_img", img1)  # 梯度幅值图
cv2.imshow("max_img", img2)  # 非极大值抑制灰度图
cv2.imshow("final_img", img3)  # 双阈值检测和边缘连接


cv2.imwrite('thresholdTOcanny.jpg',img3)
img3 = np.array(img3,np.uint8)


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
closed = cv2.morphologyEx(img3, cv2.MORPH_CLOSE, kernel)
# cv2.imshow('closed1', closed)

# perform a series of erosions and dilations
# closed = cv2.erode(closed, None, iterations=4)
# closed = cv2.dilate(closed, None, iterations=4)
#
# cv2.imshow('closed2', closed)

contours,hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
c = sorted(contours, key=cv2.contourArea, reverse=True)[0]



# compute the rotated bounding box of the largest contour
rect = cv2.minAreaRect(c)
box = np.int0(cv2.boxPoints(rect))

# draw a bounding box arounded the detected barcode and display the image
# cv2.drawContours(img, [box], -1, (0, 255, 0), 2)
# cv2.imshow("Image", img)

split_res = oranginalimg.copy()
cutImage = oranginalimg.copy()
num =0
for cnt in contours:

  rect = cv2.minAreaRect(cnt)
  box = cv2.boxPoints(rect)
  box = np.int0(box)
  # print('box',box)
  x, y, w, h = cv2.boundingRect(cnt)
  area = w * h
  print(area)
  print(' x, y, w, h', x, y, w, h)
  if 250 < area:
   cv2.rectangle(split_res, (x, y), (x + w, y + h), (0, 0, 255), 2)

   num += 1
   # cutImg = Rotate(cutImage, box, rect).getImg()
   cutImg = ImageCut.Rotate(cutImage, box, rect).getImg()
   size = cutImg.shape
   w = size[1]  # 宽度
   h = size[0]  # 高度
   print(size)
   print(w)
   print(h)
   # cv2.imshow('cutImg', cutImg)

   if w == 0:
       continue
   elif h==0:
       continue

   if w>h:
       output_ROTATE_90_CLOCKWISE = cv2.rotate(cutImg, cv2.ROTATE_90_CLOCKWISE)
       # cv2.imshow('output_ROTATE_90_CLOCKWISE', output_ROTATE_90_CLOCKWISE)
   else:
       output_ROTATE_90_CLOCKWISE = cutImg

   # imgs.append(cutImg)
   # cv2.imshow(f"cutImg-{num}",output_ROTATE_90_CLOCKWISE)
   # saveImg = cv2.resize(output_ROTATE_90_CLOCKWISE, (133, 201))
   # cv2.imwrite("D:\\master\\"+grayfolderName+"\\"+namestr+str(picnum)+".jpg",saveImg);
   # result = verify.predict(saveImg)
   # print(result)
   # cv2.putText(split_res, result, (x,y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)


cv2.imshow('split_res', split_res)
cv2.waitKey(0)
# print('end')


