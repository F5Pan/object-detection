import cv2
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
from keras.utils import np_utils
from keras.models import load_model

class Verify:

    def __init__(self, img):

        self.img = img  # 原始图像

    def verify(self):
        reImage = cv2.resize(self.img, (300, 300))

        imageCvt = cv2.cvtColor(reImage, cv2.COLOR_BGR2GRAY)
        grayImage = imageCvt
        # cv.imshow('number',grayImage)
        testImage = reImage.reshape(1, 300, 300, 3).astype('float32') / 255

        model = load_model("drug.h5")
        resultRate = model.predict(testImage)
        result = np.argmax(resultRate, axis=1)
        resultRate1 = '紅膠囊= {:.2f},三角 = {:.2f},圓形= {:.2f},藍膠囊= {:.2f}'.format(resultRate[0][0], resultRate[0][1], resultRate[0][2],resultRate[0][3])

        print(resultRate1)

        rate = resultRate[0][0]
        rate1 = resultRate[0][1]
        rate2 = resultRate[0][2]
        rate3 = resultRate[0][3]

        print(rate)
        print(rate1)
        print(rate2)
        print(rate3)


        if (rate > 0.8):
            str = 'CapsuleRed'
        elif (rate1 > 0.8):
            str = 'TriangleMintsRed'
        elif (rate2 > 0.8):
            str = 'CircleYellow'
        elif (rate3 > 0.8):
            str = 'CapsuleBlue'
        else:
            str = '都不是'
        return str

class Rotate:
    def __init__(self, img, box, rect):

        self.img = img  # 原始图像
        self.box = box  # cv2.boxPoints(rect) for OpenCV 3.x 获取最小外接矩形的4个顶点坐标
        self.rect = rect  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）


    def getImg(self):
        # 做旋转图像
        # 这里外接矩形的旋转角度是无法确定的，
        # 且由于**作者个人**的素材图片的最大选旋转角度不会超过45°,所以这里加以限制
        a = self.imagecrop(self.img, np.int0(box))
        return a

    def imagecrop(self, image, box):
        xs = [x[1] for x in box]
        ys = [x[0] for x in box]
        # print(xs)
        # print(min(xs), max(xs), min(ys), max(ys))
        cropimage = image[min(xs):max(xs), min(ys):max(ys)]
        # print(cropimage.shape)
        # cv2.imwrite('cropimage.png', cropimage)
        return cropimage




m1 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
m2 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
imgs = []
result = []
num1 =23
# 第一步：高斯
# oranginalimg = cv2.imread("C:\\Users\\wayne\\testImage\\drug14.jpg")
oranginalimg = cv2.imread("D:\\master\\A\\A"+str(num1)+".jpg")
oranginalimg = cv2.resize(oranginalimg, (380, 380))
img = cv2.resize(oranginalimg, (380, 380))
sobel = cv2.Canny(img, 50, 100)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (5, 5), 2)

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



# cv2.imshow("original_img", oranginalimg)  # 原始图像
# cv2.imshow("gray_img", img)  # 灰階高斯
# cv2.imshow("Opencv_canny", sobel)  # 角度值灰度图
# cv2.imshow("grad_img", img1)  # 梯度幅值图
# cv2.imshow("max_img", img2)  # 非极大值抑制灰度图
cv2.imshow("final_img", img3)  # 最终效果图

img3 = np.array(img3,np.uint8)


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
closed = cv2.morphologyEx(img3, cv2.MORPH_CLOSE, kernel)
cv2.imshow('closed1', closed)

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
  # print(' x, y, w, h', x, y, w, h)
  if 250 < area:
   cv2.rectangle(split_res, (x, y), (x + w, y + h), (0, 0, 255), 2)


   num += 1
   imgs.append(Rotate(cutImage, box, rect).getImg())
   cv2.imshow(f"cutImg-{num}",Rotate(cutImage, box, rect).getImg())
   result = Verify(Rotate(cutImage, box, rect).getImg()).verify()
   print(result)
   cv2.putText(split_res, result, (x,y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)


for i in range(len(imgs)):
    cv2.imshow(f"img-{i}", imgs[i])
    new_img=cv2.resize(imgs[i],(380,380))
    # new_img.tofile("D:/master/AA/" + 'a'+ str(num1)+'.jpg')
    # cv2.imwrite("D:/master/AA/" + 'a'+ str(num1)+'.jpg', new_img)

cv2.imshow('split_res', split_res)
cv2.waitKey(0)
# print('end')
