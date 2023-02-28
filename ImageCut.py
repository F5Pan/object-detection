import numpy as np
import cv2

class Rotate:
    def __init__(self, img, box, rect):

        self.img = img  # 原始图像
        self.box = box  # cv2.boxPoints(rect) for OpenCV 3.x 获取最小外接矩形的4个顶点坐标
        self.rect = rect  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）


    def getImg(self):
        # 做旋转图像
        # 这里外接矩形的旋转角度是无法确定的，
        # 且由于**作者个人**的素材图片的最大选旋转角度不会超过45°,所以这里加以限制
        a = self.imagecrop(self.img, np.int0(self.box))
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

    def rotateimg(img):
        (h, w, d) = img.shape  # 讀取圖片大小
        center = (w // 2, h // 2)  # 找到圖片中心

        # 第一個參數旋轉中心，第二個參數旋轉角度(-順時針/+逆時針)，第三個參數縮放比例
        M = cv2.getRotationMatrix2D(center, 90, 1.0)

        # 第三個參數變化後的圖片大小
        rotate_img = cv2.warpAffine(img, M, (w, h))

        return rotate_img