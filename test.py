import cv2
from matplotlib import pyplot as plt


# 第一步：高斯
# oranginalimg = cv2.imread("C:\\Users\\wayne\\testImage\\drug14.jpg")
img = cv2.imread("D:\\master\\images\\background\\solidcolor\\S__7422006.jpg")
# oranginalimg = cv2.imread("D:\\master\\images\\1\\A1.jpg")
# img = cv2.cvtColor(oranginalimg, cv2.COLOR_BGR2RGB)
#


equalized_img = cv2.equalizeHist(img)
cv2.imshow("equal_image", equalized_img)