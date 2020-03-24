import cv2
import numpy as np


img = cv2.imread("test1.jpg")

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # HSV空间

lower_blue = np.array([78, 50, 50])  # blue
upper_blue = np.array([99, 255, 255])

lower_green = np.array([150, 150, 150])  # green
upper_green = np.array([255, 255, 255])

lower_red = np.array([0, 100, 100])  # red
upper_red = np.array([10, 150, 255])

red_mask = cv2.inRange(hsv, lower_red, upper_red)  # 取红色
blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)  # 蓝色
black_mask = cv2.inRange(img, lower_green, upper_green)  # 绿色

red = cv2.bitwise_and(img, img, mask=red_mask)  # 对原图像处理
green = cv2.bitwise_and(img, img, mask=black_mask)
blue = cv2.bitwise_and(img, img, mask=blue_mask)

res = green+red+blue
# print(hsv[225][1225][:])
# cv2.imshow('img', hsv[:800][:800][:])
cv2.imshow('img', img[:950][:950])
cv2.waitKey(0)

cv2.destroyAllWindows()
