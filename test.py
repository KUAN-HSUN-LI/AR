####################################################
#   作者:zhigang,
####################################################
import cv2
import numpy as np
import matplotlib.pyplot as plt


class ShapeAnalysis:
    def __init__(self):
        self.shapes = {'triangle': 0, 'rectangle': 0, 'polygons': 0, 'circles': 0}

    def analysis(self, frame):
        h, w, ch = frame.shape
        result = np.zeros((h, w, ch), dtype=np.uint8)
        # 二值化圖像
        print("start to detect lines...\n")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, edged_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        binary = cv2.GaussianBlur(frame, (3, 3), 0)
        edged_img = cv2.Canny(binary, 35, 125)
        # frame1 = cv2.resize(frame, (128, 72))
        plt.imshow(cv2.cvtColor(edged_img, cv2.COLOR_BGR2RGB))
        plt.show()

        contours, hierarchy = cv2.findContours(edged_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in range(len(contours)):
            # 提取與繪制輪廓

            # 輪廓逼近
            epsilon = 0.01 * cv2.arcLength(contours[cnt], True)
            approx = cv2.approxPolyDP(contours[cnt], epsilon, True)

            # 分析幾何形狀
            corners = len(approx)
            shape_type = ""
            if corners == 3:
                count = self.shapes['triangle']
                count = count+1
                self.shapes['triangle'] = count
                shape_type = "三角形"
                # continue
            if corners == 4:
                # continue
                count = self.shapes['rectangle']
                count = count + 1
                self.shapes['rectangle'] = count
                shape_type = "矩形"
            if corners >= 10:
                # continue
                count = self.shapes['circles']
                count = count + 1
                self.shapes['circles'] = count
                shape_type = "圓形"
            if 4 < corners < 10:
                # continue
                count = self.shapes['polygons']
                count = count + 1
                self.shapes['polygons'] = count
                # 求解中心位置
            mm = cv2.moments(contours[cnt])
            if mm['m00'] == 0.0:
                continue
            cx = int(mm['m10'] / mm['m00'])
            cy = int(mm['m01'] / mm['m00'])

            # 顏色分析
            color = frame[cy][cx]
            # if color[0] > 1:
            #     continue
            # if not (color[2] > 100 and color[1] < 100 and color[0] < 100):
            #     continue
            if not (color[2] < 150 and color[1] < 150 and color[0] > 150):
                continue
            # if color[1] > 1:
            #     continue
            color_str = "(" + str(color[0]) + ", " + str(color[1]) + ", " + str(color[2]) + ")"

            # 計算面積與周長
            p = cv2.arcLength(contours[cnt], True)
            area = cv2.contourArea(contours[cnt])
            # if area < 100:
            #     continue
            # print("周長: %.3f, 面積: %.3f 顏色: %s 形狀: %s " % (p, area, color_str, shape_type))
            cv2.circle(result, (cx, cy), 3, (0, 0, 255), -1)
            cv2.drawContours(result, contours, cnt, (0, 255, 0), 2)
            # print(contours)
            cv2.fillPoly(result, pts=[contours[cnt]], color=(255, 0, 0))
        plt.imshow(cv2.cvtColor(self.draw_text_info(result), cv2.COLOR_BGR2RGB))
        plt.show()
        return self.shapes

    def draw_text_info(self, image):
        c1 = self.shapes['triangle']
        c2 = self.shapes['rectangle']
        c3 = self.shapes['polygons']
        c4 = self.shapes['circles']
        cv2.putText(image, "triangle: "+str(c1), (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 1)
        cv2.putText(image, "rectangle: " + str(c2), (10, 40), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 1)
        cv2.putText(image, "polygons: " + str(c3), (10, 60), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 1)
        cv2.putText(image, "circles: " + str(c4), (10, 80), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 1)
        return image


if __name__ == "__main__":
    src = cv2.imread("test1.jpg")
    print(src.shape)
    ld = ShapeAnalysis()
    ld.analysis(src)
