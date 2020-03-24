import cv2
import torch
from model import simpleNet
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

model = simpleNet()
model.load_state_dict(torch.load("model.pkl"))
model.train(False)


def target_pos(img):
    transform = transforms.Compose([
        # transforms.Resize((128, 72)),
        transforms.ToTensor(),
    ])
    with torch.no_grad():

        # img = Image.fromarray(img)
        img = transform(img).unsqueeze(0)

        output = model(img)
        # print(output)
        return output.numpy()


vis = Image.open("test2.jpg")

vis = vis.resize((128, 72), Image.ANTIALIAS)


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


pos = target_pos(vis)
im = cv2.imread("test2.jpg")
im = cv2.resize(im, (1280, 720))
cv2.circle(im, (int(pos[0][0] * 1280), int(pos[0][1] * 720)), 10, (0, 0, 255), 3)
cv2.circle(im, (int(pos[0][2] * 1280), int(pos[0][3] * 720)), 10, (0, 0, 255), 3)
cv2.circle(im, (int(pos[0][4] * 1280), int(pos[0][5] * 720)), 10, (0, 0, 255), 3)
print(sigmoid(pos[0][6]))
plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
plt.show()


# cam = cv2.VideoCapture(0)
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
# height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
# print("Image Size: %d x %d" % (width, height))
# while True:
#     ret, img = cam.read()
#     cv2.imwrite("test3.jpg", img)
#     vis = img.copy()
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     pos = target_pos(img)
#     cv2.circle(vis, (int(pos[0][0] * 1280), int(pos[0][1] * 720)), 10, (0, 0, 255), 1)
#     cv2.circle(vis, (int(pos[0][2] * 1280), int(pos[0][3] * 720)), 10, (0, 0, 255), 1)
#     cv2.circle(vis, (int(pos[0][4] * 1280), int(pos[0][5] * 720)), 10, (0, 0, 255), 1)
#     cv2.imshow("getCamera", vis)
#     print(sigmoid(pos[0][6]))
#     if 0xFF & cv2.waitKey(5) == 27:
#         break
# cv2.destroyAllWindows()
