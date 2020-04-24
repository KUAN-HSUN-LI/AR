import argparse
import cv2
import numpy as np
from math import acos, degrees
import sympy as sy

FOCAL = 4.73 * 10**-3
PIXEL_DIS = 1.6 * 10**-6
REAL_A = np.array([0, 0, 0])
REAL_B = np.array([0.275, 0.058, 0])
REAL_C = np.array([0.21, -0.083, 0])
CAMERA = np.array([0, 0, 1.04], dtype=float)
img_height = FOCAL


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_img", "-i", type=image, default="Short.png")
    parser.add_argument("--mode", "-m", default="newton")
    args = parser.parse_args()
    return args


def image(img_name):
    img = cv2.imread(img_name)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def find_marker(img, rgb):
    dot = np.where((img[:, :, 0] == rgb[0]) & (img[:, :, 1] == rgb[1]) & (img[:, :, 2] == rgb[2]))
    return np.array(dot).squeeze(-1)


def calc_dist(pointA, pointB, zoom=1):
    return np.sqrt(np.sum((pointA - pointB)**2)) * zoom


def calc_img_angle(img_center, pointA, pointB, img_height):
    img_center2A = calc_dist(img_center, pointA, PIXEL_DIS)
    img_center2B = calc_dist(img_center, pointB, PIXEL_DIS)
    dist_AB = calc_dist(pointA, pointB, PIXEL_DIS)
    height = img_height
    dist_OA = np.sqrt(img_center2A ** 2 + height ** 2)
    dist_OB = np.sqrt(img_center2B ** 2 + height ** 2)
    return acos((dist_OA ** 2 + dist_OB ** 2 - dist_AB ** 2) / (2 * dist_OA * dist_OB))


def calc_real_angle(pointA, pointO, pointB):
    dist_AO = calc_dist(pointA, pointO)
    dist_OB = calc_dist(pointO, pointB)
    dist_AB = calc_dist(pointA, pointB)

    return acos(np.dot(pointO - pointA, pointO - pointB) / (dist_OB*dist_AO))


def check_move(camera, moveX, moveY, moveZ):
    try:
        check = 0
        img_angle_AOB = calc_img_angle(img_center, iconA, iconB, img_height)
        img_angle_BOC = calc_img_angle(img_center, iconB, iconC, img_height)
        img_angle_COA = calc_img_angle(img_center, iconC, iconA, img_height)
        temp_angle_AOB = calc_real_angle(REAL_A, CAMERA, REAL_B)
        temp_angle_BOC = calc_real_angle(REAL_B, CAMERA, REAL_C)
        temp_angle_COA = calc_real_angle(REAL_C, CAMERA, REAL_A)
        temp_err = (img_angle_AOB - temp_angle_AOB)**2 + (img_angle_BOC - temp_angle_BOC)**2 + (img_angle_COA - temp_angle_COA)**2
        camera[0] += moveX
        camera[1] += moveY
        camera[2] += moveZ
        new_angle_AOB = calc_real_angle(REAL_A, camera, REAL_B)
        new_angle_BOC = calc_real_angle(REAL_B, camera, REAL_C)
        new_angle_COA = calc_real_angle(REAL_C, camera, REAL_A)
        new_err = (img_angle_AOB - new_angle_AOB)**2 + (img_angle_BOC - new_angle_BOC)**2 + (img_angle_COA - new_angle_COA)**2
        if new_err < temp_err:
            return True
    except ValueError:
        return False
    return False


F = sy.acos("img_angle") - sy.acos("((x-a1)*(x-a2) + (y-b1)*(y-b2) + (z-c1)*(z-c2))/(((x-a1)**2+(y-b1)**2+(z-c1)**2)**(1/2) * ((x-a2)**2+(y-b2)**2+(z-c2)**2)**(1/2))")
DIFFX = sy.diff(F, sy.Symbol('x'))
DIFFY = sy.diff(F, sy.Symbol('y'))
DIFFZ = sy.diff(F, sy.Symbol('z'))


def angle_diff(img_angle, pointO, pointA, pointB, diff):
    x, y, z = pointO
    a1, b1, c1 = pointA
    a2, b2, c2 = pointB
    IMG_ANGLE = sy.Symbol("img_angle")
    X = sy.Symbol('x')
    Y = sy.Symbol('y')
    Z = sy.Symbol('z')
    A1 = sy.Symbol('a1')
    B1 = sy.Symbol('b1')
    C1 = sy.Symbol('c1')
    A2 = sy.Symbol('a2')
    B2 = sy.Symbol('b2')
    C2 = sy.Symbol('c2')
    return sy.N(diff.xreplace({IMG_ANGLE: img_angle, X: x, Y: y, Z: z, A1: a1, B1: b1, C1: c1, A2: a2, B2: b2, C2: c2}))


if __name__ == "__main__":
    args = get_args()
    # init
    img = args.input_img
    img_center = np.array([img.shape[0] / 2, img.shape[1] / 2], dtype=int)
    iconA = find_marker(img, (255, 0, 0))
    iconB = find_marker(img, (255, 100, 0))
    iconC = find_marker(img, (0, 0, 255))

    dist_AB = calc_dist(iconA, iconB, PIXEL_DIS)
    real_AB = calc_dist(REAL_A, REAL_B)
    S = FOCAL * (dist_AB + real_AB) / dist_AB
    CAMERA[2] = S

    img_angle_AOB = calc_img_angle(img_center, iconA, iconB, FOCAL)
    img_angle_BOC = calc_img_angle(img_center, iconB, iconC, FOCAL)
    img_angle_COA = calc_img_angle(img_center, iconC, iconA, FOCAL)

    real_angle_AOB = calc_real_angle(REAL_A, CAMERA, REAL_B)
    real_angle_BOC = calc_real_angle(REAL_B, CAMERA, REAL_C)
    real_angle_COA = calc_real_angle(REAL_C, CAMERA, REAL_A)
    if args.mode == "newton":
        step = 1.0
        while step >= 1e-3:
            temp_camera = CAMERA.copy()
            temp_angle_AOB = calc_real_angle(REAL_A, CAMERA, REAL_B)
            temp_angle_BOC = calc_real_angle(REAL_B, CAMERA, REAL_C)
            temp_angle_COA = calc_real_angle(REAL_C, CAMERA, REAL_A)
            if check_move(CAMERA.copy(), step, 0, 0):
                CAMERA[0] += step
            if check_move(CAMERA.copy(), -step, 0, 0):
                CAMERA[0] -= step
            if check_move(CAMERA.copy(), 0, step, 0):
                CAMERA[1] += step
            if check_move(CAMERA.copy(), 0, -step, 0):
                CAMERA[1] -= step
            if check_move(CAMERA.copy(), 0, 0, step):
                CAMERA[2] += step
            if check_move(CAMERA.copy(), 0, 0, -step):
                CAMERA[2] -= step
            if np.array_equal(CAMERA, temp_camera):
                step *= 0.1
    elif args.mode == "differential":
        lr = 0.05
        for idx in range(500):
            print(idx, end='\r')
            diffX = 0
            diffY = 0
            diffZ = 0
            real_angle_AOB = calc_real_angle(REAL_A, CAMERA, REAL_B)
            diffX += angle_diff(img_angle_AOB, CAMERA, REAL_A, REAL_B, DIFFX) * (img_angle_AOB - real_angle_AOB)
            diffY += angle_diff(img_angle_AOB, CAMERA, REAL_A, REAL_B, DIFFY) * (img_angle_AOB - real_angle_AOB)
            diffZ += angle_diff(img_angle_AOB, CAMERA, REAL_A, REAL_B, DIFFZ) * (img_angle_AOB - real_angle_AOB)

            real_angle_BOC = calc_real_angle(REAL_B, CAMERA, REAL_C)
            diffX += angle_diff(img_angle_BOC, CAMERA, REAL_B, REAL_C, DIFFX) * (img_angle_BOC - real_angle_BOC)
            diffY += angle_diff(img_angle_BOC, CAMERA, REAL_B, REAL_C, DIFFY) * (img_angle_BOC - real_angle_BOC)
            diffZ += angle_diff(img_angle_BOC, CAMERA, REAL_B, REAL_C, DIFFZ) * (img_angle_BOC - real_angle_BOC)

            real_angle_COA = calc_real_angle(REAL_C, CAMERA, REAL_A)
            diffX += angle_diff(img_angle_COA, CAMERA, REAL_B, REAL_C, DIFFX) * (img_angle_COA - real_angle_COA)
            diffY += angle_diff(img_angle_COA, CAMERA, REAL_B, REAL_C, DIFFY) * (img_angle_COA - real_angle_COA)
            diffZ += angle_diff(img_angle_COA, CAMERA, REAL_B, REAL_C, DIFFZ) * (img_angle_COA - real_angle_COA)
            CAMERA[0] -= diffX * lr
            CAMERA[1] -= diffY * lr
            CAMERA[2] -= diffZ * lr
    img_angle_AOB = calc_img_angle(img_center, iconA, iconB, FOCAL)
    img_angle_BOC = calc_img_angle(img_center, iconB, iconC, FOCAL)
    img_angle_COA = calc_img_angle(img_center, iconC, iconA, FOCAL)

    real_angle_AOB = calc_real_angle(REAL_A, CAMERA, REAL_B)
    real_angle_BOC = calc_real_angle(REAL_B, CAMERA, REAL_C)
    real_angle_COA = calc_real_angle(REAL_C, CAMERA, REAL_A)
    print("A position:", REAL_A)
    print("B position:", REAL_B)
    print("C position:", REAL_C)
    print("camera position:", CAMERA)
    print(f"distance OA = {calc_dist(CAMERA, REAL_A):0.4f} m")
    print(f"distance OB = {calc_dist(CAMERA, REAL_B):0.4f} m")
    print(f"distance OC = {calc_dist(CAMERA, REAL_C):0.4f} m")
