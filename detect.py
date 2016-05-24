# -*- coding: utf-8 -*-

import cv2
import math
import numpy as np

PI = 3.14159

class Ray():

    """The stroke width ray. """

    def __init__(self, start):
        """init the ray with the start point. """
        self.start = start
        self.points = [start]

    def add(self, point):
        self.points.append(point)

    def set_end(self, end):
        self.end = end

        
def show(title, img):
    cv2.imshow(title, img)
    cv2.waitKey()


def swt(canny, gradient_x, gradient_y, flag):
    h, w = canny.shape
    swt_img = np.ones((h, w)) * (-1)

    # compute sw of each pixel.
    rays = []
    pres = 0.05
    for x in range(h):
        for y in range(w):
            if canny[x][y] > 0:
                ray = Ray((x, y))
                # if gradient[x][y] == 0.0:
                #     print "why gradient is 0?"
                #     continue
                # g_x = gradient_x[x][y]
                # g_y = gradient_y[x][y]
                # gradient = math.sqrt(g_x * g_x + g_y * g_y)
                # g_x /= gradient
                # g_y /= gradient
                g_x = gradient_x[x][y]
                g_y = gradient_y[x][y]
                gradient = math.sqrt(gradient_x[x][y]*gradient_x[x][y] + gradient_y[x][y]*gradient_y[x][y])
                gradient_com = math.sqrt(g_x * g_x + g_y * g_y)
                g_x /= gradient * flag
                g_y /= gradient * flag
                last_x = cur_x = x
                last_y = cur_y = y
                while True: 
                    cur_x += g_x * pres
                    cur_y += g_y * pres

                    if int(math.floor(cur_x)) == last_x and int(math.floor(cur_y)) == last_y:
                        continue

                    if cur_x < 0 or cur_x >= h or cur_y < 0 or cur_y >= w:
                        break
                    
                    last_x = int(math.floor(cur_x))
                    last_y = int(math.floor(cur_y))

                    ray.add((last_x, last_y))

                    if canny[last_x][last_y] > 0:
                        ray.set_end((last_x, last_y))

                        # if gradient[last_x][last_y] == 0.0:
                        #     print "why gradient is 0 at while loop?"
                        #     break
                        g_cx = gradient_x[last_x][last_y]
                        g_cy = gradient_y[last_x][last_y]
                        gradient = math.sqrt(g_cx * g_cx + g_cy * g_cy)
                        g_cx /= gradient * flag
                        g_cy /= gradient * flag

                        try:
                            if abs(g_x * g_cx + g_y * g_cy) -1 < 0.00001 or math.acos(g_x * g_cx + g_y * g_cy) < PI / 2.0:
                                minus = (ray.end[0] - ray.start[0], ray.end[1] - ray.start[1])
                                length = np.dot(minus, minus)
                                for point in ray.points:
                                    if swt_img[point[0]][point[1]] < 0:
                                        swt_img[point[0]][point[1]] = length
                                    if swt_img[point[0]][point[1]] > length:
                                        swt_img[point[0]][point[1]] = length
                                rays.append(ray)

                                break

                            else:
                                continue

                        except ValueError:
                            print g_x * (g_cx) + g_y * (g_cy)
                            pass
    # medianlize the stroke width
    for ray in rays:
        swt_array = []
        for point in ray.points:
            swt_array.append(swt_img[point[0]][point[1]])

        sorted_sw = sorted(swt_array, key=lambda x: x)
        
        median = sorted_sw[len(ray.points)/2]

        for point in ray.points:
            swt_img[point[0]][point[1]] = min(median, swt_img[point[0]][point[1]])

    return swt_img, rays 


def detect_text(im):
    # 1. get the canny and gradient.
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    high = ret
    low = 0.5 * high
    canny = cv2.Canny(gray, low, high, 3)
    cv2.imwrite("canny.jpg", canny)

    h, w = gray.shape
    # scaled = np.zeros((h, w, 1), np.float32)
    scaled = cv2.normalize(gray, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    smoothd = cv2.GaussianBlur(scaled, (5, 5), sigmaX=0)

    scharr_x = cv2.Scharr(smoothd, -1, 1, 0)
    scharr_y = cv2.Scharr(smoothd, -1, 0, 1)

    scharr_x = cv2.GaussianBlur(scharr_x, (3, 3), sigmaX=0)
    scharr_y = cv2.GaussianBlur(scharr_y, (3, 3), sigmaX=0)

    cv2.imwrite("scharr_x.jpg", scharr_x)
    cv2.imwrite("scharr_y.jpg", scharr_y)

    # 2. get stroke width and median the length
    swt_img, rays = swt(canny, scharr_x, scharr_y, 1)
    cv2.imwrite("swt.png", swt_img)

    swt_img, rays = swt(canny, scharr_x, scharr_y, -1)
    cv2.imwrite("swt_rev.png", swt_img)

    # 3. get connected chains
    connect_img = connect_chains(swt_img, rays)
    cv2.imwrite("connect.png", connect_img)

    # 4. find the area with text in it.
    detected_img = generate(connect_img)
    cv2.imwrite("final.png", detected_img)


def main(filename):
    im = cv2.imread(filename)

    # return the detected image and the flag 
    img, flag = detect_text(im)


if __name__ == '__main__':
    filename = '/home/molly/detectText/6a60ed1e2a30360d290bdccf201c42f1.jpg'
    main(filename)
