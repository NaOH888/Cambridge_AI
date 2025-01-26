
import cv2 as cv
import os


img_size = 512, 512
count = 0
for pic_path in os.listdir(r"C:\Users\26922\Desktop\pic"):
    pic = cv.imread(fr"C:\Users\26922\Desktop\pic\{pic_path}")
    x, y = pic.shape[0], pic.shape[1]
    print(max(x / y, y / x))
    if max(x / y, y / x) + 0.15 > 1:
        pic_ = cv.resize(pic[0 : min(x, y) - 1, 0 : min(x, y) - 1, :], img_size)
        cv.imwrite(fr"./dataset(butterflies)/{count}.png", pic_)
    count += 1
    if count >= 200:
        break