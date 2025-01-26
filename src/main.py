
import cv2 as cv
import os


img_size = 512, 512
count = 0
for pic_path in os.listdir(r"C:\Users\26922\Desktop\flowers"):
    pic = cv.imread(fr"C:\Users\26922\Desktop\flowers\{pic_path}")
    x, y = pic.shape[0], pic.shape[1]
    if max(x / y, y / x) + 0.12 > 1:
        pic_ = cv.resize(pic[0 : min(x, y) - 1, 0 : min(x, y) - 1, :], img_size)
        cv.imwrite(fr"./data/{count}.png", pic_)
    count += 1
    if count >= 200:
        break