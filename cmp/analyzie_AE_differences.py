import cv2
import numpy as np
import glob
import os
import time

def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    cv2.imshow('labeled.png', labeled_img)

file = '/Users/daniele/Downloads/d4.png'

img = cv2.imread(file)

for x in range(0, img.shape[1], 260):
    crop = img[:, x:x + 260].astype(np.float32)
    print(crop.shape)
    crop = np.mean(crop, axis=2) #/ 3.0

    #crop = 255.*crop / np.max(crop)
    print(crop.shape, np.min(crop), np.max(crop))


    out = (crop).astype(np.uint8)
    out_original = out.copy()
    max = 0.5*np.max(out)
    while True:

        t1 = time.time()
        out_binary = np.zeros(out.shape,out.dtype)
        out_binary[out>0.2*max] = 255
        ret, labels = cv2.connectedComponents(out_binary)

        imshow_components(labels)
        t2 = time.time()
        print("Time:", t2-t1)
        cv2.imshow("image", out_binary)
        cv2.imshow("out", out_original)
        c = cv2.waitKey(0)
        if c == ord('q'):
            break
        kernel = np.ones((2,2), np.uint8)
        out = cv2.erode(out, kernel, iterations=1)
