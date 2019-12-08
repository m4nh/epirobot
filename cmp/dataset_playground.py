import cv2
import numpy as np
import glob
import os

#folder = '/Users/daniele/Downloads/CMP-BATCH-0/buoni/'
folder = '/Users/daniele/Downloads/CMP-BATCH-0/crimpatura'

images = sorted(glob.glob(os.path.join(folder, '*')))
bacth_size = 8

for i in range(0, len(images), bacth_size):

    batch_names = images[i:i + bacth_size]

    batch = []
    for n in batch_names:
        img = cv2.imread(n, 0).astype(np.float32) / 255.
        batch.append(img)

    batch = np.array(batch)

    out = np.mean(batch, axis=0)

    print("OUT", out.shape, np.min(out), np.max(out))
    cv2.imshow("image", (out * 255.).astype(np.uint8))
    cv2.waitKey(0)

    # batch = np.array(list(map(cv2.imread, batch)))
    #
    # print(batch.shape)
    #
    # for j in range(bacth_size):
    #     cv2.imshow("image", batch[j,::])
    #     cv2.waitKey(0)
