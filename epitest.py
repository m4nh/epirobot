import os
import cv2
import glob, os
import numpy as np


def DOG(image):

    gray_blur  = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray_blur, 15).astype(float)

    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    # Filter the blurred grayscale images using filter2D

    filtered_blurred_x = cv2.filter2D(gray_blur, -1, sobel_x)
    filtered_blurred_y = cv2.filter2D(gray_blur, -1, sobel_y)


    # Compute the orientation of the image
    orien = cv2.phase(
        np.array(filtered_blurred_x, np.float32),
        np.array(filtered_blurred_y, dtype=np.float32),
        angleInDegrees=False
    )
    orien = orien + np.pi * 0.5
    return ((orien /(1.0*np.pi))*255.).astype(np.uint8)


def epiRow(y, max_stack = 32):
    stack = None

    for index, img in enumerate(images):
        if index >= max_stack:
            break
        row = img[y, :].reshape((1, img.shape[1], 3))
        if stack is None:
            stack = row
        else:
            stack = np.vstack((stack, row))
    stack = cv2.flip(stack, 0)
    cv2.imshow("epi", stack)

    orientation = DOG(stack)
    print("MIN MAX", np.min(orientation), np.max(orientation))
    cv2.imshow("epi_or", cv2.applyColorMap(orientation,cv2.COLORMAP_PARULA))
    print(stack.shape)

def epiSlice(y, max_stack = 8):
    stack = None

    for index, img in enumerate(images):
        if index >= max_stack:
            break
        row = img[y, :].reshape((1, img.shape[1], 3))
        if stack is None:
            stack = row
        else:
            stack = np.vstack((stack, row))
    stack = cv2.flip(stack, 0)
    orientation = DOG(stack)
    return orientation[-2,:].reshape((1,-1))


def mouseMove(event, x, y, flags, param):
    """ COMMENT """
    global output_image

    if event == cv2.EVENT_MOUSEMOVE:
        out = output_image.copy()
        cv2.line(out, (0, y), (630, y), (255, 0, 255), 1)
        cv2.imshow("image", out)

    if event == cv2.EVENT_LBUTTONDOWN:
        epiRow(y)
    if event == cv2.EVENT_RBUTTONDOWN:

        void = np.zeros(output_image.shape[:2], np.uint8)
        for i in range(output_image.shape[0]):
            void[i,:] = epiSlice(i)
        cv2.imshow("slice", cv2.applyColorMap(void, cv2.COLORMAP_JET))




#folder = '/private/tmp/epirobot2/'
#folder = '/Users/daniele/Downloads/benchmark/training/sideboard'
folder = '/private/tmp/dataset_train/frame_d3d22c685dc449978c9055e3efc221d2'
images = list(sorted(glob.glob(os.path.join(folder, '*.jpg'))))

images = list(map(cv2.imread, images))



cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("image", mouseMove)

index = 0
output_image = images[index].copy()
cv2.imshow("image", output_image)
v = 5
while True:

    c = cv2.waitKey(0)
    if c == ord('d'):
        index = (index + v) % len(images)
        output_image = images[index].copy()
        cv2.imshow("image", output_image)
    if c == ord('a'):
        index = (index - v) % len(images)
        output_image = images[index].copy()
        cv2.imshow("image", output_image)
    if c == ord('q'):
        cv2.destroyAllWindows()
        break
