from PIL import ImageGrab
import cv2
import numpy as np

def interest_region(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, [255,255,255])
    masked = cv2.bitwise_and(img, mask)
    return masked

def process_img(img):
    vertices = np.array([[0, 640], [0, 200], [738, 200], [738, 640]])
    processed_img = interest_region(img, [vertices])
    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    return processed_img

def hook():
    screen = np.array(ImageGrab.grab(bbox=(0,0,738,640)))
    processed_image = process_img(screen)
    return processed_image

if __name__ == '__main__':
    while(1):
        import matplotlib.pyplot as plt
        screen = np.array(ImageGrab.grab(bbox=(0,0,738,640)))
        processed_image = process_img(screen)
        plt.imshow(processed_image)
        plt.show()
    