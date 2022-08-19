from PIL import ImageGrab
import cv2
import numpy as np

def interest_region(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, [255,255,255])
    masked = cv2.bitwise_and(img, mask)
    return masked

def process_img(img):
    vertices = np.array([[0, 640], [0, 150], [738, 150], [738, 640]])
    processed_img = interest_region(img, [vertices])
    return processed_img

def hook():
    screen = np.array(ImageGrab.grab(bbox=(0,40,738,640)))
    processed_image = process_img(screen)
    return processed_image
