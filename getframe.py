import win32gui
import win32ui
from ctypes import windll
from PIL import Image
import numpy

def getFrame():
    hwnd = win32gui.FindWindow(None, 'Super Mario World (U) - Snes9x 1.60')

    windll.user32.SetProcessDPIAware()
    left, top, right, bot = win32gui.GetClientRect(hwnd)
    w = right - left
    h = bot - top

    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()

    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)

    saveDC.SelectObject(saveBitMap)

    result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 3)

    bmpinfo = saveBitMap.GetInfo()
    bmpstr = saveBitMap.GetBitmapBits(True)

    im = Image.frombuffer(
        'RGB',
        (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
        bmpstr, 'raw', 'BGRX', 0, 1)

    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)

    img_array = numpy.array(im)
    img_array = img_array[:,:,0]
    img_array = img_array[60:-10]
    img_array = img_array/255.0

    return img_array

if __name__ == '__main__':
    while(1):
        import matplotlib.pyplot as plt
        frame = getFrame()
        plt.imshow(frame)
        plt.show()
    