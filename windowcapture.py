import numpy as np
import win32gui, win32ui, win32con


class WindowCapture:
    w=0
    h=0
    hwnd = None

    def __init__(self,window_name=None,w=1920,h=1080):
        if window_name == None:
            self.hwnd = None
            self.w = w
            self.h = h
        else:
            self.hwnd = win32gui.FindWindow(None,window_name)
            if not self.hwnd:
                raise Exception('Window not found: {}'.format(window_name))
            window_rect = win32gui.GetWindowRect(self.hwnd)
            self.w = window_rect[2]-window_rect[0]
            self.h = window_rect[3]-window_rect[1]
    



    def get_screenshot(self):

        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj,self.w,self.h)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0,0),(self.w,self.h),dcObj,(0,0),win32con.SRCCOPY)

        #dataBitMap.SaveBitmapFile(cDC, 'debug.bmp')
        signedIntsArray = dataBitMap.GetBitmapBits(True)
        img= np.fromstring(signedIntsArray,dtype='uint8')
        img.shape = (self.h,self.w,4)

        #release stuff to free space
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd,wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())

        #remove alpha chanel
        img = img[...,:3]
        img = np.ascontiguousarray(img)
        return img
    
    def list_window_names(self):
        def winEnumHandler(hwnd,ctx):
            if win32gui.IsWindowVisible(hwnd):
                print(hex(hwnd),win32gui.GetWindowText(hwnd))
        win32gui.EnumWindows(winEnumHandler,None)
    
