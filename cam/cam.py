import cv2



class Camera():
    '''
    这个模块被用来调用摄像头。包括如下接口：
    .get_frame(name='image'):
        name参数传入窗口名字，将建立新窗口，播放摄像头画面。
        按A截图。
    '''
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

    def get_frame(self,name):
        cv2.destroyAllWindows()
        cv2.namedWindow(name)
        while True:
            ret, frame = self.cap.read()
            cv2.imshow('image',frame)
            key = cv2.waitKey(10)
            if key in [ord('a'),ord('A')]:
                cv2.destroyAllWindows()
                return frame


