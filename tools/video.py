import cv2.cv2 as cv2
from tools.image import Image
import numpy as np


class Video:
    def __init__(self, path):
        self.path = path
        self.__read_file()

    def __read_file(self):
        self.data = cv2.VideoCapture(self.path)

    def apply_and_show(self, function):
        while self.data.isOpened():
            _, frame = self.data.read()

            frame = function(frame)

            for f in frame:
                cv2.imshow(f[0], f[1])

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.data.release()
        cv2.destroyAllWindows()


class Recorder:
    def __init__(self, file, target_size=(512, 512)):
        fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
        self.out = cv2.VideoWriter(file, fourcc, 20.0, target_size)

    def erosion(self, img: Image, k_size, iterations):
        def generator():
            erosion = img.data
            kernel = np.ones((k_size, k_size), np.uint8)

            for i in range(0, iterations):
                erosion = cv2.erode(erosion, kernel, iterations=1)
                yield erosion
            self.release()

        self.write(generator())

    def dilation(self, img: Image, k_size, iterations):
        def generator():
            dilation = img.data
            kernel = np.ones((k_size, k_size), np.uint8)

            for i in range(0, iterations):
                dilation = cv2.dilate(dilation, kernel, iterations=1)
                yield dilation
            self.release()

        self.write(generator())

    def write(self, generator):
        for frame in generator:
            self.out.write(frame)

    def release(self):
        self.out.release()
