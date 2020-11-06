import cv2

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
