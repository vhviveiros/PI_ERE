from tools.video import Video
from tools.image import Image, ImageEditor
from tools.utils import real_path
import cv2
import numpy as np


def roi(frame):
    lower = np.array([146, 135, 175])
    higher = np.array([180, 255, 255])
    hsv_frame = Image(data=frame).bgr2hsv().data
    frame = Image(data=frame, target_size=(800, 600))
    frame_editor = ImageEditor(frame)
    mask = Image(data=cv2.inRange(hsv_frame, lower, higher), target_size=(800, 600))

    contours, _ = cv2.findContours(mask.data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        for c in contours:
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            if area > 1000:
                frame_editor.draw_square(x, y, w, h, [255, 255, 255], thickness=1)

    return [['frame', frame.data], ['mask', mask.data]]


video = Video(real_path(__file__, 'sasuke.mp4'))
video.apply_and_show(roi)