from tools.image import Image, Histogram
from tools.utils import abs_path
import cv2.cv2 as cv2

img_gray = Image(abs_path('Jureg.jpg'), target_size=(512, 512), flag=cv2.IMREAD_GRAYSCALE)
img = Image(abs_path('Jureg.jpg'), target_size=(512, 512))

hist = Histogram(img_gray).show()
