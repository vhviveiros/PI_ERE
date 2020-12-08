from tools.image import Image, HistogramPlot
from tools.utils import abs_path
import cv2.cv2 as cv2

img_gray = Image(abs_path('semana_5/pik.png'), target_size=(512, 512), flag=cv2.IMREAD_GRAYSCALE).show()
HistogramPlot(img_gray).show()
img = Image(abs_path('semana_5/Jureg.jpg'), target_size=(512, 512)).show()
HistogramPlot(img).show()
