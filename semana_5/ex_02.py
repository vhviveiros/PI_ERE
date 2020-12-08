from tools.image import Image, HistogramPlot
from tools.utils import abs_path
import cv2.cv2 as cv2

img_gray = Image(abs_path('semana_5/equalize.png'), target_size=(512, 512), flag=cv2.IMREAD_GRAYSCALE)
img_gray.show('Original')
img_gray.equalize().show('Equalized')
