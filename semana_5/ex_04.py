from tools.video import Recorder
from tools.image import Image
from tools.utils import abs_path

Recorder(abs_path('erosion.avi')).erosion(Image(abs_path('j.png'), target_size=(512, 512)), 2, 50)
Recorder(abs_path('dilation.avi')).dilation(Image(abs_path('j.png'), target_size=(512, 512)), 5, 100)
