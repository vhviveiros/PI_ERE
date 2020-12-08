from tools.image import Image, ImageLimiarizator
from tools.utils import abs_path

image = Image(image_file=abs_path('semana_4/moedas.jpg'), target_size=(512, 512)).bgr2gray()
window = ImageLimiarizator(image, "Lim").with_adaptive_limiarization_controls().show()
