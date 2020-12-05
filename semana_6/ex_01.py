from tools.image import Image, ImageEditor
from tools.utils import abs_path

img = Image(abs_path('avengers.jpeg'), target_size=(512, 512))
print(img.count_colors())
editor = ImageEditor(img)
editor.compute_k_means(8)
print(editor.img.count_colors())
editor.img.show()