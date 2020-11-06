from tools.image import Image, ImageEditor
from tools.utils import real_path

base = Image(real_path(__file__, 'blend_base.png'), target_size=(512, 512))
water_mark = Image(real_path(__file__, 'Jureg.jpg'), target_size=(150, 150))

editor = ImageEditor(base)
editor.blend(water_mark, base.shape()[0] - 270, base.shape()[1] - 170, 0.3)

base.show('')