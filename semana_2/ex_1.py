# %% Dessa vez você vai criar uma função chamada crop(). Essa função receberá por parâmetro uma imagem, uma posição x, y e uma altura e largura. A função deverá retornar o pedaço recortado.

from tools.image import Image, ImageEditor
from tools.utils import real_path

img = Image(real_path(__file__, 'semana_2/example.png'))
editor = ImageEditor(img)
editor.crop(300, 430, 80, 200).show()
