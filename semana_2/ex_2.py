# Agora você criará uma função que "cola" uma imagem menor em uma imagem maior.

# Crie uma função chamada paste() que receberá, como parâmetro, uma imagem src(source), uma imagem dst(destiny) e uma posição x, y qualquer. Ela retornará a imagem modificada.

# A imagem dst será "colada" na posição x, y da imagem src. Confira o exemplo abaixo.

# newImg = paste(messiImg, ballImg, x, y)

from tools.image import Image, ImageEditor
from tools.utils import real_path

img = Image(real_path(__file__, 'semana_2/example.png'))
editor = ImageEditor(img)
ball = editor.crop(300, 430, 80, 200)
editor.paste(ball, 40, 418)
img.show()
