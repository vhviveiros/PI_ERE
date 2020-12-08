# %% Crie um algoritmo que receba 4 informações de um retângulo: posição x, posição y, altura e largura.

# A seguir, faça com que uma nova imagem seja salva em disco, com um quadrado pintado com a cor preta ocupando as informações do retângulo.

from tools.image import Image, ImageEditor
from tools.utils import real_path

img = Image(real_path(__file__, 'semana_1/example2.png'))
editor = ImageEditor(img)
editor.draw_square(20, 20, 80, 80, 0)
img.show()
img.save_to('.')
