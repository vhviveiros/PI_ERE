# Existe um jogo na Internet chamado "Brasonic", que consiste numa versão tupiniquim do ouriço mais popular dos jogos, Sonic. Basicamente, ele é uma versão do personagem trocando suas cores pelo verde e amarelo da bandeira nacional.

# Nesse exercício, você deverá fazer um algoritmo que "recebe" uma imagem do Sonic e "troca" suas cores para o verde e amarelo. Para tal, modifique os pixels e salve a nova imagem em disco!
from tools.image import Image, ImageEditor
from tools.utils import real_path


def where(pixel):
    if pixel[0] > 170:
        return True
    if pixel[1] <= 110 and pixel[2] <= 110:
        return True
    return False


img = Image(real_path(__file__, 'sonic.png'))  # BGR
editor = ImageEditor(img)
editor.swap_channels(0, 1, where)  # GBR
img.show()
