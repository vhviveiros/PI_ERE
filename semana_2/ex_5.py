# Existe um jogo na Internet chamado "Brasonic", que consiste numa versão tupiniquim do ouriço mais popular dos jogos, Sonic. Basicamente, ele é uma versão do personagem trocando suas cores pelo verde e amarelo da bandeira nacional.

# Nesse exercício, você deverá fazer um algoritmo que "recebe" uma imagem do Sonic e "troca" suas cores para o verde e amarelo. Para tal, modifique os pixels e salve a nova imagem em disco!
from tools.image import Image, ImageEditor
from tools.utils import real_path
import cv2

bozo = Image(real_path(__file__, 'bozo.png'))
l_h, l_w, _ = bozo.shape()

lula = Image(real_path(__file__, 'lula.png')).resize((l_h, l_w))


def where(pixel):
    return all(i >= 170 for i in pixel)


ImageEditor(lula).remove_around(150, 30, 200, 350).remove_where(where)

result = Image(data=cv2.addWeighted(lula.data, 0.3, bozo.data, 0.5, 50))
result.show()
