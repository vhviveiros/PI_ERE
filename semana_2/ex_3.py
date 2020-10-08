# %%Dessa vez, você deverá criar um algoritmo para saber qual imagem é mais "vermelha", "verde" ou "azul".

# Para isso, você deverá pegar uma imagem qualquer e extrair cada um dos canais dela, em separado. Assim, você irá gerar 3 novas imagens, uma contendo apenas o canal vermelho, outra contendo somente o verde e a última, azul.
from tools.image import Image, ImageEditor
from tools.utils import real_path

img = Image(real_path(__file__, 'example.png'))
print(img.greatest_rgb_channel())
