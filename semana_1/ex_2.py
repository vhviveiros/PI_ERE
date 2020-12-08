# %% Pegue uma foto que está no seu computador, faça upload dela e, por fim, faça ela ser carregada e exibida na célula abaixo.
from tools.image import Image
from tools.utils import real_path

img = Image(real_path(__file__, 'semana_1/example.jpg'))
img.show()
