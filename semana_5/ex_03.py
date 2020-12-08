from tools.image import Image, ImageGenerator, ImageSearch
from tools.utils import abs_path

search = ImageSearch(ImageGenerator.generate_from(abs_path('semana_5/')))
pik = ImageGenerator.generate_from(abs_path('semana_5/'), files=['pik.png', 'pik2.png', 'pik3.png'])

for i in pik:
    file_dir = i.get_file_dir()
    print('\n', file_dir[0] + file_dir[1], '\t->\t', search.search(i), '\n')
