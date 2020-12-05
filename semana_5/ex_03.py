from tools.image import Image, ImageGenerator, ImageSearch
from tools.utils import abs_path

pik_imgs = [1, 2, 6]
search = ImageSearch(ImageGenerator.generate_from(abs_path('')))
pik = ImageGenerator.generate_from(abs_path(''), files=['pik.png', 'pik2.png', 'pik3.png'])

for i in pik:
    file_dir = i.get_file_dir()
    print('\n', file_dir[0] + file_dir[1], '\t->\t', search.search(i), '\n')
