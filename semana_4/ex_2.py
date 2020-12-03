from tools.image import Image, ImageFilter, Subplot
from tools.utils import abs_path

## Img 1
img_1 = Image(image_file=abs_path('filter_1.png'), target_size=(512, 512))
## Averaging
averaging_filter = ImageFilter(img_1).averaging().filtered
## Gaussian
gaussian_filter = ImageFilter(img_1).gaussian(k_size=11).filtered

## Img 2
img_2 = Image(image_file=abs_path('filter_2.png'), target_size=(512, 512))
## Median
median = ImageFilter(img_2).median(11).filtered

## Img 3
img_3 = Image(image_file=abs_path('filter_3.png'), flag=0)
## Sobel
sobel = ImageFilter(img_3).sobel(0, 2, k_size=5).filtered
## Laplacian
laplacian = ImageFilter(img_3).laplacian().filtered

subplot = Subplot([img_1, img_2, img_3,
                   averaging_filter, median, sobel,
                   gaussian_filter, None, laplacian], 3, 3).show()
