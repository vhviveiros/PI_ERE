import cv2.cv2 as cv2
import numpy as np
import os
from .utils import abs_path
from glob import glob
from matplotlib import pyplot as plt
import mahotas as mt
import time


class Image:
    def __init__(self, image_file=None, data=None, divide=False, reshape=False, target_size=None):
        self.target_size = target_size
        self.divide = divide
        self.reshape = reshape

        if image_file is not None:
            self.image_file = image_file
            self.data = self.__load_file()

        if data is not None:
            self.data = data

        if self.data is not None and self.target_size is not None:
            self.data = cv2.resize(self.data, self.target_size)

    def __load_file(self, flag=None):
        img = cv2.imread(self.image_file, flag)
        if self.divide:
            img = img / 255
        if self.reshape:
            img = np.reshape(img, img.shape + (1,))
            img = np.reshape(img, (1,) + img.shape)
        return img

    def show(self, text='image'):
        cv2.imshow(text, self.data)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return self

    def get_file_dir(self):
        return os.path.splitext(os.path.basename(self.image_file))

    def save_to(self, path_dir):
        filename, fileext = self.get_file_dir()
        result_file = abs_path(
            path_dir, "%s_processed%s" % (filename, fileext))
        cv2.imwrite(result_file, self.data)

    def shape(self):
        return self.data.shape

    def resize(self, size):
        cv2.resize(self.data, size)
        return self

    def hist(self):
        result = np.squeeze(cv2.calcHist(
            [self.data], [0], None, [255], [1, 256]))
        result = np.asarray(result, dtype='int32')
        return result

    def save_hist(self, save_folder=''):
        plt.figure()
        histg = cv2.calcHist([self.data], [0], None, [254], [
            1, 255])  # calculating histogram
        plt.plot(histg)
        filename, fileext = self.get_file_dir()
        result_file = abs_path(
            save_folder, "%s_histogram%s" % (filename, '.png'))
        plt.savefig(result_file)
        plt.close()
        return self

    def haralick(self):
        # calculate haralick texture features for 4 types of adjacency
        textures = mt.features.haralick(self.data)

        # take the mean of it and return it
        ht_mean = textures.mean(axis=0)
        return ht_mean

    def greatest_rgb_channel(self):
        b, g, r = cv2.split(self.data)
        sum = {np.sum(r): "Vermelho", np.sum(g): "Verde", np.sum(b): "Azul"}
        return sum[np.amax(list(sum.keys()))]

    def get_channel(self, channel):
        return self.data[:, :, channel]

    def bgr2rgb(self):
        self.data = cv2.cvtColor(self.data, cv2.COLOR_BGR2RGB)
        return self

    def bgr2hsv(self):
        self.data = cv2.cvtColor(self.data, cv2.COLOR_BGR2HSV)
        return self

    def rgb2hsv(self):
        self.data = cv2.cvtColor(self.data, cv2.COLOR_RGB2HSV)
        return self

    def hsv2rgb(self):
        self.data = cv2.cvtColor(self.data, cv2.COLOR_HSV2RGB)
        return self

    def hsv2bgr(self):
        self.data = cv2.cvtColor(self.data, cv2.COLOR_HSV2BGR)
        return self

    def bgr2gray(self):
        self.data = cv2.cvtColor(self.data, cv2.COLOR_BGR2GRAY)
        return self

    def gray2bgr(self):
        self.data = cv2.cvtColor(self.data, cv2.COLOR_GRAY2BGR)
        return self


class ImageGenerator:
    def generate_from(self, path, divide=False, reshape=False, only_data=False):
        image_files = glob(path + "/*g")
        for image_file in image_files:
            if only_data:
                yield Image(image_file, divide, reshape).data
            else:
                yield Image(image_file, divide, reshape)


class ImageSaver:
    def __init__(self, images):
        self.images = images

    def save_to(self, path_dir):
        for img in self.images:
            img.save_to(path_dir)


class ImageEditor:
    def __init__(self, img: Image):
        self.img = img

    def draw_square(self, x, y, w, h, color, thickness=-1):
        cv2.rectangle(self.img.data, (x, y),
                      (x + w, y + h), color, thickness=thickness)
        return self

    def crop(self, x, y, w, h):
        return Image(data=self.img.data[y:y + h, x:x + w].copy())

    def paste(self, img: Image, x, y):
        shape_h, shape_w, _ = img.data.shape
        self.img.data[y:y + shape_h, x:x + shape_w] = img.data
        return self

    def blend(self, img: Image, x, y, alpha):
        rows, cols, channels = img.data.shape
        overlay = cv2.addWeighted(self.img.data[y:y + rows, x:x + cols], 1 - alpha, img.data, alpha, 0)
        self.paste(Image(data=overlay), x, y)
        return self

    def swap_channels(self, channel1, channel2, where):
        c1 = self.img.get_channel(channel1).copy()
        c2 = self.img.get_channel(channel2).copy()

        if where is None:
            self.img.data[:, :, channel1] = c2
            self.img.data[:, :, channel2] = c1
        else:
            for i in range(0, len(self.img.data)):
                for j in range(0, len(self.img.data[i])):
                    if where(self.img.data[i][j]):
                        self.img.data[i][j][channel1] = c2[i][j]
                        self.img.data[i][j][channel2] = c1[i][j]
        return self

    def remove_channel(self, channel):
        self.img.data[:, :, channel] = 0
        return self

    def remove_around(self, x, y, w, h):
        selection = self.crop(x, y, w, h)
        self.img.data[:, :, :] = 0
        self.paste(selection, x, y)
        return self

    def remove_where(self, where):
        for i in range(0, len(self.img.data)):
            for j in range(0, len(self.img.data[i])):
                if where(self.img.data[i][j]):
                    self.img.data[i][j] = [0, 0, 0]


class ImageLimiarizator:
    min_lim = 0
    max_lim = 255

    max_adapt = 255
    block_size = 11
    block_sizes = [3, 7, 9, 11]
    c_value = 100

    img_limiar = None
    img_adaptive_limiar = None

    def __init__(self, img: Image, title=''):
        self.img = img
        self.title = title
        self.limiar = img
        cv2.namedWindow(title)

    def with_limiarization_controls(self, opt=cv2.THRESH_BINARY):
        self.img_limiar = self.img

        def on_change_lim(min_lim, max_lim):
            self.min_lim = min_lim
            self.max_lim = max_lim
            limiar, img_limiar = cv2.threshold(self.img.data, min_lim, max_lim, opt)
            self.limiar = Image(data=limiar)
            self.img_limiar = Image(data=img_limiar)
            time.sleep(0.05)  # Prevents crashing
            self.show()

        def on_change_min(lim):
            on_change_lim(lim, self.max_lim)

        def on_change_max(lim):
            on_change_lim(self.min_lim, lim)

        cv2.createTrackbar('Mínimo Limiarização', self.title, self.min_lim, self.max_lim, on_change_min)
        cv2.createTrackbar('Máximo Limiarização', self.title, self.max_lim, self.max_lim, on_change_max)
        return self

    def with_adaptive_limiarization_controls(self, method=cv2.ADAPTIVE_THRESH_MEAN_C, opt=cv2.THRESH_BINARY):
        self.img_adaptive_limiar = self.img

        def on_change(max_adapt, block_size, c):
            self.max_adapt = max_adapt
            adapt = cv2.adaptiveThreshold(self.img.data, self.max_adapt, method, opt, block_size, c)
            self.img_adaptive_limiar = Image(data=adapt)
            time.sleep(0.05)
            self.show()

        def on_change_block_size(pos):
            on_change(self.max_adapt, self.block_sizes[pos], self.c_value)

        def on_change_max_value(value):
            on_change(value, self.block_size, self.c_value)

        def on_change_c_value(c):
            on_change(self.max_adapt, self.block_size, c)

        cv2.createTrackbar('Tamanho de bloco', self.title, 0, 3, on_change_block_size)
        cv2.createTrackbar('Valor máximo', self.title, 0, self.max_adapt, on_change_max_value)
        cv2.createTrackbar('Valor constante', self.title, 0, self.c_value, on_change_c_value)

        return self

    def show(self):
        cv2.imshow("Original", self.img.data)
        if self.img_limiar:
            cv2.imshow(self.title, self.img_limiar.data)

        if self.img_adaptive_limiar:
            cv2.imshow(self.title, self.img_adaptive_limiar.data)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return self
