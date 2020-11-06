import cv2
import numpy as np
import os
from .utils import abs_path
from glob import glob
from matplotlib import pyplot as plt
import mahotas as mt


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
            if self.target_size:
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
        return Image(data=self.img.data[y:y+h, x:x+w].copy())

    def paste(self, img: Image, x, y):
        shape_h, shape_w, _ = img.data.shape
        self.img.data[y:y+shape_h, x:x+shape_w] = img.data
        return self

    def blend(self, img: Image, x, y, alpha):
        rows, cols, channels = img.data.shape
        overlay = cv2.addWeighted(self.img.data[y:y + rows, x:x + cols], 1-alpha, img.data, alpha, 0)
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
