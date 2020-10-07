import cv2
import numpy as np
import os
from .utils import abs_path
from glob import glob
from matplotlib import pyplot as plt
import mahotas as mt


class Image:
    def __init__(self, image_file=None, data=None, divide=False, reshape=False):
        self.divide = divide
        self.reshape = reshape

        if image_file is not None:
            self.image_file = image_file
            self.data = self.__load_file()

        if data is not None:
            self.data = data

    def __load_file(self, target_size=(512, 512), flag=None):
        img = cv2.imread(self.image_file, flag)
        if self.divide:
            img = img / 255
        img = cv2.resize(img, target_size)
        if self.reshape:
            img = np.reshape(img, img.shape + (1,))
            img = np.reshape(img, (1,) + img.shape)
        return img

    def show(self, text='image'):
        cv2.imshow(text, self.data)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_file_dir(self):
        return os.path.splitext(os.path.basename(self.image_file))

    def save_to(self, path_dir):
        filename, fileext = self.get_file_dir()
        result_file = abs_path(
            path_dir, "%s_processed%s" % (filename, fileext))
        cv2.imwrite(result_file, self.data)

    def shape(self):
        return self.data.shape

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

    def haralick(self):
        # calculate haralick texture features for 4 types of adjacency
        textures = mt.features.haralick(self.data)

        # take the mean of it and return it
        ht_mean = textures.mean(axis=0)
        return ht_mean


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

    def crop(self, x, y, w, h):
        return Image(data=self.img.data[y:y+h, x:x+w])

    def paste(self, img: Image, x, y):
        shape = img.data.shape
        shape_h = shape[0]
        shape_w = shape[1]
        self.img.data[y:y+shape_h, x:x+shape_w] = img.data
