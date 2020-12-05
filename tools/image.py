from __future__ import annotations

import cv2.cv2 as cv2
import numpy as np
import os
from .utils import abs_path
from glob import glob
import matplotlib
import matplotlib.pyplot as plt
import mahotas as mt
import time
from math import sqrt, pow
from tkinter import *
from PIL import Image as ImagePil
from PIL import ImageTk
import tkinter.filedialog as filedialog

matplotlib.use('TkAgg')


class Image:
    def __init__(self, image_file=None, data=None, divide=False, reshape=False, target_size=None, flag=None):
        self.target_size = target_size
        self.divide = divide
        self.reshape = reshape

        if image_file is not None:
            self.image_file = image_file
            self.data = self.__load_file(flag)

        if data is not None:
            self.data = data

        if self.data is not None and self.target_size is not None:
            self.data = cv2.resize(self.data, self.target_size)

    def __load_file(self, flag):
        img = cv2.imread(self.image_file, flag)
        if self.divide:
            img = img / 255
        if self.reshape:
            img = np.reshape(img, img.shape + (1,))
            img = np.reshape(img, (1,) + img.shape)
        return img

    def apply(self, function, *args):
        return function(self.data, *args)

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

    def haralick(self):
        # calculate haralick texture features for 4 types of adjacency
        textures = mt.features.haralick(self.data)

        # take the mean of it and return it
        ht_mean = textures.mean(axis=0)
        return ht_mean

    def invert(self):
        self.data = cv2.bitwise_not(self.data)
        return self

    def equalize(self):
        self.data = cv2.equalizeHist(self.data)
        return self

    def calc_histogram(self):
        """
        Only works with BGR
        """
        copy = Image(data=self.data)
        copy.bgr2hsv()
        hist = cv2.calcHist([copy.data], [0, 1], None, [50, 60], [0, 180, 0, 256], accumulate=False)
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist

    def greatest_rgb_channel(self):
        b, g, r = cv2.split(self.data)
        sum = {np.sum(r): "Vermelho", np.sum(g): "Verde", np.sum(b): "Azul"}
        return sum[np.amax(list(sum.keys()))]

    def get_channel(self, channel):
        return self.data[:, :, channel]

    def bgr2rgb(self):
        self.data = cv2.cvtColor(self.data, cv2.COLOR_BGR2RGB)
        return self

    def rgb2bgr(self):
        self.data = cv2.cvtColor(self.data, cv2.COLOR_RGB2BGR)
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

    def is_gray_scale(self):
        return len(self.data.shape) < 3

    def count_colors(self):
        uniques = np.unique(self.data.reshape(-1, self.data.shape[-1]), axis=0)
        return len(uniques)


class ImageGenerator:
    @staticmethod
    def generate_from(path, files=None, divide=False, reshape=False, only_data=False, target_size=(512, 512)):
        if files:
            image_files = [path + '/' + file for file in files]
        else:
            image_files = glob(path + "/*g")

        for image_file in image_files:
            if only_data:
                yield Image(image_file, divide=divide, reshape=reshape, target_size=target_size).data
            else:
                yield Image(image_file, divide=divide, reshape=reshape, target_size=target_size)


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

    def idk_a_proper_name_for_this(self, img: Image, color):
        for i in range(0, img.data.shape[0]):
            for j in range(0, img.data.shape[1]):
                if all(c == color for c in self.img.data[i][j]):
                    self.img.data[i][j] = img.data[i][j]
        return self

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

    def compute_k_means(self, k):
        reshape = self.img.data.reshape(-1, 3)
        reshape = np.float32(reshape)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.1)

        _, labels, centroids = cv2.kmeans(reshape, k, None, criteria, 40, cv2.KMEANS_RANDOM_CENTERS)

        centroids = np.uint8(centroids)
        img = centroids[labels.flatten()]
        img = img.reshape(self.img.data.shape)
        self.img.data = img


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
        cv2.namedWindow(title)

    def with_limiarization_controls(self, opt=cv2.THRESH_BINARY):
        self.img_limiar = self.img

        def on_change_lim(min_lim, max_lim):
            self.min_lim = min_lim
            self.max_lim = max_lim
            _, img_limiar = cv2.threshold(self.img.data, min_lim, max_lim, opt)
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


class ImageFilter:
    def __init__(self, img: Image):
        self.img = img
        self.filtered = img

    def __apply(self, f, *args):
        return self.img.apply(f, *args)

    def averaging(self, k_size=5) -> ImageFilter:
        self.filtered = Image(data=self.__apply(cv2.blur, (k_size, k_size)))
        return self

    def gaussian(self, k_size=5, sigmax=0, sigmay=None) -> ImageFilter:
        self.filtered = Image(data=self.__apply(cv2.GaussianBlur, (k_size, k_size), sigmax, None, sigmay))
        return self

    def median(self, k_size=5) -> ImageFilter:
        self.filtered = Image(data=self.__apply(cv2.medianBlur, k_size))
        return self

    def sobel(self, dx, dy, k_size=5, ddepth=cv2.CV_8U) -> ImageFilter:
        self.filtered = Image(data=self.__apply(cv2.Sobel, ddepth, dx, dy, None, k_size))
        return self

    def laplacian(self, ddepth=cv2.CV_64F, k_size=None) -> ImageFilter:
        self.filtered = Image(data=self.__apply(cv2.Laplacian, ddepth, None, k_size))
        return self


class Subplot:
    def __init__(self, images, ncols, nrows):
        assert ncols * nrows == len(images)
        self.ncols = ncols
        self.nrows = nrows
        self.images = images

    def show(self):
        for i in range(0, len(self.images)):
            img = self.images[i]
            if img:
                plt.subplot(self.nrows, self.ncols, i + 1), plt.imshow(img.data, 'gray')
        plt.show()


class HistogramPlot:
    def __init__(self, img: Image):
        self.img = img
        self.__calc()

    def __calc(self):
        color = 'b' if self.img.is_gray_scale() else ('b', 'g', 'r')
        plt.figure()
        for i, col in enumerate(color):
            hist = np.squeeze(cv2.calcHist([self.img.data], [i], None, [256], [0, 256]))
            plt.plot(hist, color=col)

    def save(self, save_folder=''):
        filename, fileext = self.img.get_file_dir()
        result_file = abs_path(
            save_folder, "%s_histogram%s" % (filename, '.png'))
        plt.savefig(result_file)
        plt.close()
        return self

    def show(self):
        plt.show()


class ImageSearch:
    def __init__(self, generator: ImageGenerator.generate_from):
        self.base = list(generator)

    def search(self, img: Image):
        result = ()

        for i in self.base:
            corr = cv2.compareHist(i.calc_histogram(), img.calc_histogram(), cv2.HISTCMP_CORREL)
            chi = cv2.compareHist(i.calc_histogram(), img.calc_histogram(), cv2.HISTCMP_CHISQR)
            bhatt = cv2.compareHist(i.calc_histogram(), img.calc_histogram(), cv2.HISTCMP_BHATTACHARYYA)

            file_dir = i.get_file_dir()
            result += (file_dir[0] + file_dir[1], sqrt(pow(corr, 2) + pow(chi, 2) + pow(bhatt, 2)))

        return result


class GrabCutGUI(Frame):
    def __init__(self):
        # invoca o construtor da classe pai Frame
        Frame.__init__(self, Tk())

        # inicializar a interface gráfica
        self.iniciaUI()

    def iniciaUI(self):
        # preparando a janela
        self.master.title("Janela da Imagem Segmentada")
        self.pack()

        # computa ações de mouse
        self.computaAcoesDoMouse()

        # carregando a imagem do disco
        self.imagem = self.carregaImagemASerExibida()

        self.master.geometry(str(self.imagem.width()) + "x" + str(self.imagem.height()))

        # criar um canvas que receberá a imagem
        self.canvas = Canvas(self.master, width=self.imagem.width(), height=self.imagem.height(), cursor="cross")

        # desenhar a imagem que carreguei no canvas
        self.canvas.create_image(0, 0, anchor=NW, image=self.imagem)
        self.canvas.image = self.imagem  # pra imagem não ser removida pelo garbage collector

        # posiciona todos os elementos no canvas
        self.canvas.pack()

    def computaAcoesDoMouse(self):
        self.startX = None
        self.startY = None
        self.rect = None
        self.rectangleReady = None

        self.master.bind("<ButtonPress-1>", self.callbackBotaoPressionado)
        self.master.bind("<B1-Motion>", self.callbackBotaoPressionadoEmMovimento)
        self.master.bind("<ButtonRelease-1>", self.callbackBotaoSolto)

    def callbackBotaoSolto(self, event):
        if self.rectangleReady:
            # criar uma nova janela
            windowGrabcut = Toplevel(self.master)
            windowGrabcut.wm_title("Segmentation")
            windowGrabcut.minsize(width=self.imagem.width(), height=self.imagem.height())

            # criar canvas pra essa nova janela
            canvasGrabcut = Canvas(windowGrabcut, width=self.imagem.width(), height=self.imagem.height())
            canvasGrabcut.pack()

            # aplicar grabcut na imagem
            mask = np.zeros(self.imagemOpenCV.shape[:2], np.uint8)
            print(mask.shape)
            rectGcut = (int(self.startX), int(self.startY), int(event.x - self.startX), int(event.y - self.startY))
            fundoModel = np.zeros((1, 65), np.float64)
            objModel = np.zeros((1, 65), np.float64)

            # invocar grabcut
            cv2.grabCut(self.imagemOpenCV, mask, rectGcut, fundoModel, objModel, 5, cv2.GC_INIT_WITH_RECT)

            # preparando imagem final
            maskFinal = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            imgFinal = self.imagemOpenCV * maskFinal[:, :, np.newaxis]

            blurred = self.__blurred_background()

            for x in range(0, self.imagemOpenCV.shape[1]):
                for y in range(0, self.imagemOpenCV.shape[0]):
                    if maskFinal[y][x] == 0:
                        imgFinal[y][x] = blurred[y][x]

            # converter de volta do opencv pra Tkinter
            imgFinal = cv2.cvtColor(imgFinal, cv2.COLOR_BGR2RGB)
            imgFinal = ImagePil.fromarray(imgFinal)
            imgFinal = ImageTk.PhotoImage(imgFinal)

            # inserir a imagem segmentada no canvas
            canvasGrabcut.create_image(0, 0, anchor=NW, image=imgFinal)
            canvasGrabcut.image = imgFinal

    def __blurred_background(self):
        base_filter = ImageFilter(Image(data=self.imagemOpenCV))
        return base_filter.gaussian(11, 5).filtered.data

    def callbackBotaoPressionadoEmMovimento(self, event):
        # novas posicoes de x e y
        currentX = self.canvas.canvasx(event.x)
        currentY = self.canvas.canvasy(event.y)

        # atualiza o retângulo a ser desenhado
        self.canvas.coords(self.rect, self.startX, self.startY, currentX, currentY)

        # verifica se existe retângulo desenhado
        self.rectangleReady = True

    def callbackBotaoPressionado(self, event):
        # convertendo o x do frame, pro x do canvas e copiando isso em startX
        self.startX = self.canvas.canvasx(event.x)
        self.startY = self.canvas.canvasy(event.y)

        if not self.rect:
            self.rect = self.canvas.create_rectangle(0, 0, 0, 0, outline="blue")

    def carregaImagemASerExibida(self):
        caminhoDaImagem = filedialog.askopenfilename()

        # se a imagem existir, entra no if
        if caminhoDaImagem:
            self.imagemOpenCV = cv2.imread(caminhoDaImagem)

            # converte de opencv para o formato PhotoImage
            image = cv2.cvtColor(self.imagemOpenCV, cv2.COLOR_BGR2RGB)

            # converte de OpenCV pra PIL
            image = ImagePil.fromarray(image)

            # converte de PIL pra PhotoImage
            image = ImageTk.PhotoImage(image)

            return image

    def show(self):
        self.mainloop()
        return self
