import cv2
import PIL
import numpy as np

class ImgPreprocessor:

    def __init__(self, img_paths):

        self.imgs = np.array([cv2.imread(file) for file in img_paths])
        self.size = int(len(img_paths))

    def pre_process(self, IMG_WIDTH, IMG_HEIGHT):

        for i in range(len(self.imgs)):
            # Convert to RGB
            self.imgs[i] = cv2.cvtColor(self.imgs[i], cv2.COLOR_BGR2RGB)

            # Convert to new dimension
            self.imgs[i] = cv2.resize(self.imgs[i],(IMG_WIDTH, IMG_HEIGHT))


    def gaussian_blur(self):

        for i in range(len(self.imgs)):
            # Smoothening
            self.imgs[i] = cv2.GaussianBlur(self.imgs[i],(5,5),0)
