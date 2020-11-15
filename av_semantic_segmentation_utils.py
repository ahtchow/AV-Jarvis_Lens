import cv2
import os
import PIL
from PIL import Image, ImageOps, ImageFile
import numpy as np

AV_SS_V = "1.0"
ImageFile.LOAD_TRUNCATED_IMAGES = True

def create_resized_data_bank(dataset_dir, resize_shape=(256, 256), databank_name='DATABANK_'):
    
    # Create a new folder for databank with resized imgs
    parent_folder = databank_name + "S_" + str(resize_shape[0]) + "_V_" + AV_SS_V
    images_folder = parent_folder + "/images"
    labels_folder = parent_folder + "/labels"

    for f in [parent_folder, images_folder, labels_folder]:
        try:
            os.makedirs(f)
        except OSError:
            print('Folder', f, 'already exists')

    # File system - Subdirectories from Mapillary Datasete
    # Merging all the images to make one large Databank, define data split ratios later
    folders = ['training/images', 'training/labels', 'validation/images', 'validation/labels', 'testing/images']
    for folder in folders:
        data_folder = os.path.join(dataset_dir, folder)
        files = sorted(os.listdir(data_folder))
        print(files)    
        # Take images and move to bank
        for file in files:
            file_path = os.path.join(data_folder, file)
            image = Image.open(file_path).resize(resize_shape)
            if "images" in folder:
                image.save(os.path.join(images_folder, file))
            if "labels" in folder:
                image.save(os.path.join(labels_folder, file))


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
