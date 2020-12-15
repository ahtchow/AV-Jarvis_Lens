import cv2
import os
import PIL
from PIL import Image, ImageOps, ImageFile
import numpy as np
from tqdm import tqdm
import imageio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as mcolors 

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
            pass

    # File system - Subdirectories from Mapillary Datasete
    # Merging all the images to make one large Databank, define data split ratios later
    folders = ['validation/images', 'validation/labels',
                'training/images', 'training/labels']
    dataset_dir = os.getcwd() + dataset_dir
    print(f"Generating Databank: Reshaping to {resize_shape} images...")
    for folder in tqdm(folders, position=0, leave=True):
        data_folder = os.path.join(dataset_dir, folder)
        files = sorted(os.listdir(data_folder))
        # Take images and move to bank
        print(f"Getting Images from: {data_folder}")
        for file in tqdm(files, position=0, leave=True):
            file_path = os.path.join(data_folder, file)
            image = Image.open(file_path).resize(resize_shape)
            if "images" in folder:
                image.save(os.path.join(images_folder, file))
            if "labels" in folder:
                image.save(os.path.join(labels_folder, file))


def print_image_stats(image_dir, label=False):
    
    pic = mpimg.imread(image_dir)
    itype = 'Image' if not label else 'Label'
    print(f'STATS ON {itype}')
    print('Type of the {} : '.format(itype) , type(pic)) 
    print('Shape of the {} : {}'.format(itype, pic.shape)) 
    print('{} Hight {}'.format(itype, pic.shape[0])) 
    print('{} Width {}'.format(itype, pic.shape[1])) 
    print('Dimension of {} {}\n'.format(itype, pic.ndim))


def color_table(colors, title, colors_sort = True, emptycols = 0): 
   
    # cell dimensions 
    width = 212
    height = 22
    swatch_width = 48
    margin = 12
    topmargin = 40
   
    # Sorting colors bbased on hue, saturation, 
    # value and name. 
    if colors_sort is True: 
        to_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))), 
                         name) 
                        for name, color in colors.items()) 
          
        names = [name for hsv, name in to_hsv] 
          
    else: 
        names = list(colors) 
   
    length_of_names = len(names) 
    length_cols = 4 - emptycols 
    length_rows = length_of_names // length_cols + int(length_of_names % length_cols > 0) 
   
    width2 = width * 4 + 2 * margin 
    height2 = height * length_rows + margin + topmargin 
    dpi = 72
   
    figure, axes = plt.subplots(figsize =(width2 / dpi, height2 / dpi), dpi = dpi) 
    figure.subplots_adjust(margin / width2, margin / height2, 
                        (width2-margin)/width2, (height2-topmargin)/height2) 
      
    axes.set_xlim(0, width * 4) 
    axes.set_ylim(height * (length_rows-0.5), -height / 2.) 
    axes.yaxis.set_visible(False) 
    axes.xaxis.set_visible(False) 
    axes.set_axis_off() 
    axes.set_title(title, fontsize = 24, loc ="left", pad = 10) 
   
    for i, name in enumerate(names): 
          
        rows = i % length_rows 
        cols = i // length_rows 
        y = rows * height 
   
        swatch_start_x = width * cols 
        swatch_end_x = width * cols + swatch_width 
        text_pos_x = width * cols + swatch_width + 7
   
        axes.text(text_pos_x, y, name, fontsize = 14, 
                horizontalalignment ='left', 
                verticalalignment ='center') 
   
        axes.hlines(y, swatch_start_x, swatch_end_x, 
                  color = colors[name], linewidth = 18) 
   
    return figure


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
