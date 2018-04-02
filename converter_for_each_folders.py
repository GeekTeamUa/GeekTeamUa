# Before start deleted folder 'errors' in folder "Д_1"

from PIL import Image
import numpy
import json
import glob
import os

data = json.load(open("config.json"))

data_folders = glob.glob(data["dataset"][0]+'*')


for folder in data_folders:
    all_images_in_folder = glob.glob(data["dataset"][0]+str(os.path.basename(folder))+"/*")
    for img in all_images_in_folder:
        im = Image.open(img).convert('L')
        (width, height) = im.size
        greyscale_image = list(im.getdata())
        greyscale_image = numpy.array(greyscale_image)
        greyscale_image = greyscale_image.reshape((height, width))

    # Save in folder(takes longer time)
    # numpy.save(os.path.join(save_path, str(os.path.basename(folder))), greyscale_image)  
   
    # Save in current directory(works faster because of unknown reasons)
    numpy.save(str(os.path.basename(folder)), greyscale_image)    


# numpy.save("array", img)
