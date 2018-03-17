from PIL import Image
import numpy
import glob
import os

save_path = '/home/dmytro/PycharmProjects/script_convert_jpg_to_numpy/npy_data'

data_folders = glob.glob('/home/dmytro/PycharmProjects/dataset/letters/*')


for folder in data_folders:
    all_images_in_folder = glob.glob('/home/dmytro/PycharmProjects/dataset/letters/'+str(os.path.basename(folder))+"/*")
    for img in all_images_in_folder:
        im = Image.open(img).convert('L')  # Make monochrome image.I think that is spare step, but it isn't working withot this convert
        (width, height) = im.size
        greyscale_image = list(im.getdata())
        greyscale_image = numpy.array(greyscale_image)
        greyscale_image = greyscale_image.reshape((height, width))

    # Save in folder(takes longer time)
    # numpy.save(os.path.join(save_path, str(os.path.basename(folder))), greyscale_image)  
   
    # Save in current directory(works faster because of unknown reasons)
    numpy.save(str(os.path.basename(folder)), greyscale_image)    


# numpy.save("array", img)
