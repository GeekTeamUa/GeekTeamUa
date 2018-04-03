from PIL import Image
import glob
import json
import numpy


def read_data():

    data = json.load(open("config.json"))   # Open config with path

    list_of_folders = glob.glob(data["dataset"][0]+"*")     # list of path of folders (ex. /home/.../letters/A_1)

    dataset = []

    for target in range(len(list_of_folders)):
        for path_to_img in glob.glob(list_of_folders[target]+"/*"):     #
            im = Image.open(path_to_img).convert('L')
            (width, height) = im.size
            greyscale_image = list(im.getdata())
            greyscale_image = numpy.array(greyscale_image)
            greyscale_image = greyscale_image.reshape((height, width))
            dataset.append((greyscale_image, path_to_img, target))
        target += 1

    return dataset



    # numpy.save(data["features_path"], dataset)


<<<<<<< HEAD
=======
list_of_folders = glob.glob(data["dataset"][0]+"*")     # list of path of folders (ex. /home/.../letters/A_1)
>>>>>>> 1388abe4e254da7c650180d7948c400e09cddc45


<<<<<<< HEAD
=======
for target in range(len(list_of_folders)):
    for path_to_img in glob.glob(list_of_folders[target]+"/*"):     #
        im = Image.open(path_to_img).convert('L')
        (width, height) = im.size
        greyscale_image = list(im.getdata())
        greyscale_image = numpy.array(greyscale_image)
        greyscale_image = greyscale_image.reshape((height, width))
        label_list.append((greyscale_image, path_to_img, target))
    target += 1
>>>>>>> 1388abe4e254da7c650180d7948c400e09cddc45


numpy.save(data["features_path"], label_list)





