from PIL import Image
import glob
import json
import numpy


def read_data():

    data = json.load(open("config.json"))   # Open config with path

    list_of_folders = glob.glob(data["UAMnist_dataset"][0]+"*")     # list of path of folders (ex. /home/.../letters/A_1)

    dataset = []

    for target in range(len(list_of_folders)):
        for path_to_img in glob.glob(list_of_folders[target]+"/*"):     #
            im = Image.open(path_to_img).convert('L')
	    im5 = im.resize((28, 28), Image.ANTIALIAS)  # Resize our images to 28x28
            (width, height) = im5.size
            greyscale_image = list(im5.getdata())
            greyscale_image = numpy.array(greyscale_image)
            greyscale_image = greyscale_image.reshape((height, width))
            dataset.append((greyscale_image, path_to_img, target))
        target += 1

    numpy.save(data["features_path"], dataset)

    return dataset



