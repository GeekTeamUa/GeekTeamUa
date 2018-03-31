from PIL import Image
import glob
import json
import numpy


data = json.load(open("config.json"))


list_of_folders = glob.glob(data["dataset"][0]+"*")

label_list = []

for target in range(len(list_of_folders)):
    for path_to_img in glob.glob(list_of_folders[target]+"/*"):
        im = Image.open(path_to_img).convert('L')
        (width, height) = im.size
        greyscale_image = list(im.getdata())
        greyscale_image = numpy.array(greyscale_image)
        greyscale_image = greyscale_image.reshape((height, width))
        label_list.append((greyscale_image, path_to_img, target))
    target += 1
















