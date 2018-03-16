from PIL import Image
import numpy
import glob


filelist = glob.glob("/home/dmytro/PycharmProjects/dataset/letters/А_1/*")      # listw didn't see A_1 folder, so i changed en. "A" to ukrainian "A".Lol.


all_images_in_folder = [i for i in filelist]   # Create list of all images in folder


for img in all_images_in_folder:
    im = Image.open(img).convert('L')         # Make monochrome image
    (width, height) = im.size
    greyscale_map = list(im.getdata())
    greyscale_map = numpy.array(greyscale_map)
    greyscale_map = greyscale_map.reshape((height, width))

numpy.save("array_of_A1", greyscale_map)

print(greyscale_map)
