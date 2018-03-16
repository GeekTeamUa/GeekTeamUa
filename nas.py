from PIL import Image
import numpy as np
import glob
path = '/home/viktor/PycharmProjects/wqq/A_2'
image_list = []
for infile in glob.glob("/home/viktor/PycharmProjects/wqq/A_2/*.jpg"):
    img = Image.open(infile)
    arr = np.array(img)
    image_list.append(arr)
print(image_list)

