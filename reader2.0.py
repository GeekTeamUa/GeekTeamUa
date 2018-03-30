import glob
import json


data = json.load(open("config.json"))


list_of_folders = glob.glob(data["dataset"][0]+"*")

label_list = []

for index in range(len(list_of_folders)):
    for image in glob.glob(list_of_folders[index]+"/*"):
        label_list.append((image, index))
    index += 1














