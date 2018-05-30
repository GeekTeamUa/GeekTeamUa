from PIL import Image
import numpy as np
import keras
from keras.models import model_from_json

basewidth = 28
baseheight = 28

file = "/home/dmytro/Documents/testing_samples/digits/images.png"

img = Image.open(file).convert("RGBA")
wpercent = (basewidth / float(img.size[0]))
hsize = int((float(img.size[1]) * float(wpercent)))
img = img.resize((basewidth, hsize), Image.ANTIALIAS)
hpercent = (baseheight / float(img.size[1]))
wsize = int((float(img.size[0]) * float(hpercent)))
img = img.resize((28, 28), Image.ANTIALIAS)
arr = np.array(img)

arr_img = arr.reshape((-1, 28, 28, 1))
#print (arr.shape())

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
loaded_model.compile(loss=keras.losses.categorical_crossentropy,
                     optimizer=keras.optimizers.Adadelta(),
                     metrics=['accuracy'])
res = loaded_model.predict(arr_img)
print(res)