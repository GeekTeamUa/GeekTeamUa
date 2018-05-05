from flask import Flask, render_template, request
from keras.models import model_from_json
from PIL import Image
import numpy as np
import keras
import os

app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/result', methods=['GET', 'POST'])
def result():
    select = request.form.get("letter");

    # Saving image into the picture directory

    img = request.files['img']
    file = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
    img.save(file)

    basewidth = 28
    baseheight = 28
    img = Image.open(file)

    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    hpercent = (baseheight / float(img.size[1]))
    wsize = int((float(img.size[0]) * float(hpercent)))
    img = img.resize((wsize, baseheight), Image.ANTIALIAS)
    img_array = np.array(img)
    img_array=img_array.reshape(1,28,28,1)
    os.remove(file)

    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model.h5")
    loaded_model.compile(loss=keras.losses.categorical_crossentropy,
                         optimizer=keras.optimizers.Adadelta(),
                         metrics=['accuracy'])
    res = loaded_model.predict_classes(img_array)

   # print (res)
    res_str=np.array_str(res)
    res_str=res_str.replace("[","")
    res_str = res_str.replace("]", "")
#    return render_template("index.html",img=res_str)
    if (select == "1"):
        return render_template("index.html", final_text = "test1",img=res_str)
    if (select == "2") :
        return render_template("index.html", final_text = "test2",img=res_str)
    if (select == "3") :
        return render_template("index.html", final_text = "test3",img=res_str)


# @app.route('/upload', methods=['POST'])
# def upload_file():
#     file = request.files['image']
#     f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#
#     file.save(f)
#     return render_template("index.html", final_text="test")


if __name__ == "__main__":
    app.run(debug=True)
