from flask import Flask, flash, redirect, render_template, request, url_for, send_from_directory
import os
import numpy as np
import joblib
import BusinessLayer.chatbotBL as CBL

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input

app = Flask(__name__)

app.config['SECRET_KEY'] = 'my secret key'

UPLOAD_FOLDER = 'uploads'

model = load_model('models/pneumonia_model.h5')
model2 = load_model('models/malaria_model111.h5')

colors = [
    "#F7464A", "#46BFBD", "#FDB45C", "#FEDCBA",
    "#ABCDEF", "#DDDDDD", "#ABCABC", "#4169E1",
    "#C71585", "#FF4500", "#FEDCBA", "#46BFBD"]

def api(full_path):
    data = image.load_img(full_path, target_size=(150, 150, 3)) 
    data = np.expand_dims(data, axis=0)
    data = data * 1.0 / 255
    predicted = model.predict(data)
    return predicted

def api2(full_path):
    data = image.load_img(full_path, target_size=(128, 128)) 
    x = image.img_to_array(data)
    #x = x/255
    x = np.expand_dims(x, axis=0)
    #x = preprocess_input(x)
    predicted = model2.predict(x)
    predicted = np.argmax(predicted)
    if predicted==0:
        prediction='Malaria PRESENT'
    else:
        prediction='Malaria ABSENT'
    return prediction


@app.route('/upload', methods = ['POST', 'GET'])
def upload():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        try:
            file = request.files['image']
            full_name = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(full_name)
            indices = {0: 'Pneumonia PRESENT', 1: 'Pneumonia ABSENT'}
            result = api(full_name)
            if(result>0.50):
                label = indices[1]
                #accuracy = result
            else:
                label = indices[0]
                #accuracy = (1.0-result)
            #accuracy = round(accuracy, 2)
            return render_template('predict_pneumonia.html', image_file_name = file.filename, label = label)
        except:
            flash("Please select the image first !!", "danger")      
            return redirect(url_for("Pneumonia"))
        

@app.route('/upload2', methods = ['POST', 'GET'])
def upload2():
    if request.method == 'GET':
        return render_template('index2.html')
    else:
        try:
            file = request.files['image']
            full_name = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(full_name)
            #indices = {0: 'Parasitic', 1: 'Uninfected'}
            result = api2(full_name)
            return render_template('predict_malaria.html', image_file_name = file.filename, label = result)
        except:
            flash("Please select the image first !!", "danger")      
            return redirect(url_for("Malaria"))


#@app.route('/result', methods = ['POST', 'GET'])
#def result():
    
        
@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)





@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")

@app.route('/Pneumonia')
def Pneumonia():
    return render_template('index.html')

@app.route('/Malaria')
def Malaria():
    return render_template('index2.html')

@app.route('/Diabetes')
def Diabetes():
    return render_template('diabetes.html')

@app.route('/Heart')
def Heart():
    return render_template('heart.html')

'''Chatbot '''
@app.route('/Chatbot')
def index():
    return render_template('index3.html', title="Chatbot ")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return str(CBL.chat_bow(userText.lower()))
'''End'''


def ValuePredictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1,size)
    if(size==8):
        loaded_model = joblib.load("models/diabetes_model")
        result = loaded_model.predict(to_predict)
    elif(size==9):
        loaded_model = joblib.load("models/heart-model")
        result = loaded_model.predict(to_predict)
    return result[0]

@app.route('/result',methods = ["POST"])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
       
        if(len(to_predict_list)==8):
            result = ValuePredictor(to_predict_list,8)
        elif(len(to_predict_list)==9):
            result = ValuePredictor(to_predict_list,9)
        
    if(int(result)==1):
        prediction='Sorry! Suffering'
    else:
        prediction='Congrats! You are healthy' 
    return(render_template("result.html", prediction=prediction))

if __name__=='__main__':
    app.run(debug=True)
