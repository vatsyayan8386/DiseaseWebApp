# Important Modules
from flask import Flask, render_template, url_for, flash, redirect, session

# from flask.ext.session import Session
# from forms import RegistrationForm, LoginForm
from sklearn.externals import joblib
from flask import request
import numpy as np
import tensorflow
# from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D
# from flask_sqlalchemy import SQLAlchemy
# from model_class import DiabetesCheck, CancerCheck

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D
# from tensorflow.keras.layers import GlobalMaxPooling2D, Activation
# from tensorflow.keras.layers.normalization import BatchNormalization
# from tensorflow.keras.layers.merge import Concatenate
# from tensorflow.keras.models import Model

import os
#from flask import send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

# from this import SQLAlchemy
from flask_session import Session
app = Flask(__name__, template_folder='template')
#app.secret_key = os.urandom(24)

#login_manager = flask_login.lo()
#login_manager.init_app(app)
# RELATED TO THE SQL DATABASE
#app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
#app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///site.db"
#db=SQLAlchemy(app)

# from model import User,Post

# //////////////////////////////////////////////////////////

dir_path = os.path.dirname(os.path.realpath(__file__))
# UPLOAD_FOLDER = dir_path + '/uploads'
# STATIC_FOLDER = dir_path + '/static'
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

# graph = tf.get_default_graph()
# with graph.as_default():;
from tensorflow.keras.models import load_model

model = load_model('model111.h5')
model222 = load_model("my_model.h5")


# FOR THE FIRST MODEL

# call model to predict an image
def api(full_path):
    data = image.load_img(full_path, target_size=(50, 50, 3))
    data = np.expand_dims(data, axis=0)
    data = data * 1.0 / 255

    # with graph.as_default():
    predicted = model.predict(data)
    return predicted


# FOR THE SECOND MODEL
def api1(full_path):
    data = image.load_img(full_path, target_size=(64, 64, 3))
    data = np.expand_dims(data, axis=0)
    data = data * 1.0 / 255

    # with graph.as_default():
    predicted = model222.predict(data)
    return predicted


# home page

# @app.route('/')
# def home():
#  return render_template('index.html')


# procesing uploaded file and predict it
@app.route('/upload', methods=['POST', 'GET'])
def upload_file():
    if request.method == 'GET':
        return render_template('Malaria.html')
    else:
        try:
            file = request.files['image']
            full_name = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(full_name)

            indices = {0: 'PARASITIC', 1: 'Uninfected', 2: 'Invasive carcinomar', 3: 'Normal'}
            result = api(full_name)
            print(result)

            predicted_class = np.asscalar(np.argmax(result, axis=1))
            accuracy = round(result[0][predicted_class] * 100, 2)
            label = indices[predicted_class]
            return render_template('Malaria_positive.html', image_file_name=file.filename, label=label,
                                   accuracy=accuracy)
        except:
            flash("Please select the image first !!", "danger")
            return redirect(url_for("Malaria"))


@app.route('/upload11', methods=['POST', 'GET'])
def upload11_file():
    if request.method == 'GET':
        return render_template('Pneumonia.html')
    else:
        try:
            file = request.files['image']
            full_name = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(full_name)
            indices = {0: 'Normal', 1: 'Pneumonia'}
            result = api1(full_name)
            if (result > 50):
                #label = indices[1]
                #accuracy = result

                prediction = 'Congrats ! you are Healthy'
                return (render_template("Pneumonia_negative.html", prediction=prediction))
            else:
                #label = indices[0]
                #accuracy = 100 - result
                prediction = 'Sorry ! You have symptoms for Pneumonia'
                return (render_template("Pneumonia_positive.html", prediction=prediction))
            #return render_template('predict1.html', image_file_name=file.filename, label=label, accuracy=accuracy)
        except:
            flash("Please select the image first !!", "danger")
            return redirect(url_for("Pneumonia"))


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# //////////////////////////////////////////////

# app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///site.db"
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
# db=SQLAlchemy(app)

# class User(db.Model):
##   username = db.Column(db.String(20), unique=True, nullable=False)
#   email = db.Column(db.String(120), unique=True, nullable=False)
# image_file = db.Column(db.String(20), nullable=False, default='default.jpg')
#   password = db.Column(db.String(60), nullable=False)
# posts = db.relationship('Post', backref='author', lazy=True)

# def __repr__(self):
#   return f"User('{self.username}', '{self.email}', '{self.image_file}')"


@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/Cancer_page")
def cancer():
    return render_template("cancer.html")


@app.route("/Diabetes_page")
def diabetes():
    # if form.validate_on_submit():
    return render_template("diabetes.html")


@app.route("/Heart_page")
def heart():
    return render_template("heart.html")


@app.route("/Liver_page")
def liver():
    # if form.validate_on_submit():
    return render_template("liver.html")


@app.route("/Kidney_page")
def kidney():
    # if form.validate_on_submit():
    return render_template("kidney.html")


@app.route("/Malaria")
def Malaria():
    return render_template("Malaria.html")


@app.route("/Pneumonia")
def Pneumonia():
    return render_template("Pneumonia.html")


def ValuePredictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1, size)
    if (size == 8):  # Diabetes
        loaded_model = joblib.load("model1")
        result = loaded_model.predict(to_predict)
    elif (size == 30):  # Cancer
        loaded_model = joblib.load("model")
        result = loaded_model.predict(to_predict)
    elif (size == 12):  # Kidney
        loaded_model = joblib.load("model3")
        result = loaded_model.predict(to_predict)
    elif (size == 10):
        loaded_model = joblib.load("model4")
        result = loaded_model.predict(to_predict)
    elif (size == 11):  # Heart
        loaded_model = joblib.load("model2")
        result = loaded_model.predict(to_predict)
    return result[0]


@app.route('/result', methods=["POST"])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        if (len(to_predict_list) == 30):  # Cancer
            result = ValuePredictor(to_predict_list, 30)
        elif (len(to_predict_list) == 8):  # Daiabtes
            result = ValuePredictor(to_predict_list, 8)
        elif (len(to_predict_list) == 12):
            result = ValuePredictor(to_predict_list, 12)
        elif (len(to_predict_list) == 11):
            result = ValuePredictor(to_predict_list, 11)
            # if int(result)==1:
            #   prediction ='diabetes'
            # else:
            #   prediction='Healthy'
        elif (len(to_predict_list) == 10):
            result = ValuePredictor(to_predict_list, 10)
    if (int(result) == 1):
        if  (len(to_predict_list) == 30):
            prediction = 'Sorry ! You have symptoms for Breast Cancer'
            return (render_template("Cancer_positive.html", prediction=prediction))
        elif (len(to_predict_list) == 8):
            prediction = 'Sorry ! You have symptoms for Diabetes'
            return (render_template("Diabetes_positive.html", prediction=prediction))
        elif (len(to_predict_list) == 12):
            prediction = 'Sorry ! You have symptoms for Chronic Kidney Disease'
            return (render_template("Kidney_positive.html", prediction=prediction))
        elif (len(to_predict_list) == 10):
            prediction = 'Sorry ! You have symptoms for Liver Disease'
            return (render_template("Liver_positive.html", prediction=prediction))
        elif (len(to_predict_list) == 11):
            prediction = 'Sorry ! You have symptoms for Heart Disease'
            return (render_template("Heart_positive.html", prediction=prediction))

    else:
        if  (len(to_predict_list) == 30):
            prediction = 'Congrats ! you are Healthy'
            return (render_template("Cancer_negative.html", prediction=prediction))
        elif (len(to_predict_list) == 8):
            prediction = 'Congrats ! you are Healthy'
            return (render_template("Diabetes_negative.html", prediction=prediction))
        elif (len(to_predict_list) == 12):
            prediction = 'Congrats ! you are Healthy'
            return (render_template("Kidney_negative.html", prediction=prediction))
        elif (len(to_predict_list) == 10):
            prediction = 'Congrats ! you are Healthy'
            return (render_template("Liver_negative.html", prediction=prediction))
        elif (len(to_predict_list) == 11):
            prediction = 'Congrats ! you are Healthy'
            return (render_template("Heart_negative.html", prediction=prediction))



if __name__ == "__main__":
    app.run(debug=True)
'''if __name__ == "__main__":
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    sess = Session()
    sess.init_app(app)

app.debug = True
app.run()'''