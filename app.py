from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import pdfdemo as pdfgen
import esdemo as puti
import mergingpdf as mpdf
import appendingpdf as ap
from flask import send_file
import frontfacedetection as fd
import os
import cv2

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, session,render_template
from werkzeug.utils import secure_filename
import sqlite3

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

UPLOAD_FOLDER = 'static/uploads/'

# allow files of a specific type
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

# function to check the file extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model_path2 = 'model.h5' # load .h5 Model

CTS = load_model(model_path2)
from keras.preprocessing.image import load_img, img_to_array

def model_predict2(image_path,model):
    print("Predicted")
    #image = load_img(image_path,target_size=(224,224))
    img=cv2.imread(image_path)
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector=cv2.CascadeClassifier(harcascadePath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    file_namess = os.path.basename(image_path)
    print("Filenames==",file_namess)

    path = 'C:\\Users\\Vrunda\\Desktop\\newui\\cropped'

    for (x,y,w,h) in faces:
        
        crop_img = img[y:(y+h), x:(x+w)]
        #cv2.imwrite("cropped/"+file_namess, crop_img)
        
        cv2.imwrite(os.path.join(path , file_namess), crop_img)
    image_path="cropped/"+file_namess
    print("image path==",image_path)

    image = load_img(image_path,target_size=(224,224))
    image = img_to_array(image)
    image = image/255
    image = np.expand_dims(image,axis=0)
    
    result = np.argmax(model.predict(image))
    #prediction = classes2[result]  
    
    if result == 0:
        return "Down Syndrome","result.html"        
    elif result == 1:
        return "Non Down Syndrome","result.html"
   
@app.route('/patreg',methods=['POST'])
def patreg():
	pid=request.form['pid']
	pname=request.form['pname']
	age=request.form['age']
	gender=request.form['g1']
	session['patname'] = request.form['pname']
	pdfgen.process(pid,pname,age,gender)
	return render_template("uploadpic.html")  
      
@app.route("/")
def index():
    return render_template("index.html")

@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')

@app.route("/signup")
def signup():

    username = request.args.get('user','')
    # name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    con = sqlite3.connect('data.db')
    cur = con.cursor()
    cur.execute("insert into information (`user`,`email`, `password`,`mobile`) VALUES (?, ?, ?, ?)",(username,email,password,number))
    con.commit()
    con.close()
    return render_template("signin.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('data.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from information where `user` = ? AND `password` = ?",(mail1,password1))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("addpatient.html")
    else:
        return render_template("signup.html")

@app.route('/home')
def home():
	return render_template('uploadpic.html')

@app.route('/predict2',methods=['GET','POST'])
def predict2():
    print("Entered")
    
    
    print("Entered here")
    file = request.files['files'] # fet input
    print("File ==",file)
    filename = file.filename 
    print("File name==",filename)       
    print("@@ Input posted = ", filename)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
            
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        fd.process(file_path)

        print("@@ Predicting class......")
       
        pred, output_page = model_predict2(file_path,CTS)
        uname= session['patname']
        ap.process(uname,uname,pred)
        #puti.process(uname,UPLOAD_FOLDER+file.filename)
        #mpdf.process(uname)
        return render_template(output_page, pred_output = pred, img_src=UPLOAD_FOLDER + file.filename)
    else:
        msg="Unsupported file format"
        return render_template("uploadpic.html",message=msg)
        
@app.route('/download', methods=['POST'])
def download_file():
	path=""
	uname= session['patname']
	print("uname==",uname)
	path=str(uname)+"final.pdf"
	return send_file(path, as_attachment=True)
@app.route("/logout",methods=['POST'])
def log_out():
    session.clear()
    return render_template("signin.html")


   
if __name__ == '__main__':
    app.run(debug=False)
