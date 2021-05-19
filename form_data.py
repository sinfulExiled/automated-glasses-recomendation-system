import base64
import cv2
import json
import time
import requests
from datetime import timedelta
from imutils.video import WebcamVideoStream
from flask import Flask, render_template, json, request, redirect, session, jsonify, Response, url_for, \
    send_from_directory
from flask_mysqldb import MySQL, MySQLdb
from werkzeug.security import generate_password_hash, check_password_hash
import bcrypt
from camera import VideoCamera

app = Flask(__name__)
output = []

app.secret_key = "mykey"

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'mysql0772005661'
app.config['MYSQL_DB'] = 'virtualtryon'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
mysql = MySQL(app)


@app.route('/home')
def home():
    return render_template("home.html", result=output)


@app.route('/signUp', methods=["GET", "POST"])
def register():
    if request.method == 'GET':
        return render_template("signUpUser.html")
    else:
        name = request.form['name']
        surname = request.form['surname']
        email = request.form['email']
        nation = request.form['nation']
        password = request.form['password']
        dob = request.form['dob']
        number = request.form['number']
        gender = request.form['gender']
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO users (name, surname, email, nation, password, dob, number,gender) VALUES (%s,%s,%s,"
                    "%s,%s,%s,%s,%s)", (name, surname, email, nation, password, dob, number, gender))
        mysql.connection.commit()
        # session['name'] = request.form['name']
        # session['email'] = request.form['email']
        return redirect(url_for('home'))


@app.route('/')
def home_page():
    return render_template("loginUser.html", result=output)


@app.route('/cnn_predict')
def cnn_predict():
    return render_template("cnn.html", result=output)


@app.route('/kmeans_clustering')
def kmeans_clustering():
    return render_template("kmeans.html", result=output)


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        curl = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        curl.execute("SELECT * FROM users WHERE email=%s", (email,))
        user = curl.fetchone()
        curl.close()

        if len(user) > 0:
            if password == user["password"]:
                # session['name'] = user['name']
                # session['email'] = user['email']
                return render_template("home.html")
            else:
                print("Error password and email not match")
                return "Error password and email not match"
        else:
            print("Error user not found")
            return "Error user not found"
    else:
        return render_template("loginUser.html")


def gen(camera, makesave, imageformat):
    while True:
        data = camera.get_frame(makesave, imageformat)

        frame = data[0]

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed', methods=["GET", "POST"])
def video_feed():
    promtmsg = "noSave"
    imageformat = "null"
    if request.method == 'POST':
        promtmsg = "Save"
        imageformat = request.form['format']

    return Response(gen(VideoCamera(), promtmsg, imageformat), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("static/saves", filename)

@app.route('/savef', methods=["GET", "POST"])
def savef():
    imageformat = "jpg"
    imageformat = request.form['format']
    image_names = VideoCamera().save(imageformat)
    return render_template("cnn_results.html", image_names=image_names, result=output)

@app.route('/savekmeans', methods=["GET", "POST"])
def savek():
    imageformat = request.form['format']
    image_names = VideoCamera().savekmeans(imageformat)
    imageLocarr = []
    imageLoc = image_names[0]
    imageLocarr.append(imageLoc)
    print(imageLoc)
    faceshape = image_names[1]
    faceangle = image_names[2]
    print(faceshape)

    return render_template("kmeans_results.html",faceangle=faceangle, image_result=faceshape, image_names=imageLocarr, result=output)


if __name__ == "__main__":
    app.run(debug=True)
