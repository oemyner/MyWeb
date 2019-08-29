# encoding:utf-8
# !/usr/bin/env python
from werkzeug.utils import secure_filename
from flask import Flask, render_template, jsonify, request, make_response, send_from_directory, abort
import time
import os
import color
import implement
# from strUtil import Pic_str
import base64

app = Flask(__name__)
UPLOAD_FOLDER = 'upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
basedir = os.path.abspath(os.path.dirname(__file__))
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'gif', 'GIF'])
f=[]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/upload')
def upload_test():
    return render_template('up.html')


# 上传文件
@app.route('/up_photo', methods=['POST'])
def api_upload():
    print("ok")
    file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    f = request.files['file']
    if f and allowed_file(f.filename):
        fname = secure_filename(f.filename)
        # ext = fname.rsplit('.', 1)[1]
        # new_filename = Pic_str().create_uuid() + '.' + ext
        f.save(os.path.join(file_dir,fname))

        return jsonify({"success": 0, "msg": "上传成功"})
    else:
        return jsonify({"error": 1001, "msg": "上传失败"})


@app.route('/download/', methods=['POST'])
def download():
    # if request.method == "GET":
    #     if os.path.isfile(os.path.join('upload', filename)):
    #         return send_from_directory('upload', filename, as_attachment=True)
    #     pass
    color.enhance_api(f)
    implement.reverse(f)
    if f:
        res=f.read()
    return "res"


# show photo
@app.route('/show/<string:filename>', methods=['GET'])
def show_photo(filename):
    file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
    if request.method == 'GET':
        if filename is None:
            pass
        else:
            image_data = open(os.path.join(file_dir, '%s' % filename), "rb").read()
            response = make_response(image_data)
            response.headers['Content-Type'] = 'image/png'
            return response
    else:
        pass


if __name__ == '__main__':
    app.run(HOST="127.0.0.1", PORT=8000)