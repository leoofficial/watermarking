import os
import random
import cv2
import numpy as np

from flask import Flask, json, render_template, send_from_directory, request
import watermark as wm

app = Flask(__name__)


def clear_files(func):
    def wrapper():
        for image in os.listdir(os.path.join('static', 'images')):
            os.remove(os.path.join('static', 'images', image))
        for image in os.listdir(os.path.join('static', 'keys')):
            os.remove(os.path.join('static', 'keys', image))
        res = func()
        return res
    wrapper.__name__ = func.__name__
    return wrapper


@app.route('/')
def home_page():
    if not os.path.exists(os.path.join('static', 'keys')):
        os.makedirs(os.path.join('static', 'images'))
    if not os.path.exists(os.path.join('static', 'keys')):
        os.makedirs(os.path.join('static', 'images'))
    return render_template('index.html')


@app.route('/embed')
def embed_page():
    return render_template('embed.html')


@app.route('/embed', methods=['POST'])
@clear_files
def embed():
    host = request.files['host']
    watermark = request.files['watermark']
    robustness = int(request.form['robustness'])
    if '.' in host.filename and host.filename.rsplit('.', 1)[1] in {'jpeg', 'jpg'} and \
       '.' in watermark.filename and watermark.filename.rsplit('.', 1)[1] in {'jpeg', 'jpg', 'png'}:
        random_idx = str(random.randrange(1 << 16))
        host_path = os.path.join('static', 'images', 'host') + random_idx
        watermark_path = os.path.join('static', 'images', 'watermark') + random_idx
        key_path = os.path.join('static', 'keys', 'key') + random_idx
        host.save(host_path)
        host = cv2.imread(host_path)
        watermark.save(watermark_path)
        watermark = cv2.imread(watermark_path, 0)
        host, key = wm.embed(host, watermark, robustness)
        host_path = host_path + '.jpg'
        key_path = key_path + '.npz'
        cv2.imwrite(host_path, host)
        np.savez(key_path, id=random_idx, shape=key['shape'], pxl_perm_mat=key['pxl_perm_mat'])
        return json.jsonify({
            'status': 'success',
            'data': {
                'host': host_path,
                'key': key_path,
            }
        })
    return json.jsonify({
        'status': 'fail',
        'data': {
            'host': None,
            'key': None,
        }
    })


@app.route('/extract')
def extract_page():
    return render_template('extract.html')


@app.route('/extract', methods=['POST'])
@clear_files
def extract():
    host = request.files['host']
    key = request.files['key']
    if '.' in host.filename and host.filename.rsplit('.', 1)[1] in {'jpeg', 'jpg'} and \
       '.' in key.filename and key.filename.rsplit('.', 1)[1] == 'npz':
        host_path = os.path.join('static', 'images', 'host')
        watermark_path = os.path.join('static', 'images', 'watermark')
        key_path = os.path.join('static', 'keys', 'key.npy')
        host.save(host_path)
        host = cv2.imread(host_path)
        key.save(key_path)
        key = np.load(key_path)
        watermark = wm.extract(host, key)
        watermark_path = watermark_path + '.jpg'
        cv2.imwrite(watermark_path, watermark)
        return json.jsonify({
            'status': 'success',
            'watermark': watermark_path
        })
    return json.jsonify({
        'status': 'fail',
        'watermark': None
    })


@app.route('/static/<path:path>')
def get_file(path):
    return send_from_directory('static', path)


@app.route('/favicon.ico')
def get_favicon():
    return send_from_directory(os.path.join('assets', 'icons'), 'favicon.ico')


if __name__ == '__main__':
    app.run()
