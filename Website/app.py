#!venv/bin/python
import os
from flask import Flask, url_for, redirect, render_template, request, abort, session, abort, Response, send_from_directory, make_response
from functools import wraps
import gc
from flask import Flask, abort, request, jsonify, g, url_for
import json
from bson.json_util import dumps

app = Flask(__name__)



@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('login.html')
    

@app.route('/home', methods=['GET', 'POST'])
def home():
    print('Enter')
    return render_template('login.html')



@app.route('/login', methods=['GET', 'POST'])
def login():
    print('Enter')
    return render_template('login.html')

@app.route('/fetchLoginStatus')
def fetchLoginStatus():
    status = 'OK'
    dictTest = {}
    dictTest['result'] = status
    return json.dumps(dictTest)



if __name__ == '__main__':
    try:
        app.run(host= '0.0.0.0',port=5003,debug=True, threaded=False)
    except Exception as e:
        print('Error in starting::{}'.format(e))

