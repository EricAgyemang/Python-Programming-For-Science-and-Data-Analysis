# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 08:58:49 2021

@author: mmbono1
"""

from flask import Flask,request


app=Flask(__name__)

@app.route('/')
def index():
    return 'Rest API Binding Health Check'
@app.route('/add')
def add():
    num1=request.args.get('num1')
    num2=request.args.get('num2')
    result=int(num1)+int(num2)
    return str(result)
@app.route('/predict',methods=['GET','POST'])
def httpmethods():
    if request.method=='POST':
        return 'You want to update the claim insight'
    else:
        return 'You want to read claim insight'
if __name__=='__main__':
    app.debug=True
    app.run(host= '127.0.0.1',port=5000)
    