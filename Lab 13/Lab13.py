# -*- coding: utf-8 -*-
"""

from flask import Flask, request

#Declare Flask Application

app= Flask(__name__)

# Declare your web application API route
# This is REST API binding

@app.route('/')
def index():
    return 'REST API Binding Health Check'

#Application Route for add two numbers REST API call
@app.route('/add')
def add():
    num1=request.args.get('num1')
    num2=request.args.get('num2')
    
    result=int(num1)+int(num2)
    return str(result)

#HTTP Methods (GET and POST)

@app.route('/predict',methods=['GET','POST'])
def httpmethods():
    if request.method=='POST':
        return 'You want to update the claim insight'
    else:
        return 'You want to read claim insight'

# Main program entry point to accept REST API calls

if __name__=='__main__':
    app.debug=True
    app.run(host='127.0.0.1',port=5000)
    


