#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ast import Return
from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('Player.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['age']
    data2 = request.form['Preferred Foot']
    data3 = request.form['weight']
    data4 = request.form['Height']
    data5= request.form['crossing']
    data6= request.form['Finishing']
    data7= request.form['Heading_accuracy']
    data8= request.form['shortpassing']
    data9= request.form['volley']
    
    arr = np.array([[data1, data2, data3, data4,data5,data6,data7,data8,data9]])
    pred = model.predict(arr)
    return render_template('after.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)



# %%
