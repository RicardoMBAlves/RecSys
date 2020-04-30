# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 17:20:28 2020

@author: rmalves
"""

from flask import Flask, request, render_template
import pickle

app = Flask(__name__, template_folder="templates")


@app.before_first_request
def load_model():
    global pipe
    with open("model.pkl", "rb") as f:
        pipe = pickle.load(f)
        
        
@app.route("/")
def index():
    
    random_repos = 1
    return render_template("index.html", random_repos = random_repos)


@app.route("/result", methods=["POST"])
           
def predict():
    repos = int(request.form["repos"])
    suggestions = pipe.similar_items(repos)
    return render_template("result.html", suggestions=suggestions)

if __name__ =="__main__":
    app.run(debug=True, port=8000)
    
    