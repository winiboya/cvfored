from flask import Flask, render_template, request, redirect, url_for, session
import requests
import os
import plotly.graph_objects as go
import json
import plotly
from stats import Analytics

import requests


app = Flask(__name__)
app.secret_key = 'supersecretkey'

file = "test_file.csv"
# analyze = analytics(file)


UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results')
def reuslts():
    analyze = Analytics('test_file.csv')
    fig1, fig2 = analyze.all()
    average  = round(analyze.get_average())
    student_count = 103
    total_mins = 64

    graphJSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
    chart2 = fig2.to_html(full_html=False)
    return render_template('results.html', graphJSON=graphJSON, chart2 = chart2, filename = file, average=average, student_count=student_count, total_mins=total_mins)
