from flask import Flask, render_template, request, redirect, url_for, session
import plotly.graph_objects as go
import json
import plotly
from stats import analytics
import requests
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey'


UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ASSUMING OUTPUT OBTAINED
file = "test_file.csv"


@app.route('/')
def index():
    analyze = analytics(file)
    average  = analyze.get_average()
    student_count = 103
    total_mins = 64
    fig1, fig2 = analyze.all()

    graphJSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
    chart2 = fig2.to_html(full_html=False)
    return render_template('index.html', graphJSON=graphJSON, chart2 = chart2, filename = file, average=average, student_count=student_count, total_mins=total_mins)
    
# @app.route('/')
# def index():
#     return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)