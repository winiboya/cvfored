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

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        
    file = request.files['file']
    if file.filename == '':
        print('No selected file')
        return redirect(request.url)
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Confirm the file is saved
        if os.path.exists(filepath):
            print('File successfully uploaded and saved')
            return redirect(url_for('results', filename=file.filename))
        else:
            print('Failed to save the file')
            return redirect(request.url)
    

@app.route('/results')
def reuslts():

    # full video analysis
    analyze = Analytics('test_file.csv', ["math", "science", "english"], ["00:00", "00:00", "00:10"], ["00:30", "00:10", "00:15"])
    all_stats = analyze.stats()
    line_chart, table, average, student_count, average_student_count, minutes, std = all_stats['line_chart'], all_stats['table'], all_stats['average'], all_stats['student_count'], all_stats['average_student_count'], all_stats['minutes'], all_stats['std']

    # topic analysis
    if analyze.topic_names is not None:
        averages_fig, average_student_count_fig, mins_fig, topics = analyze.topic_results()

    graphJSON = json.dumps(line_chart, cls=plotly.utils.PlotlyJSONEncoder)
    chart2 = table.to_html(full_html=False)
    return render_template('results.html', graphJSON=graphJSON, chart2 = chart2, filename = file, average=average, student_count=student_count, minutes=minutes)
