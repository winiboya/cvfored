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
    
@app.route('/', methods=['GET', 'POST'])
def index():

    response = None
    if request.method == 'POST':
        topic_names = request.form.getlist('topic_name')
        start_times = request.form.getlist('start_time')
        end_times = request.form.getlist('end_time')

        topics = [
            {"name": name, "start": start, "end": end}
            for name, start, end in zip(topic_names, start_times, end_times)
        ]

        response = "<h2>Submitted Topics</h2>"
        for i, topic in enumerate(topics):
            response += (f"<b>Topic {i + 1}:</b> {topic['name']}<br>"
                         f"<b>Start:</b> {topic['start']}<br>"
                         f"<b>End:</b> {topic['end']}<br><br>")

    return render_template('index.html', response=response)

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
    graphJSON2 = json.dumps(averages_fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('results.html', graphJSON=graphJSON, graphJSON2 = graphJSON2, average=average, student_count=student_count, total_mins=minutes)