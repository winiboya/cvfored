from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import requests
import threading
import os
import plotly.graph_objects as go
import json
import plotly
from stats import Analytics

import sys
sys.path.insert(0, "../src")
from pipeline import Pipeline


app = Flask(__name__)
app.secret_key = 'supersecretkey'

processing_data = {}

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    
    
def process_video(filename):
    try:
        # Update global processing_data
        global processing_data
        processing_data[filename] = {'status': 'processing', 'progress': 0}
        
        # pipeline = Pipeline("uploads/" + filename)
        # pipeline.run()
        
        processing_data[filename] = {
            'status': 'analyzing'
        }
        
        # Run analytics
        analytics = Analytics('predictions.csv', ["math", "science", "english"], 
                           ["00:00", "00:00", "00:10"], 
                           ["00:30", "00:10", "00:15"])
        
        # Update processing status and store results
        processing_data[filename] = {
            'status': 'complete',
            'analytics': analytics
        }
    except Exception as e:
        # Handle errors during processing
        processing_data[filename] = {
            'status': 'error',
            'error': str(e)
        }
    
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



@app.route('/status/<filename>')
def status(filename):
    global processing_data
    if filename not in processing_data:
        return jsonify({'status': 'not_found'})
    
    status_data = processing_data[filename]
    print(status_data)
    return jsonify({
        'status': status_data['status'],
        'error': status_data.get('error'),
        'progress': status_data.get('progress', 0)
    })

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Save the file
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Initialize processing status
        processing_data[filename] = {'status': 'processing', 'progress': 0}

        # Start processing in a separate thread
        thread = threading.Thread(target=process_video, args=(filename,))
        thread.daemon = True  # Make thread daemon so it dies when main thread dies
        thread.start()

        return jsonify({'filename': filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    
    

@app.route('/results')
def results():
    filename = request.args.get('filename')
    
    print("Filename:", filename)
    
    if filename not in processing_data:
        print("uh oh")
        return redirect('/')
    
    analytics = processing_data[filename]['analytics']
    all_stats = analytics.stats()
    line_chart, table, average, student_count, average_student_count, minutes, std = all_stats['line_chart'], all_stats['table'], all_stats['average'], all_stats['student_count'], all_stats['average_student_count'], all_stats['minutes'], all_stats['std']

    # topic analysis
    if analytics.topic_names is not None:
        averages_fig, average_student_count_fig, mins_fig, topics = analytics.topic_results()

    df = analytics.table()
    chart_data = [
        {
            "timestamp": str(row['Frame Number']),
            "focusPercentage": round(float(row['Percentages']), 1),
        }
        for _, row in df.iterrows()
    ]

    graphJSON = json.dumps(chart_data)
    graphJSON2 = json.dumps(averages_fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('results.html', graphJSON=graphJSON, graphJSON2 = graphJSON2, average=average, student_count=student_count, total_mins=minutes)