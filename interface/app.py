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
    
    
    
@app.route('/', methods=['GET', 'POST'])
def index():

    return render_template('index.html')



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
    
    
    topic_names = request.form.getlist('topic_name')
    start_times = request.form.getlist('start_time')
    end_times = request.form.getlist('end_time')

    topics = None
    if topic_names and topic_names[0].strip():  # Check if there's at least one non-empty topic
        topics = [
            {"name": name, "start": start, "end": end}
            for name, start, end in zip(topic_names, start_times, end_times)
        ]
        
        if not topics:
            topics = None
            print("No topics")
    print(topics)
    

    try:
        # Save the file
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Initialize processing status
        processing_data[filename] = {'status': 'processing', 'progress': 0, 'topics': topics}

        # Start processing in a separate thread
        thread = threading.Thread(target=process_data, args=(filename,))
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
        print("File not found")
        return render_template('apology.html')
    
    analytics = processing_data[filename]['analytics']
    all_stats = analytics.stats()
    line_chart, table, average, student_count, average_student_count, minutes, std = all_stats['line_chart'], all_stats['table'], all_stats['average'], all_stats['student_count'], all_stats['average_student_count'], all_stats['minutes'], all_stats['std']

    topic_data = None
    # topic analysis
    if analytics.topic_names is not None:
        averages_fig, average_student_count_fig, mins_fig, topics = analytics.topic_results()
        topic_data = [
            {
                "category": name,
                "focusPercentage": topics[name]['average'],
                "studentCount": topics[name]['average_student_count'],
                "minutes": topics[name]['minutes'],
                "standardDeviation": topics[name]['std']
            }
            for name in analytics.topic_names
        ]
        
    df = analytics.table()
    chart_data = [
        {
            "timestamp": str(row['Frame Number']),
            "focusPercentage": round(float(row['Percentages']), 1),
        }
        for _, row in df.iterrows()
    ]

    print(topic_data)
    # graphJSON2 = json.dumps(averages_fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('results.html', focusData=json.dumps(chart_data), topicData=json.dumps(topic_data), average=average, student_count=student_count, total_mins=minutes)



def process_data(filename):
    try:
        # Update global processing_data
        global processing_data
        file_data = processing_data[filename]
        topics = file_data.get('topics')
        processing_data[filename].update({'status': 'processing', 'progress': 0})

        
        # pipeline = Pipeline("uploads/" + filename)
        # pipeline.run()
        
        processing_data[filename] = {
            'status': 'analyzing'
        }
        
        if topics is None:
            analytics = Analytics('predictions.csv')
        
        else:
            analytics = Analytics('predictions.csv', topic_names=[topic['name'] for topic in topics],
                           topic_starts=[topic['start'] for topic in topics],
                           topic_ends=[topic['end'] for topic in topics])
            
        

        
        
        # Update processing status and store results
        processing_data[filename].update({
            'status': 'complete',
            'analytics': analytics
        })
    except Exception as e:
        # Handle errors during processing
        processing_data[filename].update({
            'status': 'error',
            'error': str(e)
        })
        
def validate_topics(topics):
    pass