# Computer Vision for Education

This repository contains the code for the paper "A Computer Vision Approach For Assessing Audience Distraction In Educational Settings" by Carolina Zubler and Winiboya Aboyure.

# Getting Started

Create a virtual environment and install the required packages using the following command:

```
python3 -m venv env
source env/bin/activate

pip install -r requirements.txt
brew install ffmpeg
```

# Running the code

## Extracting Test Data
```
cd src
python extract_test_data.py --input_dir "input_dir" --frame_output_dir "output_frames" --face_output_dir "output_faces"
```

## Training the Model

```
cd models/gaze_detection
python gaze_detection_model.py --model_path="gaze_detection_model.h5" --train_data_path="../../test_faces/train" --validate_data_path="../../test_faces/validate" 
```