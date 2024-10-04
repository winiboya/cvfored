import sys

sys.path.insert(0, '../models/gaze_detection')
sys.path.insert(0, '../models/face_extraction')

from gaze_detection_model import GazeDetectionModel
from face_extraction_model import FaceExtractionModel

def main():
    print("Starting the training process...")
    
    face_extraction = FaceExtractionModel(
        prototxt_path='', 
        caffe_model_path='', 
        input_directory='', 
        output_directory=''
    )
    
    face_extraction.run()
    
    # Initialize the model with desired parameters
    gaze_detection = GazeDetectionModel(
        model_path='gaze_detection_model.h5',
        train_data_path='../testdata/gaze/',
        batch_size=32,
        target_size=(224, 224),
        epochs=10
    )

    # Run the training and saving process
    # gaze_detection.run()
    # model.predict_and_save_images('output/', 10)
    # model.predict_and_save_single_image("image.png", "not_focused")

if __name__ == '__main__':
    main()
