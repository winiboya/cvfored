import sys

sys.path.insert(0, "../models/gaze_detection")
sys.path.insert(0, "../models/face_extraction")
sys.path.insert(0, '../utils')

from gaze_detection import GazeDetectionModel
from face_extraction_model import FaceExtractionModel
from video_frame_extraction import VideoFrameExtraction


class Pipeline:
    def __init__(self, video_path):
        self.video_path = video_path
        self.frame_output_dir = "frame_output_dir"
        self.face_output_dir = "face_output_dir"
        self.video_frame_extraction = VideoFrameExtraction(self.frame_output_dir)
        self.face_extraction = FaceExtractionModel(
            prototxt_path="../models/face_extraction/deploy.prototxt",
            caffe_model_path="../models/face_extraction/res10_300x300_ssd_iter_140000.caffemodel",
            input_directory=self.frame_output_dir,
            output_directory=self.face_output_dir
        )
        self.gaze_detection = GazeDetectionModel("../models/gaze_detection/gaze_detection_model.h5", "../test_faces/train", "../test_faces/valid")

        
    def run(self):
        saved_frame_count, timestamps = self.video_frame_extraction.extract_frames(self.video_path)
        self.face_extraction.extract_faces()
        
        self.gaze_detection.make_predictions(self.face_output_dir, "predictions.txt", output_images=False)
        
    def clean_up(self):
        pass

def main():
    
    pipeline = Pipeline("../lecture.MOV")
    pipeline.run()
    
    

    #  face_extraction = FaceExtractionModel(
    #         prototxt_path="../models/face_extraction/deploy.prototxt",
    #         caffe_model_path="../models/face_extraction/res10_300x300_ssd_iter_140000.caffemodel",
    #         input_directory=frame_output_dir,
    #         output_directory=face_output_dir
    #     )


if __name__ == "__main__":
    main()
