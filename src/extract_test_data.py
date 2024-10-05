import sys
import os

sys.path.insert(0, '../models/gaze_detection')
sys.path.insert(0, '../models/face_extraction')
sys.path.insert(0, '../utils')

from face_extraction_model import FaceExtractionModel
from video_frame_extraction import VideoFrameExtraction

class TestDataExtraction:
    """
    A class for extracting test data from video files.
    
    Attributes:
        input_dir (str): The directory containing the input video files.
        frame_output_dir (str): The directory where extracted frames will be saved.
        face_output_dir (str): The directory where extracted faces will be saved.
        video_frame_extraction (VideoFrameExtraction): An instance of the VideoFrameExtraction class.
        face_extraction (FaceExtractionModel): An instance of the FaceExtractionModel class.
    
    Methods:
        __init__(input_dir, frame_output_dir, face_output_dir):
            Initializes the TestDataExtraction object with the given input directory, frame output directory, and face output directory.
        extract_frames():
            Extracts frames from the video files in the input directory.
        extract_faces():
            Extracts faces from the extracted
    """
    def __init__(self, input_dir, frame_output_dir,face_output_dir):
        """
        Initializes the TestDataExtraction object with the given input directory, frame output directory, and face output directory.
        
        Args:
            input_dir (str): The directory containing the input video files.
            frame_output_dir (str): The directory where extracted frames will be saved.
            face_output_dir (str): The directory where extracted faces will be saved.
        """
        self.input_dir = input_dir
        self.frame_output_dir = frame_output_dir
        self.face_output_dir = face_output_dir
        
        self.video_frame_extraction = VideoFrameExtraction(self.frame_output_dir)
        self.face_extraction = FaceExtractionModel(
            prototxt_path="../models/face_extraction/deploy.prototxt", 
            caffe_model_path="../models/face_extraction/res10_300x300_ssd_iter_140000.caffemodel", 
            input_directory=frame_output_dir, 
            output_directory=face_output_dir
        )
        
    def extract_frames(self):
        """
        Uses the VideoFrameExtraction class to extract frames from the video files in the input directory.
        """ 
        
        count = 1
        for filename in os.listdir(self.input_dir):
            if filename.endswith(".MOV"):
                video_path = os.path.join(self.input_dir, filename)
                self.video_frame_extraction.extract_frames(video_path, "video" + str(count))
                count += 1
                
    def extract_faces(self):
        """
        Uses the FaceExtractionModel class to extract faces from the extracted frames.
        """
        for filename in os.listdir(self.frame_output_dir):
            if filename.endswith(".jpg"):
                file_prefix = filename[:-4]
                self.face_extraction.extract_faces(file_prefix)
    


def main():
    input_dir = "small_vids"
    frame_output_dir = "output_frames"
    face_output_dir = "output_faces"
    
    data_extraction = TestDataExtraction(input_dir, frame_output_dir, face_output_dir)
    data_extraction.extract_frames()
    data_extraction.extract_faces()
    
if __name__ == "__main__":
    main()