import sys
import os
import argparse

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
            input_directory=frame_output_dir, 
            output_directory=face_output_dir
        )
        
    def extract_frames(self):
        """
        Uses the VideoFrameExtraction class to extract frames from the video files in the input directory.
        """ 
        
        
        count = 1
        num_frames = 0
        for filename in os.listdir(self.input_dir):
            if filename.endswith(".MOV"):
                video_path = os.path.join(self.input_dir, filename)
                new_num_frames, _ = self.video_frame_extraction.extract_frames(video_path, "video" + str(count))
                num_frames += new_num_frames
                count += 1
        return num_frames
                
    def extract_faces(self):
        """
        Uses the FaceExtractionModel class to extract faces from the extracted frames.
        """
        # for filename in os.listdir(self.frame_output_dir):
        #     if filename.endswith(".jpg"):
        #         file_prefix = filename[:-4]
        #         self.face_extraction.extract_faces(file_prefix)
        return self.face_extraction.extract_faces()
        
    


def main():
    parser = argparse.ArgumentParser(description="Extract frames and faces from video files.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the input video files.")
    parser.add_argument("--frame_output_dir", type=str, required=True, help="Directory where extracted frames will be saved.")
    parser.add_argument("--face_output_dir", type=str, required=True, help="Directory where extracted faces will be saved.")

    args = parser.parse_args()
    
    data_extraction = TestDataExtraction(args.input_dir, args.frame_output_dir, args.face_output_dir)
    total_frames = data_extraction.extract_frames()
    print(f"Total number of frames extracted: {total_frames}")
    total_faces = data_extraction.extract_faces()
    print(f"Total number of faces extracted: {total_faces}")
    
if __name__ == "__main__":
    main()