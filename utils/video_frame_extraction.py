import cv2
import os

class VideoFrameExtraction:
    """
    A class for extracting frames from a video file.
    
    Attributes:
        output_dir (str): The directory where extracted frames will be saved.
        
    Methods:
        __init__(output_dir):
            Initializes the VideoFrameExtraction object with the given output directory and number of frames to extract.
        extract_frames(self, video_path, file_prefix):
            Extracts frames from the video file and saves them to the output directory.
    """
    
    def __init__(self, output_dir):
        """
        Initializes the VideoFrameExtraction object with the given output directory.
        
        Args:
            output_dir (str): The directory where extracted frames will be saved.
        """
        self.output_dir = output_dir
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
    def extract_frames(self, video_path, file_prefix):
        """
        Extracts frames from the video file and saves them to the output directory.
        
        Args:
            video_path (str): The path to the video file.
            file_prefix (str): The prefix to use for the extracted frame filenames.
        """
        
        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        num_frames = int(total_frames * .02)
        frame_interval = total_frames // num_frames
        frame_number = 0
        saved_frame_count = 0

        
        while video.isOpened():
            ret, frame = video.read()
            
            if not ret:
                break
                
            if frame_number % frame_interval == 0:
                frame_filename = os.path.join(self.output_dir, f"{file_prefix}-frame{saved_frame_count}.jpg")
                cv2.imwrite(frame_filename, frame)
                
                saved_frame_count += 1
                
                if saved_frame_count >= num_frames:
                    break
                
            frame_number += 1
            
        video.release()
        cv2.destroyAllWindows()
        
        return saved_frame_count