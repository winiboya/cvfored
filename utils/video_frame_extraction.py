import cv2
import os

class VideoFrameExtraction:
    """
    A class for extracting frames from a video file.
    
    Attributes:
        output_dir (str): The directory where extracted frames will be saved.
        num_frames (int): The number of frames to extract.
        
    Methods:
        __init__(output_dir, num_frames):
            Initializes the VideoFrameExtraction object with the given output directory and number of frames to extract.
        extract_frames():
            Extracts frames from the video file and saves them to the output directory.
        get_video_info():
            Returns information about the video, such as total number of frames, width, height, and FPS.
        run():
            Runs the frame extraction process.
    """
    
    def __init__(self, output_dir="output_frames", num_frames=10):
        """
        Initializes the VideoFrameExtraction object with the given output directory and number of frames to extract.
        
        Args:
            output_dir (str): The directory where extracted frames will be saved.
            num_frames (int): The number of frames to extract.
        """
        self.output_dir = output_dir
        self.num_frames = num_frames
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
    def extract_frames(self, video_path, file_prefix):
        """
        Extracts frames from the video file and saves them to the output directory.
        """
        
        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_interval = total_frames // self.num_frames
        frame_number = 0
        saved_frame_count = 0

        
        while video.isOpened():
            ret, frame = video.read()
            
            if not ret:
                break
                
            if frame_number % frame_interval == 0:
                frame_filename = f"{self.output_dir}/{file_prefix}-{saved_frame_count}.jpg"
                cv2.imwrite(frame_filename, frame)
                
                saved_frame_count += 1
                
                if saved_frame_count >= self.num_frames:
                    break
                
            frame_number += 1
            
        video.release()
        cv2.destroyAllWindows()
        
        print(f"Extracted {saved_frame_count} frames from {video_path} to {self.output_dir}")
   
    # def get_video_info(self, video):
    #     """
    #     Returns information about the video, such as total number of frames, width, height, and FPS.
        
    #     Returns:
    #         dict: A dictionary containing video information.
    #     """
        
    #     width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #     fps = video.get(cv2.CAP_PROP_FPS)
    #     return {
    #         "total_frames": self.total_frames,
    #         "width": width,
    #         "height": height,
    #         "fps": fps
    #     }
        
    # def extract_frames_from_many_videos(self, video_paths):
    #     """
    #     Extracts frames from multiple video files and saves them to the output directory.
        
    #     Args:
    #         video_paths (list): A list of video file paths.
    #     """
        
    #     for video_path in video_paths:
    #         self.video = cv2.VideoCapture(video_path)
    #         self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
    #         self.extract_frames()
        