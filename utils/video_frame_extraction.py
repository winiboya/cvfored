import cv2
import os

class VideoFrameExtraction:
    """
    A class for extracting frames from a video file.
    
    Attributes:
        video_path (str): The path to the video file.
        output_dir (str): The directory where extracted frames will be saved.
        video (cv2.VideoCapture): The OpenCV video capture object.
        total_frames (int): The total number of frames in the video.
        num_frames (int): The number of frames to extract.
        
    Methods:
        __init__(video_path, output_dir, num_frames):
            Initializes the VideoFrameExtraction object with the given video file path, output directory, and number of frames to extract.
        extract_frames():
            Extracts frames from the video file and saves them to the output directory.
        get_video_info():
            Returns information about the video, such as total number of frames, width, height, and FPS.
        run():
            Runs the frame extraction process.
    """
    
    def __init__(self, video_path, output_dir="output_frames", num_frames=10):
        """
        Initializes the VideoFrameExtraction object with the given video file path, output directory, and number of frames to extract.
        
        Args:
            video_path (str): The path to the video file.
            output_dir (str): The directory where extracted frames will be saved.
            num_frames (int): The number of frames to extract.
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.video = cv2.VideoCapture(video_path)
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.num_frames = num_frames
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
    def extract_frames(self):
        """
        Extracts frames from the video file and saves them to the output directory.
        """
        frame_interval = self.total_frames // self.num_frames
        frame_number = 0
        saved_frame_count = 0

        
        while self.video.isOpened():
            ret, frame = self.video.read()
            
            if not ret:
                break
                
            if frame_number % frame_interval == 0:
                frame_filename = f"{self.output_dir}/frame_{saved_frame_count}.jpg"
                cv2.imwrite(frame_filename, frame)
                
                saved_frame_count += 1
                
                if saved_frame_count >= self.num_frames:
                    break
                
            frame_number += 1
            
        self.video.release()
        cv2.destroyAllWindows()
        
        print(f"Extracted {saved_frame_count} frames from {self.video_path} to {self.output_dir}")
   
    def get_video_info(self):
        """
        Returns information about the video, such as total number of frames, width, height, and FPS.
        
        Returns:
            dict: A dictionary containing video information.
        """
        
        width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.video.get(cv2.CAP_PROP_FPS)
        return {
            "total_frames": self.total_frames,
            "width": width,
            "height": height,
            "fps": fps
        }
        
    def run():
        """
        Runs the frame extraction process.
        """
        video_path = "test.MOV"
        output_dir = "output_frames"
        num_frames = 10
        
        video_frame_extraction = VideoFrameExtraction(video_path, output_dir, num_frames)
        # print(video_frame_extraction.get_video_info())
        video_frame_extraction.extract_frames(num_frames)