import cv2
import os

class VideoFrameExtraction:
    def __init__(self, video_path, output_dir="output_frames"):
        self.video_path = video_path
        self.output_dir = output_dir
        self.video = cv2.VideoCapture(video_path)
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
    def extract_frames(self, num_frames=10):
        frame_interval = self.total_frames // num_frames
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
                
                if saved_frame_count >= num_frames:
                    break
                
            frame_number += 1
            
        self.video.release()
        cv2.destroyAllWindows()
        
        print(f"Extracted {saved_frame_count} frames from {self.video_path} to {self.output_dir}")
   
    def get_video_info(self):
        width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.video.get(cv2.CAP_PROP_FPS)
        return {
            "total_frames": self.total_frames,
            "width": width,
            "height": height,
            "fps": fps
        }
        
def main():
    video_path = "test.MOV"
    output_dir = "output_frames"
    num_frames = 10
    
    video_frame_extraction = VideoFrameExtraction(video_path, output_dir)
    # print(video_frame_extraction.get_video_info())
    video_frame_extraction.extract_frames(num_frames)
    
if __name__ == '__main__':
    main()