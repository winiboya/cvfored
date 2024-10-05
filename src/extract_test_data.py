import sys
import os

sys.path.insert(0, '../models/gaze_detection')
sys.path.insert(0, '../models/face_extraction')
sys.path.insert(0, '../utils')

from face_extraction_model import FaceExtractionModel
from video_frame_extraction import VideoFrameExtraction

# class TestDataExtraction:
#     def __init__(self, input_dir, frame_output_dir,face_output_dir):
#         self.input_dir = input_dir
#         self.frame_output_dir = output_dir
#         self.face_output_dir = "extracted_faces"
        
#         if not os.path.exists(self.output_dir):
#             os.makedirs(self.output_dir)
            
#     def extract_test_data(self):
#         video_frame_extraction = VideoFrameExtraction(self.output_dir)
        
#         face_extraction = FaceExtractionModel(
#             prototxt_path="../models/face_extraction/deploy.prototxt", 
#             caffe_model_path="../models/face_extraction/res10_300x300_ssd_iter_140000.caffemodel", 
#             input_directory=f"{self.output_dir}/", 
#             output_directory="extracted_faces"
#         )
        
#         count = 1
#         for filename in os.listdir(self.input_dir):
#             if filename.endswith(".MOV"):
#                 video_path = os.path.join(f"{self.input_dir}/", filename)
                
#             video_frame_extraction.extract_frames(video_path, "video" + str(count))
            
#             count += 1
            
#         for filename in os.listdir("extracted_frames"):
#             if filename.endswith(".jpg"):
#                 file_prefix = filename[:-4]
#                 face_extraction.extract_faces(file_prefix)
                
#         print("DONE")

def main():
    input_dir = "smallvideos"
    output_dir = "extracted_frames"
    
    video_frame_extraction = VideoFrameExtraction(output_dir)
    
    face_extraction = FaceExtractionModel(
            prototxt_path="../models/face_extraction/deploy.prototxt", 
            caffe_model_path="../models/face_extraction/res10_300x300_ssd_iter_140000.caffemodel", 
            input_directory=output_dir, 
            output_directory="extracted_faces"
        )
    
    # loop over videos/ directory and call frame extraction then face extraction for each video
    count = 1
    for filename in os.listdir(input_dir):
        
    
        if filename.endswith(".MOV"):
            video_path = os.path.join(input_dir, filename)
    
        video_frame_extraction.extract_frames(video_path, "video" + str(count))
        
        
        count += 1

    for filename in os.listdir("extracted_frames"):
        if filename.endswith(".jpg"):
            file_prefix = filename[:-4]
            face_extraction.extract_faces(file_prefix)

    
    print("DONE")
    
if __name__ == "__main__":
    main()