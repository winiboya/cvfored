# import both classes
# call extract frame in a loop, store
# pass the extracted frame to face extraction model
# output faces into another dir
import sys
import os

sys.path.insert(0, '../models/gaze_detection')
sys.path.insert(0, '../models/face_extraction')
sys.path.insert(0, '../utils')

from face_extraction_model import FaceExtractionModel
from video_frame_extraction import VideoFrameExtraction

def main():

    output_dir = "input"
    video_frame_extraction = VideoFrameExtraction(output_dir, 5)
    
    face_extraction = FaceExtractionModel(
            prototxt_path="../models/face_extraction/deploy.prototxt", 
            caffe_model_path="../models/face_extraction/res10_300x300_ssd_iter_140000.caffemodel", 
            input_directory="input/", 
            output_directory="output"
        )
    
    # loop over videos/ directory and call frame extraction then face extraction for each video
    count = 1
    for filename in os.listdir("test_videos"):
        
    
        if filename.endswith(".MOV"):
            video_path = os.path.join("test_videos/", filename)
    
        video_frame_extraction.extract_frames(video_path, "video" + str(count))
        
        face_extraction.extract_faces("video" + str(count))
        
        count += 1



    
    print("DONE")
    
if __name__ == "__main__":
    main()