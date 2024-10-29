# import both classes
# call extract frame in a loop, store
# pass the extracted frame to face extraction model
# output faces into another dir
import sys
import os

sys.path.insert(0, '../models/gaze_detection')
sys.path.insert(0, '../models/face_extraction')
sys.path.insert(0, '../utils')

from face_extraction_model_test import FaceExtractionModel

def main():
    
    face_extraction = FaceExtractionModel(
            prototxt_path="deploy.prototxt", 
            caffe_model_path="res10_300x300_ssd_iter_140000.caffemodel", 
            input_directory="test_input/", 
            output_directory="test_output"
        )
    
    face_extraction.extract_faces()
    # face_extraction.one_pass_extract_faces()
    
    print("DONE")
    
if __name__ == "__main__":
    main()