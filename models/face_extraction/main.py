from face_extraction_model import FaceExtractionModel

def main():
    model = FaceExtractionModel(
        prototxt_path="deploy.prototxt", 
        caffe_model_path="res10_300x300_ssd_iter_140000.caffemodel", 
        input_directory="input/", 
        output_directory="output"
    )
    model.extract_faces()
    
if __name__ == "__main__":
    main()