from retinaface import RetinaFace
import matplotlib.pyplot as plt
import cv2
import os

min_confidence = 0
padding_factor = 1
# file = "test_image.jpg"
# image = cv2.imread(file)

class FaceExtractionModel:
    """
    A class for extracting faces from images using a pre-trained model.

    Attributes:
        opencv_dnn_model (cv2.dnn_Net): The OpenCV DNN model loaded from the provided prototxt and Caffe model files.

    Methods:
        __init__(prototxt_path, caffe_model_path, input_directory, output_directory):
            Initializes the FaceExtractionModel with the given paths to the prototxt and Caffe model files, 
            as well as the input and output directories.
        cv_dnn_detect_faces(image, min_confidence, display=True):
            Detects faces in the given image using the OpenCV DNN model with the specified minimum confidence level.
        two_pass_face_detection(image, first_conf, second_conf, im):
            Detects faces in the given image using two passes with different confidence levels.
        extract_faces():
            Runs the face extraction process on all images in the input directory.
    """
    
    def __init__(self, input_directory, output_directory):
        """
        Initializes the FaceExtractionModel with the given input and output directories.
            
        Args:
            input_directory (str): The path to the directory containing input images.
            output_directory (str): The path to the directory where extracted faces will be saved.
            
        """
        # Initialize input and output directories
        self.input_directory = input_directory
        self.output_directory = output_directory
    
    def detect_faces(self, image, min_confidence, file_prefix, display = True):
        """
        Detects faces in the given image using RetinaFace with the specified minimum confidence level.

        Args:
            image (numpy.ndarray): The input image in BGR format.
            min_confidence (float): The minimum confidence level for face detection.
            display (bool): Whether to display the input image with detected faces.
        
        Returns:
            output_image (numpy.ndarray): The input image with boxes drawn around detected faces.
            results (numpy.ndarray): The results of the face detection model.
            extractions (list): A list of extracted face images with expanded boudning boxes.
            extractions_org (list): A list of extracted face images.
            faces_count (int): The number of faces detected in the image.
        """

        # Determine height and width of input image
        image_shape = image.shape
        if len(image_shape) == 2:
            image_height, image_width = image_shape
            channels = 1 
        elif len(image_shape) == 3:
            image_height, image_width, channels = image_shape
    
        # Copy input image to draw rectangles on
        output_image = image.copy()

        # Preprocess input image
        preprocessed_image = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300),
                                                mean=(104.0, 117.0, 123.0), swapRB=False, crop=False)
        

        # Run model
        results = RetinaFace.detect_faces(image)

        # Set up bounding boxes/extractions lists, face count
        bboxes = []
        extractions_org = []
        extractions = []
        faces_count = 0
                    
        # Iterate through each face, check confidence score, and save bounding box
        for face_key, face_data in results.items():

            if face_data["score"] > min_confidence:

                bounding_box = face_data["facial_area"]

                bboxes.append(bounding_box)

        # Iterate through bounding boxes, draw on image, and extract faces
        for bbox in bboxes:
            

            x1a = int((bbox[0]))
            y1a = int((bbox[1]))
            x2a = int((bbox[2]))
            y2a = int((bbox[3]))

            cv2.rectangle(output_image, pt1=(x1a, y1a), pt2=(x2a, y2a), color=(0, 255, 0), thickness=image_width//200)

            extraction_org = image[y1a:y2a, x1a:x2a]

            extractions_org.append(extraction_org)

            faces_count += 1

            # Handle padded face extractions

            face_width_padding = int((x2a - x1a)/padding_factor)
            face_height_padding = int((y1a - y2a)/padding_factor)

            x1 = x1a - face_width_padding
            y1 = y1a + face_height_padding
            x2 = x2a + face_width_padding
            y2 = y2a - face_height_padding
            
            x1 = max(0, min(x1, image_width))
            y1 = max(0, min(y1, image_height))
            x2 = max(0, min(x2, image_width))
            y2 = max(0, min(y2, image_height))

            extraction = image[y1:y2, x1:x2]

            extractions.append(extraction)

            # Save face extraction to output directory
            output_directory = self.output_directory
            os.makedirs(output_directory, exist_ok=True)
            output_subdirectory = os.path.join(self.output_directory, file_prefix)
            os.makedirs(output_subdirectory, exist_ok=True)
            output_filename = f"face{faces_count}.jpg"
            output_path = os.path.join(output_subdirectory, output_filename)
            # output_path = os.path.join(output_subdirectory, output_filename)
            extraction_index = faces_count - 1
            cv2.imwrite(output_path, extractions[extraction_index])

        # Display drawn on image
        if display == True:

            plt.figure(figsize=[20,20])
            plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off')
            plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output");plt.axis('off')
            plt.show()

        return output_image, results, extractions_org, extractions_org, faces_count

    def extract_faces(self):
            
        # Initialize input directory
        image_directory = self.input_directory

        # Define image count variable
        image_count = 0

        for filename in os.listdir(self.input_directory):

            # If file is image:
            if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
    
                file_path = os.path.join(self.input_directory, filename)
                
                image = cv2.imread(file_path)
                
                if image is None:

                    continue

                image_count +=1

                # use image as input
                output_image, results, extractions_org, extractions_org, faces_count = self.detect_faces(image, min_confidence, filename[:-4], display = False)
                
        return faces_count