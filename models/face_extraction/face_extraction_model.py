import os
import cv2
import dlib
from time import time
import matplotlib.pyplot as plt


class FaceExtractionModel:
    """
    A class for extracting faces from images using a pre-trained Caffe model.

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
    
    def __init__(self, prototxt_path, caffe_model_path, input_directory, output_directory):
        """
        Initializes the FaceExtractionModel with the given paths to the prototxt and Caffe model files, 
        as well as the input and output directories.
            
        Args:
            prototxt_path (str): The path to the prototxt file for the Caffe model.
            caffe_model_path (str): The path to the Caffe model file.
            input_directory (str): The path to the directory containing input images.
            output_directory (str): The path to the directory where extracted faces will be saved.
            
        Raises:
            FileNotFoundError: If the prototxt or Caffe model file is not found.
        """
    
        # Raise errors if prototxt or Caffe model file not found
        if not os.path.isfile(prototxt_path):
            raise FileNotFoundError(f"Prototxt file not found at {prototxt_path}")
        if not os.path.isfile(caffe_model_path):
            raise FileNotFoundError(f"Caffe model file not found at {caffe_model_path}")
        
        # Load Caffe model
        self.opencv_dnn_model = cv2.dnn.readNetFromCaffe(prototxt=prototxt_path, caffeModel=caffe_model_path)

        # Initialize input and output directories
        self.input_directory = input_directory
        self.output_directory = output_directory
    
    def cv_dnn_detect_faces(self, image, min_confidence, display = True):
        """
        Detects faces in the given image using the OpenCV DNN model with the specified minimum confidence level.

        Args:
            image (numpy.ndarray): The input image in BGR format.
            min_confidence (float): The minimum confidence level for face detection.
            display (bool): Whether to display the input image with detected faces.
        
        Returns:
            output_image (numpy.ndarray): The input image with boxes drawn around detected faces.
            results (numpy.ndarray): The results of the face detection model.
            faces (list): A list of extracted face images with expanded boudning boxes.
            faces_org (list): A list of extracted face images.
            faces_count (int): The number of faces detected in the
        """
        
        # Determine height and width of input image
        image_height, image_width, _ = image.shape

        # Copy input image to draw rectangles on
        output_image = image.copy()

        # Preprocess input image
        preprocessed_image = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300),
                                                mean=(104.0, 117.0, 123.0), swapRB=False, crop=False)

        # Run model
        self.opencv_dnn_model.setInput(preprocessed_image)

        results = self.opencv_dnn_model.forward()    

        # Initialize lists to keep track of extracted images
        faces_org = []
        faces = []

        # Initialize count of detected faces
        faces_count = 0

        # Iterate through model results, counting faces above confidence threshold and filling lists of extracted faces
        for face in results[0][0]:
            
            face_confidence = face[2]
            
            if face_confidence > min_confidence:

                faces_count += 1

                bbox = face[3:]

                # Handle original face extractions
                x1a = int((bbox[0] * image_width))
                y1a = int((bbox[1] * image_height))
                x2a = int((bbox[2] * image_width))
                y2a = int((bbox[3] * image_height))

                cv2.rectangle(output_image, pt1=(x1a, y1a), pt2=(x2a, y2a), color=(0, 255, 0), thickness=image_width//200)

                extraction_org = image[y1a:y2a, x1a:x2a]

                if not (extraction_org is None or extraction_org.size == 0):
    
                    faces_org.append(extraction_org)

                # Handle padded face extractions
                padding_factor = 2

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

                x1a = max(0, min(x1a, image_width))
                y1a = max(0, min(y1a, image_height))
                x2a = max(0, min(x2a, image_width))
                y2a = max(0, min(y2a, image_height))

                extraction = image[y1:y2, x1:x2]

                if not (extraction is None or extraction.size == 0):

                    faces.append(extraction)

        # If display flag on, show image with rectangles
        if display:

            plt.figure(figsize=[20,20])
            plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off')
            plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output");plt.axis('off')
            plt.show()

            # Return input image with boxes, model results, list of expanded extractions, list of original extractions, and count of faces
            return output_image, results, faces, faces_org, faces_count

        # Else return input image with boxes, model results, list of expanded extractions, list of original extractions, and count of faces
        else:

            return output_image, results, faces, faces_org, faces_count
        

    def two_pass_face_detection(self, image, first_conf, second_conf, im, file_prefix):

        # Run first pass of model on image
        final_image, output, extractions, extractions_org, face_count = self.cv_dnn_detect_faces(image, first_conf, display=False)

        # Check that two extractions lists are of the same length
        if len(extractions) != len(extractions_org):
            raise ValueError("Two extractions lists not of same length.")

        # Determine number of faces after initial pass
        initial_faces_count = len(extractions_org)

        # Check if no faces found on initial pass
        if initial_faces_count == 0:

            return 0, 0

        else:

            final_faces = []
            extraction_index = 0
            final_face_index = 1

            for i in extractions_org:
                if self.cv_dnn_detect_faces(i, second_conf, display=False)[4] > 0:
                    
                    # print(f"Count: {count}, extractions: {len(extractions)}, extractions_org: {len(extractions_org)}")

                    # Add face extraction to final_faces
                    final_faces.append(extractions[extraction_index])

                    ## Show final extraction
                    # plt.imshow(extractions[extraction_index])
                    # plt.axis('off')
                    # plt.show()

                    # Save face extraction to output directory
                    output_directory = self.output_directory
                    os.makedirs(output_directory, exist_ok=True)
                    output_filename = f"{file_prefix}-face{final_face_index}.jpg"
                    output_path = os.path.join(output_directory, output_filename)
                    cv2.imwrite(output_path, extractions[extraction_index])
                    final_face_index += 1

                extraction_index +=1

            # Determine final number of faces
            final_faces_count = len(final_faces)

            # Check that final_face_index = final_faces_count
            if final_face_index != (final_faces_count + 1):
                raise ValueError("final_face_index not equal to (final_faces_count + 1).")

            return initial_faces_count, final_faces_count
        
    def extract_faces(self):
        
        # Initialize input directory
        image_directory = self.input_directory

        # Define image count variable
        image_count = 0
        
        total_faces = 0

        for filename in os.listdir(self.input_directory):

            # If file is image:
            if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
    
                file_path = os.path.join(self.input_directory, filename)
                
                image = cv2.imread(file_path)
                
                if image is None:

                    # with open('my_file.txt', 'a') as file:
                    #     file.write(f"Failed to load image: {file_path}\n")

                    continue

                image_count +=1

                # use image as input
                initial_count, final_count = self.two_pass_face_detection(image, 0.14, 0.99, var, filename[:-4])
                total_faces += final_count
                print(f"Extracted {final_count} faces from {filename} after two passes.")
                
                # with open('my_file.txt', 'a') as file:
                #     file.write(f"{image_count}\n")
                #     file.write(f"{filename}: {initial_count} faces detected after first pass.\n")
                #     file.write(f"{filename}: {final_count} faces detected after second pass.\n")

                

            # if file is not image
        return total_faces
        
                