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
    
        if not os.path.isfile(prototxt_path):
            raise FileNotFoundError(f"Prototxt file not found at {prototxt_path}")
        if not os.path.isfile(caffe_model_path):
            raise FileNotFoundError(f"Caffe model file not found at {caffe_model_path}")
        
        self.opencv_dnn_model = cv2.dnn.readNetFromCaffe(prototxt=prototxt_path, caffeModel=caffe_model_path)
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
            faces (list): A list of extracted face images.
            faces_count (int): The number of faces detected in the
        """
        
        image_height, image_width, _ = image.shape

        output_image = image.copy()

        preprocessed_image = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300),
                                                mean=(104.0, 117.0, 123.0), swapRB=False, crop=False)

        self.opencv_dnn_model.setInput(preprocessed_image)

        results = self.opencv_dnn_model.forward()    

        faces = []

        faces_count = 0

        for face in results[0][0]:
            
            face_confidence = face[2]
            
            if face_confidence > min_confidence:

                faces_count += 1

                bbox = face[3:]

                x1 = int(bbox[0] * image_width)
                y1 = int(bbox[1] * image_height)
                x2 = int(bbox[2] * image_width)
                y2 = int(bbox[3] * image_height)

                cv2.rectangle(output_image, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=image_width//200)

                cropped_img = image[y1:y2, x1:x2]

                if not (cropped_img is None or cropped_img.size == 0):

                    faces.append(cropped_img)

        if display:

            plt.figure(figsize=[20,20])
            plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off')
            plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output");plt.axis('off')
            plt.show()

            return output_image, results, faces, faces_count

        else:

            return output_image, results, faces, faces_count
        

    def two_pass_face_detection(self, image, first_conf, second_conf, im, file_prefix):
        """
        Detects faces in the given image using two passes with different confidence levels.
        
        Args:
            image (numpy.ndarray): The input image in BGR format.
            first_conf (float): The minimum confidence level for the first pass.
            second_conf (float): The minimum confidence level for the second pass.
            im (int): The image number for saving extracted faces.
            
        Returns:
            initial_faces_count (int): The number of faces detected after the first pass.
            final_faces_count (int): The number of faces detected after the second pass.
        """

        final_image, output, extractions, face_count = self.cv_dnn_detect_faces(image, first_conf, display=False)

        initial_faces_count = len(extractions)

        if initial_faces_count == 0:

            return 0, 0

        else:

            final = []
            var = 0

            for i in extractions:
                if self.cv_dnn_detect_faces(i, second_conf, display=False)[3] > 0:

                    final.append(i)
                    
                    var +=1
                    os.makedirs(self.output_directory, exist_ok=True)
                    output_filename = f"{file_prefix}-{im}_face-{var}.jpg"
                    output_path = os.path.join(self.output_directory, output_filename)
                    cv2.imwrite(output_path, i)

            final_faces_count = len(final)

            return initial_faces_count, final_faces_count
        
    def extract_faces(self, file_prefix):
        """
        Runs the face extraction process on all images in the input directory.
        
        Args:
            file_prefix (str): The prefix to use for the output image filenames.
        """
        
        var = 0

        for filename in os.listdir(self.input_directory):

            if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):

                file_path = os.path.join(self.input_directory, filename)
                
                image = cv2.imread(file_path)
                
                if image is None:

                    with open('my_file.txt', 'a') as file:
                        file.write(f"Failed to load image: {file_path}\n")

                    continue

                var +=1

                initial_count, final_count = self.two_pass_face_detection(image, 0.13, 0.99, var, file_prefix)
                
                with open('my_file.txt', 'a') as file:
                    file.write(f"{var}\n")
                    file.write(f"{filename}: {initial_count} faces detected after first pass.\n")
                    file.write(f"{filename}: {final_count} faces detected after second pass.\n")

            else:

                with open('my_file.txt', 'a') as file:
                    file.write(f"Skipping non-image file: {filename}\n")
                