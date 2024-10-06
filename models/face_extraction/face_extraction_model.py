import os
import cv2
import dlib
from time import time
import matplotlib.pyplot as plt

class FaceExtractionModel:
    def __init__(self, prototxt_path, caffe_model_path, input_directory, output_directory):
        # Verify model files
        if not os.path.isfile(prototxt_path):
            raise FileNotFoundError(f"Prototxt file not found at {prototxt_path}")
        if not os.path.isfile(caffe_model_path):
            raise FileNotFoundError(f"Caffe model file not found at {caffe_model_path}")
        
        self.input_directory = input_directory

        self.output_directory = output_directory

        # Load model
        self.opencv_dnn_model = cv2.dnn.readNetFromCaffe(prototxt=prototxt_path, caffeModel=caffe_model_path)
    
    def cv_dnn_detect_faces(self, image, min_confidence, display = True):
        
        # determine image height and width
        image_height, image_width, _ = image.shape

        # copy image for writing on
        output_image = image.copy()

        # preprocess image
        preprocessed_image = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300),
                                                mean=(104.0, 117.0, 123.0), swapRB=False, crop=False)

        # run model on preprocessed image
        self.opencv_dnn_model.setInput(preprocessed_image)

        results = self.opencv_dnn_model.forward()    

        # initialize extractions list and faces count
        faces = []
        faces_org = []

        faces_count = 0

        # iterate through faces found
        for face in results[0][0]:
            
            # check confidence level
            face_confidence = face[2]
            
            # if confidence level adequate:
            if face_confidence > min_confidence:

                # add face to count
                faces_count += 1

                # draw box around face in output_image
                bbox = face[3:]

                x1a = int((bbox[0] * image_width))
                y1a = int((bbox[1] * image_height))
                x2a = int((bbox[2] * image_width))
                y2a = int((bbox[3] * image_height))

                fourth_face_width_over_two = int((x2a - x1a)/6)
                fourth_face_height_over_two = int((y1a - y2a)/6)

                x1 = x1a - fourth_face_width_over_two
                y1 = y1a + fourth_face_height_over_two
                x2 = x2a + fourth_face_width_over_two
                y2 = y2a - fourth_face_height_over_two


                cv2.rectangle(output_image, pt1=(x1a, y1a), pt2=(x2a, y2a), color=(0, 255, 0), thickness=image_width//200)

                # extract face and add to extractions list
                cropped_img = image[y1:y2, x1:x2]

                cropped_img_org = image[y1a:y2a, x1a:x2a]

                if not (cropped_img is None or cropped_img.size == 0):

                    faces.append(cropped_img)

                if not (cropped_img_org is None or cropped_img_org.size == 0):

                    faces_org.append(cropped_img_org)

        # if display flag on:
        if display:

            # display input image with and without boxes       
            plt.figure(figsize=[20,20])
            plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off')
            plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output");plt.axis('off')
            plt.show()

            # return input image with boxes, model results, list of extractions, list of very cropped extractions, and count of faces
            return output_image, results, faces, faces_org, faces_count

        # if display flag off:   
        else:

            # return input image with boxes, model results, list of extractions, list of very cropped extractions, and count of faces
            return output_image, results, faces, faces_org, faces_count
        

    def two_pass_face_detection(self, image, first_conf, second_conf, im):

        # run first pass of model on image
        final_image, output, extractions, extractions_org, face_count = self.cv_dnn_detect_faces(image, first_conf, display=False)

        # determine number of faces after initial pass
        initial_faces_count = len(extractions_org)

        # if no initial faces found, return 0
        if initial_faces_count == 0:

            return 0, 0

        # if faces found, run second pass
        else:

            final = []
            var = 0
            count = 0

            for i in extractions_org:
                if self.cv_dnn_detect_faces(i, second_conf, display=False)[4] > 0:

                    # add face extraction to final
                    final.append(extractions[count])

                    ## show final extraction
                    # plt.imshow(i)
                    # plt.axis('off')
                    # plt.show()

                    # save final face extraction
                    var +=1
                    output_directory = self.output_directory
                    os.makedirs(output_directory, exist_ok=True)
                    output_filename = f"{im}_{var}.jpg"
                    output_path = os.path.join(output_directory, output_filename)
                    cv2.imwrite(output_path, extractions[count])

                    count +=1

            # determine and final number of faces
            final_faces_count = len(final)

            return initial_faces_count, final_faces_count
        
    def run(self):
        image_directory = self.input_directory

        # define image count variable
        var = 0

        # iterate through image directory
        for filename in os.listdir(image_directory):

            # if file is image:
            if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):

                # determine image path
                file_path = os.path.join(image_directory, filename)
                
                # read image
                image = cv2.imread(file_path)
                
                # if image fails to load:
                if image is None:

                    # write error message
                    with open('my_file.txt', 'a') as file:
                        file.write(f"Failed to load image: {file_path}\n")

                    # move to next file
                    continue

                # increase image count variable
                var +=1

                # use image as input
                initial_count, final_count = self.two_pass_face_detection(image, 0.14, 0.99, var)
                
                # write face counts to file
                with open('my_file.txt', 'a') as file:
                    file.write(f"{var}\n")
                    file.write(f"{filename}: {initial_count} faces detected after first pass.\n")
                    file.write(f"{filename}: {final_count} faces detected after second pass.\n")

                

            # if file is not image
            else:

                # write error message
                with open('my_file.txt', 'a') as file:
                    file.write(f"Skipping non-image file: {filename}\n")
                