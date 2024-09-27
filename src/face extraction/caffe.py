import os
import cv2
import dlib
from time import time
import matplotlib.pyplot as plt
from datetime import datetime

prototxt_path = "/Users/carolinazubler/Library/Mobile Documents/com~apple~CloudDocs/Documents/Yale/Senior Year/Fall 2024/CPSC 490/opencv/deploy.prototxt"
caffe_model_path = "/Users/carolinazubler/Library/Mobile Documents/com~apple~CloudDocs/Documents/Yale/Senior Year/Fall 2024/CPSC 490/opencv/res10_300x300_ssd_iter_140000.caffemodel"

if not os.path.isfile(prototxt_path):
    raise FileNotFoundError(f"Prototxt file not found at {prototxt_path}")

if not os.path.isfile(caffe_model_path):
    raise FileNotFoundError(f"Caffe model file not found at {caffe_model_path}")

opencv_dnn_model = cv2.dnn.readNetFromCaffe(prototxt=prototxt_path,
                                            caffeModel=caffe_model_path)

def cvDnnDetectFaces(image, count, opencv_dnn_model, min_confidence, display = True):

    image_height, image_width, _ = image.shape

    output_image = image.copy()

    preprocessed_image = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300),
                                               mean=(104.0, 117.0, 123.0), swapRB=False, crop=False)

    opencv_dnn_model.setInput(preprocessed_image)

    results = opencv_dnn_model.forward()    

    figures = []

    for face in results[0][0]:
        
        face_confidence = face[2]
        
        if face_confidence > min_confidence:

            count += 1

            bbox = face[3:]

            x1 = int(bbox[0] * image_width)
            y1 = int(bbox[1] * image_height)
            x2 = int(bbox[2] * image_width)
            y2 = int(bbox[3] * image_height)

            cv2.rectangle(output_image, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=image_width//200)

            cropped_img = image[y1:y2, x1:x2]
            if not (cropped_img is None or cropped_img.size == 0):

                figures.append(cropped_img)

    if display:
        
        plt.figure(figsize=[20,20])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off')
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output");plt.axis('off')
        plt.show()
        return output_image, results, figures, count
        
    else:
        
        return output_image, results, figures, count
    
#######################################################################################
    
image = cv2.imread('IMG4.jpg')
final_image, output, extractions, face_count = cvDnnDetectFaces(image, 0, opencv_dnn_model, 0.13, display=True)

final = []

if len(extractions) == 0:
    print("FAILED")

else:

    for i in extractions:
        if cvDnnDetectFaces(i, 0, opencv_dnn_model, 0.99, display=False)[3] > 0:
            final.append(i)
    
    x = len(final)

    var = 0

    with open('my_file.txt', 'a') as file:
        file.write(f"{x}\n")

    # for i in final:

        # plt.imshow(i)
        # plt.axis('off')
        # plt.show()

        # var +=1
        # output_directory = "wini_data"
        # os.makedirs(output_directory, exist_ok=True)
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # output_filename = f"output_{timestamp}_{var}.jpg"
        # output_path = os.path.join(output_directory, output_filename)
        # cv2.imwrite(output_path, i)
        