import os
import cv2
import dlib
from time import time
import matplotlib.pyplot as plt

opencv_dnn_model = cv2.dnn.readNetFromCaffe(prototxt="deploy.prototxt",
                                            caffeModel="/Users/carolinazubler/Library/Mobile Documents/com~apple~CloudDocs/Documents/Yale/Senior Year/Fall 2024/CPSC 490/opencv/res10_300x300_ssd_iter_140000.caffemodel")

def cvDnnDetectFaces(image, count, opencv_dnn_model, min_confidence=0.13, display = True):

    image_height, image_width, _ = image.shape

    output_image = image.copy()

    preprocessed_image = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300),
                                               mean=(104.0, 117.0, 123.0), swapRB=False, crop=False)

    opencv_dnn_model.setInput(preprocessed_image)
    
    start = time()

    results = opencv_dnn_model.forward()    

    end = time()

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

            # cv2.rectangle(output_image, pt1=(x1, y1-image_width//20), pt2=(x1+image_width//16, y1),
            #               color=(0, 255, 0), thickness=-1)

            cropped_img = output_image[y1:y2, x1:x2]
            plt.imshow(cropped_img)
            plt.axis('off')
            plt.show()

            cv2.putText(output_image, text=str(round(face_confidence, 1)), org=(x1, y1-25), 
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=image_width//700,
                        color=(255,255,255), thickness=image_width//200)

    if display:

        cv2.putText(output_image, text='Time taken: '+str(round(end - start, 2))+' Seconds.', org=(10, 65),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=image_width//700,
                    color=(0,0,255), thickness=image_width//500)
        
        plt.figure(figsize=[20,20])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off')
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output");plt.axis('off')
        plt.show()
        print(count)
        
    else:
        
        print(count)
        return output_image, results
    
image = cv2.imread('crowd4.png')
cvDnnDetectFaces(image, 0, opencv_dnn_model, display=True)