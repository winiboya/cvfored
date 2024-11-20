import os
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPool2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D,
)


class GazeDetectionModel:
    """
    Class for training and evaluating a gaze detection model.
    """

    def __init__(self, model_path, train_dir, valid_dir):

        self.model_path = model_path
        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.model = self._get_model()
        self.train_generator, self.valid_generator = self._get_data_generators()

    def _get_model(self):
        """
        Returns the saved gaze detection model if it exists, otherwise creates a new model.
        """
        if os.path.exists(self.model_path):
            return load_model(self.model_path)


        base_model = ResNet50(
            weights="imagenet", include_top=False, input_shape=(224, 224, 3)
        )

        base_model.trainable = False

        for layer in base_model.layers[-10:]:
            layer.trainable = True

        # create model with ResNet50 base model and custom top layers
        model = Sequential(
            [
                base_model,
                GlobalAveragePooling2D(),
                Dense(1024, activation="relu"),
                BatchNormalization(),
                Dropout(0.5),
                Dense(1, activation="sigmoid"),
            ]
        )

        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        

        return model

    def _get_data_generators(self):
        """
        Returns the training and validation data generators.
        """
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
        )
        valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode="binary",
            shuffle=True,
            classes=["not_focused", "focused"],
        )
    


        valid_generator = valid_datagen.flow_from_directory(
            self.valid_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode="binary",
            shuffle=False,
            classes=["not_focused", "focused"],
        )
      
        
     
        return train_generator, valid_generator
    
    def _prepare_data_with_weights(self, generator):
            def generator_wrapper():
                for x, y in generator:
                    # Create a sample weight array where each element is the weight for the corresponding label
                    sample_weights = np.array([self.class_weights[label] for label in y])
                    yield tf.convert_to_tensor(x, dtype=tf.float32), tf.convert_to_tensor(y, dtype=tf.float32), tf.convert_to_tensor(sample_weights, dtype=tf.float32)
            return generator_wrapper()

    def train(self, steps_per_epoch=19, epochs=20, validation_steps=4):
        """
        Trains the gaze detection model.
        """
    
        
        
        
        # if validation loss does not decrease after 5 epochs, stop training
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
        
        # adjust class weights to handle class imbalance
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(self.train_generator.classes),
            y=self.train_generator.classes,
        )
        self.class_weights = dict(enumerate(class_weights))
        
        checkpoint = ModelCheckpoint(
            "best_model.keras",
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        )
        
        
        
        history = self.model.fit(
            self._prepare_data_with_weights(self.train_generator),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            verbose=1,
            validation_data=self._prepare_data_with_weights(self.valid_generator),
            validation_steps=len(self.valid_generator),
            callbacks=[early_stopping, checkpoint],
        )

        self._plot_history(history)
        self.model.save("gaze_detection_model.keras")

    def _plot_history(self, history):
        """
        Plots the accuracy and loss of the model during training.
        """
        
        for metric in ["accuracy", "loss"]:
            

            # Plot accuracy
            plt.plot(history.history[metric], label=f"Training {metric.capitalize()}")
            plt.plot(history.history[f"val_{metric}"], label=f"Validation {metric.capitalize()}")
            plt.title(f"Model {metric.capitalize()}")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.savefig(f"{metric}_plot.png")
            plt.clf()

    

    def predict_image(self, image_path):
        """
        Predicts the gaze of the person in the image.
        """
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = self.model.predict(img_array)[0]
        
        return "focused" if prediction > 0.5 else "not_focused", float(prediction)


    def predict_image_with_labels(self, image_path, output_path, true_label=None):
        """
        Predicts the gaze of the person in the image, displays prediction and true label on image.
        """

        prediction, score = self.predict_image(image_path)
        img = load_img(image_path)
        title = f"Prediction: {prediction} (score: {score:.2f})"

        if true_label:
            title = f"True label: {true_label}, {title}"
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

      
        plt.title(title)
        plt.axis("off")
        plt.imshow(img)
        plt.savefig(output_path)
        plt.close()

    def make_predictions(self, image_dir, output_images=False):
        """
        Makes gaze predictions for all images in the given directory.
        """
        model = self.model

        if output_images and not os.path.exists("predictions"):
            os.makedirs("predictions")
            
        if output_images:
            for subdir in os.listdir(image_dir):
                subdir_path = os.path.join(image_dir, subdir)
                if os.path.isdir(subdir_path):
                    for image in os.listdir(subdir_path):
                        if image.endswith(".jpg"):
                            image_path = os.path.join(subdir_path, image)
                            face_number = os.path.splitext(image)[0] 
                            prediction, score = self.predict_image(image_path)
                            self.predict_image_with_labels(
                                image_path, f"predictions/{subdir}/{face_number}.jpg"
                            )
            
            
        else:
            with open("predictions.csv", 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(["Frame Number", "Face Number", "Prediction", "Score"])
                
                for subdir in os.listdir(image_dir):
                    subdir_path = os.path.join(image_dir, subdir)
                    if os.path.isdir(subdir_path):
                        for image in os.listdir(subdir_path):
                            if image.endswith(".jpg"):
                                image_path = os.path.join(subdir_path, image)
                                face_number = os.path.splitext(image)[0] 
                                prediction, score = self.predict_image(image_path)
                                csv_writer.writerow([subdir, face_number, prediction, score])

        # for image in os.listdir(image_dir):
        #     if image.endswith(".jpg"):
        #         image_path = os.path.join(image_dir, image)
        #         print(image_path)
        #         true_label = image_path.split("/")[-2]
        #         if output_images:
        #             self.predict_image_with_labels(
        #                 image_path, f"predictions/{true_label}-{image}"
        #             )
        #         else:
        #             prediction, score = self.predict_image(image_path)
        #             file.write(
        #                 f"File: {image}, True label: {true_label}, Prediction: {prediction}, Score: {score}\n"
        #             )



    def evaluate(self):
        """
        Evaluates the gaze detection model on the validation data.
        """
        valid_data = self._prepare_data_with_weights(self.valid_generator)
        loss, accuracy = self.model.evaluate(valid_data, verbose=1, steps=len(self.valid_generator))
        print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")

    def get_classification_report(self):
        """
        Returns the classification report for the model.
        """
        class_indices = self.valid_generator.class_indices
        index_to_class = {v: k for k, v in class_indices.items()}
        y_pred = (self.model.predict(self.valid_generator) > 0.5).astype("int32")
        y_true = self.valid_generator.classes
        y_pred_labels = [index_to_class[int(i)] for i in y_pred]
        y_true_labels = [index_to_class[int(i)] for i in y_true]
        cr = classification_report(y_true_labels, y_pred_labels)
        cr_string = f"Classification Report:\n{cr}"
        return cr_string

    def get_confusion_matrix(self):
        """
        Returns the confusion matrix for the model.
        """
        class_indices = self.valid_generator.class_indices
        index_to_class = {v: k for k, v in class_indices.items()}
        y_pred = (self.model.predict(self.valid_generator) > 0.5).astype("int32")
        y_true = self.valid_generator.classes
        y_pred_labels = [index_to_class[int(i)] for i in y_pred]
        y_true_labels = [index_to_class[int(i)] for i in y_true]
        class_labels = [index_to_class[i] for i in range(len(index_to_class))]

        cm = confusion_matrix(y_true_labels, y_pred_labels)
        cm_string = (
            f"Confusion Matrix:\n"
            f"{'':<15}{class_labels[0]:<15}{class_labels[1]:<15}\n"
            f"{class_labels[0]:<15}{cm[0, 0]:<15}{cm[0, 1]:<15}\n"
            f"{class_labels[1]:<15}{cm[1, 0]:<15}{cm[1, 1]:<15}\n"
        )

        return cm_string

    def output_valid_predictions(self, write_to_file=True, output_images=False):
        """
        Outputs the predictions for the validation data.
        """
        if write_to_file:
            self.make_predictions(
                "../../test_faces/valid/focused", "predictions.txt", output_images=False
            )
            self.make_predictions(
                "../../test_faces/valid/not_focused",
                "predictions.txt",
                output_images=False,
            )

        if output_images:
            self.make_predictions(
                "../../test_faces/valid/focused", "predictions.txt", output_images=True
            )
            self.make_predictions(
                "../../test_faces/valid/not_focused",
                "predictions.txt",
                output_images=True,
            )

    def output_valid_analytics(self):
        """
        Outputs the classification report and confusion matrix for the model.
        """
        print(self.evaluate())
        print(self.get_classification_report())
        print(self.get_confusion_matrix())


def main():
    model = GazeDetectionModel(
        "best_model.keras", "../../test_faces/train", "../../test_faces/valid"
    )
    
    # def verify_images(directory):
    #     for root, dirs, files in os.walk(directory):
    #         for filename in files:
    #             if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
    #                 try:
    #                     img_path = os.path.join(root, filename)
    #                     img = load_img(img_path)
    #                 except Exception as e:
    #                     print(f"Error loading {img_path}: {str(e)}")
    #             else:
    #                 print(f"Skipping {filename} in {root}")

    # # Add this to your __init__
    # verify_images(model.train_dir)
    # verify_images(model.valid_dir)
    # model.train()
    # model.output_valid_analytics()

    #  print class indices
    # print(model.model.class_indices)

    model.train()
    # model.evaluate()
    # model.predict_image_with_labels("../../test_faces/valid/not_focused/video1-frame9-face2.jpg", "lol_prediction.jpg")
    # train_focused_names = os.listdir(model.train_focused_dir)
    # print(train_focused_names[:10])
    # print(f"Number of training focused images: {len(os.listdir(model.train_focused_dir))}")

    # output_dir = 'predictions'
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # output_count = 0
    # for image in os.listdir("../../test_faces/valid/focused"):
    #     if image.endswith(".jpg"):
    #         image_path = os.path.join("../../test_faces/valid/focused", image)
    #         output_path = os.path.join(output_dir, f"prediction{output_count}.jpg")
    #         model.predict_image(image_path, output_path)
    #         output_count += 1
    # for image in os.listdir("../../test_faces/valid/not_focused"):
    #     if image.endswith(".jpg"):
    #         image_path = os.path.join("../../test_faces/valid/not_focused", image)
    #         output_path = os.path.join(output_dir, f"prediction{output_count}.jpg")
    #         model.predict_image(image_path, output_path)
    # # # #         output_count += 1

    # # print class indices


if __name__ == "__main__":
    main()
    
    
