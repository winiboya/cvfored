import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
from itertools import cycle
from sklearn.metrics import classification_report
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50

from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPool2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D
)
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score


class GazeDetectionModel:
    def __init__(self):

        # Directory with focused training data
        self.train_focused_dir = os.path.join("../../test_faces/train/focused")

        # Directory with not focused training data
        self.train_not_focused_dir = os.path.join("../../test_faces/train/not_focused")

        # Directory with focused validation data
        self.valid_focused_dir = os.path.join("../../test_faces/valid/focused")

        # Directory with not focused validation data
        self.valid_not_focused_dir = os.path.join("../../test_faces/valid/not_focused")

        self.model = self.get_model()
        self.train_generator, self.valid_generator = self.get_data_generators()

    def get_model(self):
        """
        Returns the gaze detection model.
        """
        if os.path.exists("gaze_detection_model.h5"):

            return load_model("gaze_detection_model.h5")

        else:

            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

            base_model.trainable = False
            
            for layer in base_model.layers[-10:]:
                layer.trainable = True

            model = Sequential([
                base_model,
                GlobalAveragePooling2D(),
                BatchNormalization(),
                Dense(1024, activation="relu", kernel_regularizer=l2(0.01)),
                Dropout(0.5),
                Dense(1, activation="sigmoid") #updated
                
            ])

            # # Layer 1
            # model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)))
            # model.add(BatchNormalization())
            # model.add(MaxPool2D(2, 2))

            # # Layer 2
            # model.add(Conv2D(64, (3, 3), activation="relu"))
            # model.add(BatchNormalization())
            # model.add(MaxPool2D(2, 2))

            # # Layer 3
            # model.add(Conv2D(128, (3, 3), activation="relu"))
            # model.add(BatchNormalization())
            # model.add(MaxPool2D(2, 2))

            # # Layer 4
            # model.add(Conv2D(256, (3, 3), activation="relu"))
            # model.add(BatchNormalization())
            # model.add(MaxPool2D(2, 2))

            # # Layer 5
            # model.add(Conv2D(256, (3, 3), activation="relu"))
            # model.add(BatchNormalization())
            # model.add(MaxPool2D(2, 2))

            # model.add(Flatten())

            # # Fully connected layer
            # model.add(Dense(512, activation="relu", kernel_regularizer=l2(0.01)))
            # # model.add(Dropout(0.5))

            # # Output layer: binary classification
            # model.add(Dense(1, activation="sigmoid"))

            model.compile(
                optimizer=tf.optimizers.Adam(learning_rate=0.0001),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )

            return model

    def get_data_generators(self):
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
            "../../test_faces/train",
            target_size=(224, 224),
            batch_size=32,
            class_mode="binary",
            shuffle=True,
        )

        valid_generator = valid_datagen.flow_from_directory(
            "../../test_faces/valid",
            target_size=(224, 224),
            batch_size=32,
            class_mode="binary",
            shuffle=False,
        )
        return train_generator, valid_generator

    def train(self):

        model = self.model

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )

        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(self.train_generator.classes),
            y=self.train_generator.classes,
        )
        class_weights = dict(enumerate(class_weights))

        # if validation loss does not decrease after 5 epochs, reduce learning rate by half
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
        )

        # model = load_model('gaze_detection_model.h5')
        # assign weights based on class imbalance

        history = self.model.fit(
            self.train_generator,
            steps_per_epoch=19,
            epochs=15,
            verbose=2,
            validation_data=self.valid_generator,
            validation_steps=4,
            callbacks=[early_stopping, reduce_lr],
            class_weight=class_weights,
        )

        self.plot_history(history)

        self.model.save("gaze_detection_model.h5")

    def plot_history(self, history):
        # Plot accuracy
        plt.plot(history.history["accuracy"], label="Training Accuracy")
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
        plt.title("Model Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        # plt.show()
        plt.savefig("accuracy_plot.png")
        
        # clear the current figure
        plt.clf()

        # Plot loss
        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.title("Model Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("loss_plot.png")

    def evaluate(self):
        # self.model.evaluate(valid_generator)
        print("\nEvaluating the model on the validation data:")
        loss, accuracy = self.model.evaluate(self.valid_generator, verbose=1)
        print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")

    def predict_image(self, image_path):
        """
        Predicts the gaze of
        the person in the image.
        """
        # Load and preprocess the image
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = self.model.predict(img_array)[0]

        if prediction < 0.5:
            return "focused", prediction
        else:
            return "not_focused", prediction

    def make_predictions(self, image_dir, output_file):
        """
        Makes gaze predictions for all images in the given directory.
        """
        file = open(output_file, "a")
        model = self.model
        for image in os.listdir(image_dir):
            if image.endswith(".jpg"):
                image_path = os.path.join(image_dir, image)
                true_label = image_path.split("/")[-2]
                prediction, score = self.predict_image(image_path)
                file.write(
                    f"File: {image}, True label: {true_label}, Prediction: {prediction}, Score: {score}\n"
                )

        file.close()

    def predict_image_with_labels(self, image_path, output_path):
        """
        Predicts the gaze of the person in the image, displays prediction and true label on image.
        """

        true_label = image_path.split("/")[-2]
        prediction, score = self.predict_image(image_path)

        plt.title(f"True label: {true_label}, Prediction: {prediction}")

        img = tf.keras.preprocessing.image.load_img(image_path)

        plt.axis("off")
        plt.imshow(img)
        plt.savefig(output_path)


def main():
    model = GazeDetectionModel()

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
    # # # # # #         output_count += 1
    # model.make_predictions("../../test_faces/valid/focused", "predictions.txt")
    # model.make_predictions("../../test_faces/valid/not_focused", "predictions.txt")
    # print class indices
    
    
    # class_indices = model.valid_generator.class_indices
    # index_to_class = {v: k for k, v in class_indices.items()}
    # y_pred = (model.model.predict(model.valid_generator) > 0.5).astype("int32")
    # y_true = model.valid_generator.classes
    # y_pred_labels = [index_to_class[int(i)] for i in y_pred]  
    # y_true_labels = [index_to_class[int(i)] for i in y_true]

    # print(classification_report(y_true_labels, y_pred_labels))

if __name__ == "__main__":
    main()
