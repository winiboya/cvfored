from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.utils import load_img
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight


from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np

import os
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class LogPredictionsCallback(Callback):
    def __init__(self, generator, log_file_path):
        """
        Initializes the callback.
        
        Args:
            generator (ImageDataGenerator): The data generator used for predictions.
            log_file_path (str): Path to save the log file.
        """
        self.generator = generator
        self.log_file_path = log_file_path
        self.all_true_labels = []
        self.all_pred_labels = []
        
    
    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch to log predictions and metrics.
        
        Args:
            epoch (int): The index of the epoch.
            logs (dict): The logs dictionary for the epoch.
        """
        self.all_true_labels.clear()
        self.all_pred_labels.clear()

        # Open log file to save predictions
        with open(self.log_file_path, 'a') as log_file:
            log_file.write(f"Epoch {epoch + 1}\n")
            log_file.write("File Name, True Label, Predicted Label\n")

            # Iterate over all batches to collect predictions and true labels
            for batch_idx, (images, labels) in enumerate(self.generator):
                predictions = self.model.predict(images)

                for i in range(len(images)):
                    # File name and true label from the generator
                    file_name = self.generator.filenames[batch_idx * self.generator.batch_size + i]
                    true_label_name = os.path.split(os.path.dirname(file_name))[-1]

                    # Convert predicted label index to class name
                    pred_label_idx = int(predictions[i] > 0.5)
                    pred_label_name = list(self.generator.class_indices.keys())[pred_label_idx]

                    probability = predictions[i][0]
                    
                    log_file.write(f"probability: {probability}\n")

                    # Log predictions to file
                    log_file.write(f"{file_name}, {true_label_name}, {pred_label_name}\n")

                    # Collect true and predicted labels for metrics
                    self.all_true_labels.append(self.generator.class_indices[true_label_name])
                    self.all_pred_labels.append(pred_label_idx)

                # Stop if all samples have been processed
                if (batch_idx + 1) * self.generator.batch_size >= self.generator.samples:
                    break

        # Log metrics at the end of the epoch
        # self.log_metrics(epoch)

    def log_metrics(self, epoch):
        """
        Logs metrics (accuracy, confusion matrix, classification report) at the end of the epoch.
        
        Args:
            epoch (int): The index of the epoch.
        """
        cm = confusion_matrix(self.all_true_labels, self.all_pred_labels)
        report = classification_report(
            self.all_true_labels, self.all_pred_labels, target_names=list(self.generator.class_indices.keys()), zero_division=0
        )
        accuracy = np.mean(np.array(self.all_true_labels) == np.array(self.all_pred_labels)) * 100

        # Save metrics to a separate file
        metrics_file = f'metrics_epoch_{epoch + 1}.txt'
        with open(metrics_file, 'w') as log_file:
            log_file.write(f"\nMetrics for Epoch {epoch + 1}:\n")
            log_file.write(f"Accuracy: {accuracy:.2f}%\n")
            log_file.write("Confusion Matrix:\n")
            log_file.write(f"{cm}\n")
            log_file.write("\nClassification Report:\n")
            log_file.write(f"{report}\n")

        print(f"Metrics for Epoch {epoch + 1} logged to {metrics_file}")

# class LogPredictionsCallback(Callback):
#     def __init__(self, generator, validation_generator, log_file_path):
#         """
#         Initializes the callback.
        
#         Args:
#             generator (ImageDataGenerator): The training data generator used for predictions.
#             validation_generator (ImageDataGenerator): The validation data generator used for predictions.
#             log_file_path (str): Path to the file where predictions will be saved.
#         """
#         self.generator = generator
#         self.validation_generator = validation_generator
#         self.log_file_path = log_file_path
        
#     def calculate_metrics(self, predictions, labels, class_indices):
#         """
#         Calculates classification metrics such as accuracy, false positives, false negatives, etc.
        
#         Args:
#             predictions (np.array): Array of predicted labels.
#             labels (np.array): Array of true labels.
#             class_indices (dict): Class indices mapping.
            
#         Returns:
#             dict: Dictionary of calculated metrics.
#         """
#         cm = confusion_matrix(labels, predictions)
#         tn, fp, fn, tp = cm.ravel()
        
#         metrics = {
#             'total_predictions': len(labels),
#             'correct_class_1': tp,
#             'correct_class_2': tn,
#             'incorrect_class_1': fn,
#             'incorrect_class_2': fp,
#             'false_positives': fp,
#             'false_negatives': fn,
#             'accuracy': (tp + tn) / len(labels) * 100,
#             'precision_class_1': tp / (tp + fp) if tp + fp != 0 else 0,
#             'recall_class_1': tp / (tp + fn) if tp + fn != 0 else 0,
#             'precision_class_2': tn / (tn + fn) if tn + fn != 0 else 0,
#             'recall_class_2': tn / (tn + fp) if tn + fp != 0 else 0
#         }
#         return metrics

#     def log_metrics(self, metrics, data_type):
#         """
#         Logs the calculated metrics into the log file.
        
#         Args:
#             metrics (dict): Dictionary containing calculated metrics.
#             data_type (str): Either 'train' or 'validation' to indicate the type of data.
#         """
#         with open(self.log_file_path, 'w') as log_file:
#             log_file.write(f"{data_type.capitalize()} Metrics\n")
#             log_file.write(f"Total Predictions: {metrics['total_predictions']}\n")
#             log_file.write(f"Class 1 - Correct: {metrics['correct_class_1']} | Incorrect: {metrics['incorrect_class_1']}\n")
#             log_file.write(f"Class 2 - Correct: {metrics['correct_class_2']} | Incorrect: {metrics['incorrect_class_2']}\n")
#             log_file.write(f"False Positives: {metrics['false_positives']} | False Negatives: {metrics['false_negatives']}\n")
#             log_file.write(f"Accuracy: {metrics['accuracy']:.2f}%\n")
#             log_file.write(f"Class 1 Precision: {metrics['precision_class_1']:.2f} | Recall: {metrics['recall_class_1']:.2f}\n")
#             log_file.write(f"Class 2 Precision: {metrics['precision_class_2']:.2f} | Recall: {metrics['recall_class_2']:.2f}\n")
#             log_file.write("\n")
    
#     def on_epoch_end(self, epoch, logs=None):
#         """
#         Called at the end of each epoch to calculate and log metrics.
        
#         Args:
#             epoch (int): The index of the epoch.
#             logs (dict): The logs dictionary for the epoch.
#         """
#         log_file_path = self.log_file_path

#         # Initialize lists to collect predictions and labels for the train and validation sets
#         train_predictions, train_labels = [], []
#         val_predictions, val_labels = [], []

#         # Iterate over the batches of training data
#         for images, labels in self.generator:
#             preds = self.model.predict(images)
#             train_predictions.extend([1 if p > 0.7 else 0 for p in preds])
#             train_labels.extend(labels)

#         # Iterate over the batches of validation data
#         for images, labels in self.validation_generator:
#             preds = self.model.predict(images)
#             val_predictions.extend([1 if p > 0.7 else 0 for p in preds])
#             val_labels.extend(labels)

#         # Calculate training metrics
#         train_metrics = self.calculate_metrics(train_predictions, train_labels, self.generator.class_indices)
#         val_metrics = self.calculate_metrics(val_predictions, val_labels, self.validation_generator.class_indices)

#         # Log the metrics
#         with open(log_file_path, 'a') as log_file:
#             log_file.write(f"Epoch {epoch+1}\n")

#         self.log_metrics(train_metrics, "train")
#         self.log_metrics(val_metrics, "validation")


class GazeDetectionModel:
    """
    A class for training and using a gaze detection model based on a pre-trained ResNet50 model.
    
    Attributes:
        model_path (str): The path to the saved model file.
        train_data_path (str): The path to the directory containing training data.
        batch_size (int): The batch size for training.
        target_size (tuple): The target size for input images.
        epochs (int): The number of epochs to train the model.
        model (tf.keras.Model): The gaze detection model.
        
    Methods:
        __init__(model_path, train_data_path, validate_data_path, batch_size, target_size, epochs):
            Initializes the GazeDetectionModel with the specified parameters.
        load_or_create_model():
            Loads the saved model if it exists, otherwise creates a new model.
        create_model():
            Creates a new gaze detection model based on a pre-trained ResNet50 model.
        create_data_generators():
            Creates data generators for loading and augmenting training data.
        train_model():
            Trains the gaze detection model using the training data.
        save_model():
            Saves the trained model to a file.
        save_correct_predictions(generator, output_dir, num_images):
            Saves images that the model correctly classified with true and predicted labels as overlay text.
        predict_and_save_images(output_dir, num_images):
            Predicts and saves a specified number of correctly classified images with labels.
        run():
            Runs the training process for the gaze detection model.
        main():
            Main function for running the training process.
    
    """
    def __init__(self, train_data_path, validate_data_path, learning_rate, predictions_output, model_path='gaze_detection_model.h5', batch_size=32, target_size=(224, 224), epochs=10):
        """
        Initializes the GazeDetectionModel with the specified parameters.
        
        Args:
            model_path (str): The path to the saved model file.
            train_data_path (str): The path to the directory containing training data.
            validate_data_path (str): The path to the directory containing validation data.
            batch_size (int): The batch size for training.
            target_size (tuple): The target size for input images.
            epochs (int): The number of epochs to train the model.
        """
        
        self.model_path = model_path
        self.train_data_path = train_data_path
        self.validate_data_path = validate_data_path
        self.batch_size = batch_size
        self.target_size = target_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.predictions_output = predictions_output
        self.model = self.load_or_create_model()
        
    def load_or_create_model(self):
        """
        Loads the saved model if it exists, otherwise creates a new model.
        
        Returns:
            tf.keras.Model: The gaze detection model.
        """
        if os.path.exists(self.model_path):
            print(f"Loading saved model from {self.model_path}")
            return load_model(self.model_path)
        else:
            print("No saved model found. Creating a new model...")
            return self.create_model()
        
    def create_model(self):
        """
        Creates a new gaze detection model based on a pre-trained ResNet50 model.
        
        Returns:
            tf.keras.Model: The gaze detection model.
        """
        # Load the pre-trained model (excluding the top classification layer)
        base_model = ResNet50(weights='imagenet', include_top=False)

        # Add new classification layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x) 
        x = Dense(1024, activation='relu')(x)  
        predictions = Dense(1, activation='sigmoid')(x) 

        # Create the full model
        model = Model(inputs=base_model.input, outputs=predictions)


        # Freeze the base model layers
        for layer in base_model.layers:
            layer.trainable = False
            
        for layer in base_model.layers[-10:]:
            layer.trainable = True
        

        # Compile the model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'], run_eagerly=False)
        
        return model

    def model_summary(self, output_file="model_summary.txt"):
        """
        Outputs a summary of the gaze detection model.
        """
        train_generator, validation_generator = self.create_data_generators()
        
        # open new file to write to
        file = open(output_file, "w")
        
        file.write(f"Model summary for gaze detection model \"{self.model_path}'\n")
        
        file.write("--------------------\n")
        file.write(f"Paths:\n")
        file.write(f"  Train Data Path: {self.train_data_path}\n")
        file.write(f"  Validation Data Path: {self.validate_data_path}\n")        
            
        # Total number of samples
        file.write("--------------------\n")
        file.write(f"Total Training Samples: {train_generator.samples}\n")
        file.write(f"Total Validation Samples: {validation_generator.samples}\n")
            
        
        # Display training data class distribution
        file.write("--------------------\n")
        file.write("Training Data Class Distribution:\n")
        for class_label, index in train_generator.class_indices.items():
            class_size = np.sum(train_generator.classes == index)
            file.write(f"  {class_label}: {class_size} samples\n")

        # Display validation data class distribution
        file.write("--------------------\n")
        file.write("Validation Data Class Distribution:\n")
        for class_label, index in validation_generator.class_indices.items():
            class_size = np.sum(validation_generator.classes == index)
            file.write(f"  {class_label}: {class_size} samples\n")

       

        # Compute and print class weights (useful for imbalanced data)
        file.write("--------------------\n")
        file.write("Class Weights:\n")
        class_weights = class_weight.compute_class_weight(
            class_weight='balanced', 
            classes=np.unique(train_generator.classes), 
            y=train_generator.classes
        )
        for i, weight in enumerate(class_weights):
            class_name = list(train_generator.class_indices.keys())[i]
            file.write(f"  {class_name}: {weight}\n")

        file.write("--------------------\n")
        file.write("Training Parameters:\n")
        file.write(f"  Batch Size: {self.batch_size}\n")
        file.write(f"  Epochs: {self.epochs}\n")
        file.write(f"  Steps per Epoch: {train_generator.samples // self.batch_size}\n")
        file.write(f"  Validation Steps: {validation_generator.samples // self.batch_size}\n")
        
        file.close()

       
        
            
        

    def create_data_generators(self):
        """
        Creates data generators for loading and augmenting training data.
        
        Returns:
            ImageDataGenerator: The training data generator.
            ImageDataGenerator: The validation data generator.
        """
    
        train_datagen = ImageDataGenerator(
            rescale=1/255,
            shear_range=0.2,
            zoom_range=0.2,
            rotation_range=20,  # Rotate images
            width_shift_range=0.2,  # Shift images horizontally
            height_shift_range=0.2,  # Shift images vertically
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],  # Random brightness        
            )

        train_generator = train_datagen.flow_from_directory(
            self.train_data_path,
            target_size=(224, 224),
            batch_size=self.batch_size,
            class_mode='binary',
        )    
        
                
        validation_datagen = ImageDataGenerator(rescale=1/255)

        validation_generator = validation_datagen.flow_from_directory(
            self.validate_data_path,
            target_size=(224, 224),
            batch_size=32,
            class_mode='binary',
        )
        
        return train_generator, validation_generator

    def train_model(self):
        """ 
        Trains the gaze detection model using the training data.
        """
        train_generator, validation_generator = self.create_data_generators()
        
        # Initialize the custom callback to log predictions
        # Initialize the custom callback with both generators
        log_file_path = self.predictions_output
        log_predictions_callback = LogPredictionsCallback(
            generator=train_generator,
            # validation_generator=validation_generator,
            log_file_path=log_file_path
    
        )
        

        
        # Assign weights to classes
        class_weights = class_weight.compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(train_generator.classes),  
        y=train_generator.classes 
    )
        

        self.model.fit(
            train_generator,
            steps_per_epoch= train_generator.samples // self.batch_size,
            validation_data=validation_generator,
            validation_steps= validation_generator.samples // self.batch_size,
            epochs=self.epochs,
            verbose=2,
            callbacks=[log_predictions_callback],
            class_weight=dict(enumerate(class_weights))
        )
    
    def save_model(self):
        """
        Saves the trained model to a file.
        """
        self.model.save(self.model_path)
        print(f"Model saved to {self.model_path}")
        
    def save_prediction_images(self, generator, output_dir, num_images=5):
        """
        Saves images that the model classified with true and predicted labels as overlay text.
        
        Args:
            generator (ImageDataGenerator): The data generator to use for prediction.
            output_dir (str): The directory to save the images.
            num_images (int): The number of images to save.
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Get class indices
        class_indices = generator.class_indices
        idx_to_class = {v: k for k, v in class_indices.items()}
        
        predicted_images = []
        correct_labels = []
        predicted_labels = []

        # Iterate over batches of images and labels
        for images, labels in generator:
            # Generate predictions for the batch
            predictions = self.model.predict(images)
            
            # Iterate over each image in the batch
            for i in range(len(images)):
                true_label_idx = np.argmax(labels[i])
                pred_label_idx = np.argmax(predictions[i])
                
                # if true_label_idx == pred_label_idx:
                    # Store correctly predicted images and labels
                predicted_images.append(images[i])
                correct_labels.append(idx_to_class[true_label_idx])
                predicted_labels.append(idx_to_class[pred_label_idx])
                
                # Stop if we have enough correct images
                if len(predicted_images) >= num_images:
                    break
                    
            if len(predicted_images) >= num_images:
                break
        
        # Save correctly predicted images with labels
        for i in range(len(predicted_images)):
            img_array = (predicted_images[i] * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_array)
            draw = ImageDraw.Draw(pil_img)
            font_size = 20
            
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()
            
            text = f"True: {correct_labels[i]}\nPred: {predicted_labels[i]}"
            position = (10, 10)
            text_bbox = draw.textbbox(position, text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            draw.rectangle([position, (position[0] + text_width, position[1] + text_height * 2)], fill=(255, 255, 255))
            draw.text(position, text, fill=(0, 0, 0), font=font)
            
            image_filename = os.path.join(output_dir, f"prediction_{i+1}.png")
            pil_img.save(image_filename)
            print(f"Saved: {image_filename}")
    
    def predict_and_save_images(self, output_dir='predictions', num_images=5):
        """
        Predicts and saves a specified number of correctly classified images with labels.
        
        Args:
            output_dir (str): The directory to save the images.
            num_images (int): The number of images to save.
        """
        print("Predicting and saving correct images...")
        _, validation_generator = self.create_data_generators()
        self.save_prediction_images(validation_generator, output_dir, num_images)
        
    
    def run(self):
        """
        Runs the training process for the gaze detection model.
        """
        self.train_model()
        self.save_model()
        print("Training complete!")

def main():
    """
    Main function for running the training process.
    """
    
    parser = argparse.ArgumentParser(description="Train the gaze detection model.")

    parser.add_argument("--model_path", type=str, default='gaze_detection_model.h5', help="Path to save the trained model.")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the directory containing training data.")
    parser.add_argument("--validate_data_path", type=str, required=True, help="Path to the directory containing validation data.")
        
    args = parser.parse_args()
    # Initialize the model with desired parameters
  
    gaze_detection = GazeDetectionModel(
        model_path=args.model_path,
        train_data_path=args.train_data_path,
        validate_data_path=args.validate_data_path,
        predictions_output="predictions_log.txt",
        batch_size=32,
        target_size=(224, 224),
        learning_rate=0.0001,
        epochs=5
    )
    
    gaze_detection.run()


    # Run the training and saving process
    # gaze_detection.model_summary()

    # gaze_detection1.run()
    # gaze_detection.predict_and_save_images("output_predictions", 5)
    
if __name__ == '__main__':
    main()
