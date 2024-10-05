from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.utils import load_img

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np

import os

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
        __init__(model_path, train_data_path, batch_size, target_size, epochs):
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
    def __init__(self, model_path='gaze_detection_model.h5', train_data_path='../testdata/gaze/', batch_size=32, target_size=(224, 224), epochs=10):
        """
        Initializes the GazeDetectionModel with the specified parameters.
        
        Args:
            model_path (str): The path to the saved model file.
            train_data_path (str): The path to the directory containing training data.
            batch_size (int): The batch size for training.
            target_size (tuple): The target size for input images.
            epochs (int): The number of epochs to train the model.
        """
        
        self.model_path = model_path
        self.train_data_path = train_data_path
        self.batch_size = batch_size
        self.target_size = target_size
        self.epochs = epochs
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
        x = GlobalAveragePooling2D()(x)  # Convert features to a vector
        x = Dense(1024, activation='relu')(x)  # Add a fully connected layer
        predictions = Dense(2, activation='softmax')(x)  # Output layer for 3 classes

        # Create the full model
        model = Model(inputs=base_model.input, outputs=predictions)


        # Freeze the base model layers
        for layer in base_model.layers:
            layer.trainable = False

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)
        
        return model

    def create_data_generators(self):
        """
        Creates data generators for loading and augmenting training data.
        
        Returns:
            ImageDataGenerator: The training data generator.
            ImageDataGenerator: The validation data generator.
        """
    
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            rotation_range=20,  # Rotate images
            width_shift_range=0.2,  # Shift images horizontally
            height_shift_range=0.2,  # Shift images vertically
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],  # Random brightness
            validation_split=0.2
        )

        train_generator = train_datagen.flow_from_directory(
            '../testdata/gaze/',
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            subset='training'
        )

        validation_generator = train_datagen.flow_from_directory(
            '../testdata/gaze/',
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            subset='validation'
        )
        return train_generator, validation_generator

    def train_model(self):
        """ 
        Trains the gaze detection model using the training data.
        """
        train_generator, validation_generator = self.create_data_generators()
        self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // validation_generator.batch_size,
            epochs=self.epochs
        )
    
    def save_model(self):
        """
        Saves the trained model to a file.
        """
        self.model.save(self.model_path)
        print(f"Model saved to {self.model_path}")
        
    def save_correct_predictions(self, generator, output_dir, num_images=5):
        """
        Saves images that the model correctly classified with true and predicted labels as overlay text.
        
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
        
        correct_images = []
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
                
                if true_label_idx == pred_label_idx:
                    # Store correctly predicted images and labels
                    correct_images.append(images[i])
                    correct_labels.append(idx_to_class[true_label_idx])
                    predicted_labels.append(idx_to_class[pred_label_idx])
                
                # Stop if we have enough correct images
                if len(correct_images) >= num_images:
                    break
                    
            if len(correct_images) >= num_images:
                break
        
        # Save correctly predicted images with labels
        for i in range(len(correct_images)):
            img_array = (correct_images[i] * 255).astype(np.uint8)
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
            
            image_filename = os.path.join(output_dir, f"correct_prediction_{i+1}.png")
            pil_img.save(image_filename)
            print(f"Saved: {image_filename}")
    
    def predict_and_save_images(self, output_dir='correct_predictions', num_images=5):
        """
        Predicts and saves a specified number of correctly classified images with labels.
        
        Args:
            output_dir (str): The directory to save the images.
            num_images (int): The number of images to save.
        """
        print("Predicting and saving correct images...")
        _, validation_generator = self.create_data_generators()
        self.save_correct_predictions(validation_generator, output_dir, num_images)
        
    
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
        
    # Initialize the model with desired parameters
    gaze_detection = GazeDetectionModel(
        model_path='gaze_detection_model.h5',
        train_data_path='../testdata/gaze/',
        batch_size=32,
        target_size=(224, 224),
        epochs=10
    )

    # Run the training and saving process
    gaze_detection.run()
    
if __name__ == '__main__':
    main()
