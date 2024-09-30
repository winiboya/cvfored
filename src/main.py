from gaze_detection_model import GazeDetectionModel

def main():
    print("Starting the training process...")
    # Initialize the model with desired parameters
    model = GazeDetectionModel(
        model_path='gaze_detection_model.h5',
        train_data_path='../testdata/gaze/',
        batch_size=32,
        target_size=(224, 224),
        epochs=10
    )

    # Run the training and saving process
    # model.run()
    model.predict_and_save_images('output/', 10)
    # model.predict_and_save_single_image("image.png", "not_focused")

if __name__ == '__main__':
    main()
