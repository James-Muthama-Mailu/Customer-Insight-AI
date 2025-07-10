# models/categorisation_model/loading_categorisation_model.py
from tensorflow.keras.models import load_model
import os

# Import the train_model function that trains a new model
# from models.categorisation_model.training_and_evaluating_categorisation_model import train_model

def load_or_train_model():
    # Use a relative path for better portability
    relative_path = 'CustomerInsightAI_improved.keras'  # Updated filename
    model_path = os.path.join(os.path.dirname(__file__), relative_path)

    # Check if the file exists at the relative path
    if os.path.exists(model_path):
        try:
            model = load_model(model_path)
            print(f"Model loaded successfully from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            # Optionally, you can attempt to train the model if loading fails
            # For now, we'll raise an exception
            raise RuntimeError(f"Failed to load the model: {e}")
    else:
        print(f"Model file not found at {model_path}")

        # If model file doesn't exist, you have two options:
        # 1. Raise an exception prompting the user to train or move the model
        # raise FileNotFoundError(f"File not found: filepath={model_path}. Please ensure the file is an accessible `.keras` zip file.")

        # 2. If you have a training function, you can train the model here.
        # Uncomment the import and use the train_model function as shown below:
        # model = train_model()
        # model.save(model_path)
        # print(f"New model trained and saved to {model_path}")
        # return model

        # For now, we'll raise an exception since we don't have the training function called
        raise FileNotFoundError(f"File not found: filepath={model_path}. Please ensure the file is an accessible `.keras` zip file.")