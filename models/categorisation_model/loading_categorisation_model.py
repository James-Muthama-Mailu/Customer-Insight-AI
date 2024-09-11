# Import the load_model function to load a pre-trained Keras models
from tensorflow.keras.models import load_model

# Import the train_model function that trains a new models
# from models.categorisation_model.training_and_evaluating_categorisation_model import train_model


def load_or_train_model():
    model_path = (r'C:\Users\James Muthama\ICS Project 1\ICS Project '
                  r'1\models\categorisation_model\CustomerInsightAI.keras')

    try:
        model = load_model(model_path)

    except Exception as e:
        print(f"Error loading model: {str(e)}")

    return model

