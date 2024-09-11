from models.categorisation_model.categorisation_model import model
from models.categorisation_model.pre_processioning_categorisation_model_data import train_x, train_y, test_x, test_y
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np


def train_model():
    # Define an EarlyStopping callback to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    history = model.fit(train_x, train_y, epochs=100, batch_size=100, validation_split=0.2,
                        callbacks=[early_stopping, reduce_lr])
    model.save("CustomerInsightAI.keras")
    return model


def evaluate_model(model):
    # Evaluate the trained models with the test data to assess its performance
    # - test_x: input data for testing
    # - test_y: true output labels for testing
    test_loss, test_accuracy = model.evaluate(test_x, test_y)

    # Print the evaluation results
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    # Return the evaluation metrics
    return test_loss, test_accuracy


model = train_model()

evaluate_model(model)
