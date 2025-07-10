import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from categorisation_model import model
from pre_processioning_categorisation_model_data import train_x, train_y, test_x, test_y


def train_improved_model():
    """
    Improved training function with better callbacks and validation strategy
    """
    # Enhanced callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,  # Increased patience
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # Less aggressive reduction
        patience=8,  # Increased patience
        min_lr=1e-6,
        verbose=1
    )

    # Save best model checkpoint
    checkpoint = ModelCheckpoint(
        'best_model.keras',
        monitor='val_accuracy',  # Monitor validation accuracy
        save_best_only=True,
        verbose=1
    )

    # Train the model with improved parameters
    history = model.fit(
        train_x, train_y,
        epochs=200,  # More epochs but with early stopping
        batch_size=64,  # Smaller batch size for better generalization
        validation_split=0.25,  # Larger validation split
        callbacks=[early_stopping, reduce_lr, checkpoint],
        verbose=1,
        shuffle=True  # Ensure data is shuffled
    )

    # Save the final model
    model.save("CustomerInsightAI_improved.keras")

    return model, history


def evaluate_improved_model(model, history):
    """
    Enhanced evaluation with detailed metrics and visualizations
    """
    # Evaluate on test data
    test_loss, test_accuracy = model.evaluate(test_x, test_y, verbose=0)

    print(f"\n=== Model Evaluation Results ===")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Get predictions for detailed analysis
    predictions = model.predict(test_x, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(test_y, axis=1)

    # Print classification report
    print(f"\n=== Classification Report ===")
    print(classification_report(true_classes, predicted_classes))

    # Print training history summary
    print_training_summary(history)

    return test_loss, test_accuracy, predictions


def print_training_summary(history):
    """
    Print a summary of training metrics instead of plotting
    """
    print(f"\n=== Training Summary ===")

    # Final metrics
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]

    print(f"Final Training Accuracy: {final_train_acc:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"Final Training Loss: {final_train_loss:.4f}")
    print(f"Final Validation Loss: {final_val_loss:.4f}")

    # Best metrics
    best_val_acc = max(history.history['val_accuracy'])
    best_val_acc_epoch = history.history['val_accuracy'].index(best_val_acc) + 1

    print(f"\nBest Validation Accuracy: {best_val_acc:.4f} (Epoch {best_val_acc_epoch})")
    print(f"Total Epochs Trained: {len(history.history['loss'])}")

    # Check for overfitting
    acc_gap = final_train_acc - final_val_acc
    loss_gap = final_val_loss - final_train_loss

    print(f"\nOverfitting Analysis:")
    print(f"Accuracy Gap (Train - Val): {acc_gap:.4f}")
    print(f"Loss Gap (Val - Train): {loss_gap:.4f}")

    if acc_gap > 0.1:
        print("⚠️  High accuracy gap suggests overfitting")
    if loss_gap > 1.0:
        print("⚠️  High loss gap suggests overfitting")


def analyze_data_distribution():
    """
    Analyze the distribution of classes in your dataset
    """
    print("=== Data Distribution Analysis ===")
    print(f"Training samples: {len(train_x)}")
    print(f"Test samples: {len(test_x)}")
    print(f"Feature dimension: {train_x.shape[1]}")
    print(f"Number of classes: {train_y.shape[1]}")

    # Check class distribution
    train_class_counts = np.sum(train_y, axis=0)
    test_class_counts = np.sum(test_y, axis=0)

    print(f"\nClass distribution in training set:")
    for i, count in enumerate(train_class_counts):
        print(f"Class {i}: {int(count)} samples ({count / len(train_y) * 100:.1f}%)")

    print(f"\nClass distribution in test set:")
    for i, count in enumerate(test_class_counts):
        print(f"Class {i}: {int(count)} samples ({count / len(test_y) * 100:.1f}%)")

    # Check for class imbalance
    min_class_count = np.min(train_class_counts[train_class_counts > 0])  # Only non-zero classes
    max_class_count = np.max(train_class_counts)
    imbalance_ratio = max_class_count / min_class_count if min_class_count > 0 else float('inf')

    print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}")
    if imbalance_ratio > 5:
        print("⚠️  Significant class imbalance detected. Consider using class weights or resampling.")

    # Check for data leakage issue
    train_classes_present = set(np.where(train_class_counts > 0)[0])
    test_classes_present = set(np.where(test_class_counts > 0)[0])

    classes_only_in_train = train_classes_present - test_classes_present
    classes_only_in_test = test_classes_present - train_classes_present

    if classes_only_in_train or classes_only_in_test:
        print(f"\n🚨 CRITICAL DATA SPLIT ISSUE DETECTED:")
        if classes_only_in_train:
            print(f"Classes only in training: {sorted(classes_only_in_train)}")
        if classes_only_in_test:
            print(f"Classes only in test: {sorted(classes_only_in_test)}")
        print("This will cause 0% accuracy! You need to fix your data splitting.")


# Run the improved training and evaluation
if __name__ == "__main__":
    # Analyze data first
    analyze_data_distribution()

    # Train the model
    print("\n=== Starting Training ===")
    trained_model, training_history = train_improved_model()

    # Evaluate the model
    print("\n=== Starting Evaluation ===")
    evaluate_improved_model(trained_model, training_history)