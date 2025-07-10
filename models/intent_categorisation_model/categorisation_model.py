import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from pre_processioning_categorisation_model_data import train_x, train_y, test_x, test_y


# Define the improved neural network model
def create_improved_model(input_dim, num_classes):
    model = Sequential([
        # Input layer - using Input() instead of input_shape parameter
        Input(shape=(input_dim,)),

        # First hidden layer - reduced size and added L2 regularization
        Dense(128, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.3),  # Reduced dropout rate

        # Second hidden layer
        Dense(64, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.4),

        # Third hidden layer - smaller size
        Dense(32, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.3),

        # Output layer
        Dense(num_classes, activation='softmax')
    ])

    return model


# Create the model
model = create_improved_model(train_x.shape[1], train_y.shape[1])

# Compile with a lower learning rate and different optimizer settings
optimizer = Adam(
    learning_rate=0.0005,  # Lower learning rate
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-7
)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']  # Remove top_3_accuracy as it's not available in this TensorFlow version
)

# Print model summary
model.summary()


# Alternative model with different architecture approach
def create_alternative_model(input_dim, num_classes):
    """
    Alternative architecture with residual-like connections and different regularization
    """
    model = Sequential([
        Input(shape=(input_dim,)),

        # First block
        Dense(256, activation=None),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.2),

        # Second block - gradually reducing size
        Dense(128, activation=None, kernel_regularizer=l2(0.005)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.3),

        # Third block
        Dense(64, activation=None, kernel_regularizer=l2(0.005)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.2),

        # Output layer
        Dense(num_classes, activation='softmax')
    ])

    return model

# Uncomment to use alternative model
# model = create_alternative_model(train_x.shape[1], train_y.shape[1])
# model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])