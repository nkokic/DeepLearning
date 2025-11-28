from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input, Normalization
from keras.optimizers import Adam

def build_lstm_model_basic():
    """
    Builds and compiles an LSTM model with default hyperparameters.
    
    Returns:
        A compiled Keras Sequential model.
    """
    # Default values (you can adjust if needed)
    units = 128
    dropout_rate = 0.1
    learning_rate = 0.0005

    model = Sequential([
        Input(shape=(128, 9)),                # Input shape: 128 time steps, 9 features
        Normalization(),                      # Normalization layer
        LSTM(units, return_sequences=True),   # First LSTM layer
        Dropout(dropout_rate),                # Dropout for regularization
        LSTM(units),                          # Second LSTM layer
        Dropout(dropout_rate),
        Dense(units, activation='relu'),      # Fully connected layer
        Dense(6, activation='softmax')        # Output layer for 6 activity classes
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model