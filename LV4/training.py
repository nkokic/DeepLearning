from model import build_lstm_model_basic
import keras
from dataloader import X_train_final, y_train_final, X_val, y_val, X_test, y_test_cat
from matplotlib import pyplot as plt

# 1. Build the model
model = build_lstm_model_basic()

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint('LV4/best_model.h5', monitor='val_loss', save_best_only=True)
]

# 2. Train the model
history = model.fit(X_train_final, y_train_final, epochs=50, validation_data=(X_val, y_val), callbacks=callbacks)

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')

plt.show()