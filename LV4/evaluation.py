from dataloader import X_test, y_test_cat, y_test
from keras.models import load_model
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

model = load_model('LV4/best_model.h5')

loss, accuracy = model.evaluate(X_test, y_test_cat)

# 4. Print results
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

y_pred_probs = model.predict(X_test)
y_pred = y_pred_probs.argmax(axis=1)

labels = [ "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

disp.plot(cmap=plt.cm.Blues)
plt.show()