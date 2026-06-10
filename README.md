# codalpha_handwritten_recognition_ML

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load Dataset
digits = load_digits()

X = digits.data
y = digits.target

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
print("Training Started...")

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Training Completed")
print("Accuracy:", round(accuracy * 100, 2), "%")

# Display Sample Prediction
index = 9

plt.imshow(digits.images[index], cmap="gray")
plt.title(
    f"Actual: {y_test[index]}  Predicted: {y_pred[index]}"
)
plt.axis("off")
plt.show()
