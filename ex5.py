import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

lfw = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
X = lfw.images
y = lfw.target
target_names = lfw.target_names
n_classes = len(target_names)

X = X / 255.0
X = X.reshape(-1, 50, 37, 1)
y = to_categorical(y, n_classes)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(50, 37, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(n_classes, activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy*100:.2f}%")
print(f"Test Loss: {loss:.4f}")

predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

for i in range(5):
    plt.imshow(X_test[i].reshape(50, 37), cmap='gray')
    plt.title(f"Predicted: {target_names[predicted_classes[i]]}\nActual: {target_names[true_classes[i]]}")
    plt.axis('off')
    plt.show()

model.save("face_recognition_cnn_model.h5")
