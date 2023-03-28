import numpy as np
import pandas as pd
import os
from sklearn.metrics import classification_report
import seaborn as sn; sn.set(font_scale=1.4)
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

# Iніціалізація вибірки
class_names = ['apple_pie','deviled_eggs','lobster_bisque']
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}

nb_classes = len(class_names)

print(class_names_label)

IMAGE_SIZE = (150, 150)

# Завантаження даних
def load_data():
    DIRECTORY = r"C:\Users\Admin\Desktop\NM_LAB4_FOOD"
    CATEGORY = ["train", "test"]

    output = []

    for category in CATEGORY:
        path = os.path.join(DIRECTORY, category)
        # print(path)
        images = []
        labels = []

        print("Loading {}".format(category))

        for folder in os.listdir(path):
            label = class_names_label[folder]

            for file in os.listdir(os.path.join(path, folder)):
                img_path = os.path.join(os.path.join(path, folder), file)
                # print(img_path)

                src = cv2.imread(img_path)
                image = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
                image = cv2.resize(image, IMAGE_SIZE)

                images.append(image)
                labels.append(label)

        images = np.array(images, dtype='float32')
        labels = np.array(labels, dtype='int32')

        output.append((images, labels))

    return output

# Створення набору даних для навчання та тестування
(train_images, train_labels), (test_images, test_labels) = load_data()

train_images, train_labels = shuffle(train_images, train_labels, random_state=25)

# Створення згорткової нейронної мережі
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation = tf.nn.relu),
    tf.keras.layers.Dense(3, activation = tf.nn.softmax)
])

# Компіляція моделі
model.compile(loss ='sparse_categorical_crossentropy', optimizer = 'adam',  metrics=['accuracy'])

model.summary()

# Навчання моделі
history = model.fit(train_images, train_labels, batch_size = 128, epochs = 3, validation_split = 0.2)

if __name__ == '__main__':
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

    # Створення класифікаційного звіту
    test_loss = model.evaluate(test_images, test_labels)

    predictions = model.predict(test_images)
    pred_labels = np.argmax(predictions, axis=1)
    print(classification_report(test_labels, pred_labels))




