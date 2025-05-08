import numpy as np
import json
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model


def preprocess_contour_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = np.zeros_like(gray)
    cv2.drawContours(contour_img, contours, -1, (255), thickness=2)
    resized = cv2.resize(contour_img, target_size)

    # Átalakítás RGB-re – minden csatorna ugyanaz a fekete-fehér kontúr
    rgb_contour = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

    # Keras-nak megfelelő alakra hozás + preprocess_input
    img_array = image.img_to_array(rgb_contour)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    return img_array


def classify_leaf(model, image_data):
    predictions = model.predict(image_data)
    predicted_class_index = np.argmax(predictions[0])

    with open("class_indices.json", "r") as f:
        class_indices = json.load(f)

    labels = {int(v): k for k, v in class_indices.items()}
    predicted_label = labels[predicted_class_index]

    print("Raw predictions:", predictions)
    print("Predicted index:", predicted_class_index)
    print("Available labels:", labels)
    print("A modell szerint ez a levél:", predicted_label)

    for i, prob in enumerate(predictions[0]):
        print(f"{labels[i]}: {prob:.2%}")

# Kép elérési út
leaf_path = 'images/oak/5613e0f4-7167-4448-8e3f-3392392ce37b.jpg'

# Normál modell bemenete
img = image.load_img(leaf_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Kontúr-alapú modell bemenete
contour_based_img_array = preprocess_contour_image(leaf_path)

# Modellek betöltése
normal_model = load_model('leaf_classifier_model.h5')
contour_based_model = load_model('leaf_classifier_model_f.h5')

# Osztályozás
classify_leaf(normal_model, img_array)
classify_leaf(contour_based_model, contour_based_img_array)
