from flask import Flask, request, jsonify, send_file, render_template
import numpy as np
import cv2
import requests
from PIL import Image

app = Flask(__name__)

# Endpoints de TensorFlow Serving
CLASSIFIER_URL = "http://localhost:8502/v1/models/tumor_classifier:predict"
SEGMENTATION_URL = "http://localhost:8502/v1/models/ResUNet:predict"

def preprocess_image(image, is_segmentation=False):
    """Preprocesa la imagen para los modelos."""
    if is_segmentation:
        image = image.resize((256, 256))  # Ajustar tamaño según el modelo de segmentación
    else:
        image = image.resize((128, 128))  # Ajustar tamaño según el modelo de clasificación

    image = np.array(image) / 255.0  # Normalizar
    image = np.expand_dims(image, axis=0)  # Agregar dimensión batch
    return image

def predict_with_serving(url, data):
    """Realiza una predicción enviando datos a TensorFlow Serving."""
    response = requests.post(url, json={"instances": data.tolist()})
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error en la predicción: {response.text}")

def generate_mask(image):
    """Genera la máscara de segmentación."""
    pred_mask = predict_with_serving(SEGMENTATION_URL, image)["predictions"][0]
    pred_mask = (np.array(pred_mask) > 0.5).astype(np.uint8)  # Convertir a binario
    return pred_mask

def overlay_mask(original, mask):
    """Superpone la máscara sobre la imagen original."""
    mask_colored = cv2.applyColorMap(mask * 255, cv2.COLORMAP_JET)  # Color falso
    overlay = cv2.addWeighted(original, 0.7, mask_colored, 0.3, 0)
    return overlay

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No se ha enviado ninguna imagen"})
    
    file = request.files["file"]
    image = Image.open(file).convert("RGB")
    
    # Predicción con tumor_classifier (clasificación)
    processed_image = preprocess_image(image, is_segmentation=False)
    prediction = predict_with_serving(CLASSIFIER_URL, processed_image)
    tumor_detected = prediction["predictions"][0][0] > 0.5  # Ajustar umbral según modelo
    
    original_array = np.array(image)
    
    if tumor_detected:
        # Generar máscara
        processed_image2 = preprocess_image(image, is_segmentation=True)
        mask = generate_mask(processed_image2)
        mask_resized = cv2.resize(mask, (original_array.shape[1], original_array.shape[0]))
        
        # Generar MRI con máscara
        overlay = overlay_mask(original_array, mask_resized)
        
        # Guardar imágenes
        cv2.imwrite("static/original.jpg", cv2.cvtColor(original_array, cv2.COLOR_RGB2BGR))
        cv2.imwrite("static/mask.jpg", mask_resized * 255)
        cv2.imwrite("static/overlay.jpg", overlay)
        mri_path = "static/original.jpg"
        mask_path = "static/mask.jpg"
        overlay_path = "static/overlay.jpg"
        
        return render_template("Result.html", mri=mri_path, mask=mask_path, overlay=overlay_path, message="Se detectó un tumor en la imagen.")
    else:
         # Redimensionar la imagen original a 256x256 para que coincida con el tamaño del modelo de segmentación
        original_resized = image.resize((256, 256))
        original_resized_array = np.array(original_resized)
        
        # Crear una imagen negra de 256x256
        black_image = np.zeros_like(original_resized_array)
        cv2.imwrite("static/black_image.jpg", cv2.cvtColor(black_image, cv2.COLOR_RGB2BGR))
        
        # Guardar la imagen procesada (escalada) en 256x256
        processed_image_resized = Image.fromarray((processed_image[0] * 255).astype(np.uint8))
        processed_image_resized = processed_image_resized.resize((256, 256))  # Redimensionar
        processed_image_resized.save("static/processed_image.jpg")
        
        # Guardar la imagen final (sin tumor) redimensionada a 256x256
        cv2.imwrite("static/final_image.jpg", cv2.cvtColor(original_resized_array, cv2.COLOR_RGB2BGR))
        
        black_image_path = "static/black_image.jpg"
        processed_image_path = "static/processed_image.jpg"
        final_image_path = "static/final_image.jpg"
        
        return render_template("Result.html", mri=processed_image_path, mask=black_image_path, overlay=final_image_path, message="No se detectó tumor en la imagen.")


if __name__ == "__main__":
    app.run(debug=True)