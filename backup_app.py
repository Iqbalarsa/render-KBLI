from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import pandas as pd
import joblib

# Load the necessary files
model = tf.keras.models.load_model('model_KBLI_v2.h5')
vectorizer = joblib.load('vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')
map_label = pd.read_excel('map_label.xlsx', dtype=str)  # Ensure this CSV has the columns 'KBLI' and 'Original_KBLI'
masterKBLI = pd.read_excel('MasterFile_KBLI.xlsx', dtype=str)  # Ensure this CSV has the columns 'KBLI', 'Judul_KBLI', 'Uraian_KBLI'

app = Flask(__name__)

# Function to preprocess and vectorize input text
def preprocess_text(input_text, vectorizer):
    input_vector = vectorizer.transform([input_text])
    input_array = np.array(input_vector.todense())
    return input_array

# Function to predict top labels for input text
def predict_top_labels(input_text, vectorizer, top_k=3):
    input_array = preprocess_text(input_text, vectorizer)
    predictions = model.predict(input_array)

    top_k_indices = np.argsort(predictions[0])[::-1][:top_k]
    top_k_labels_encoded = label_encoder.inverse_transform(top_k_indices)
    top_k_labels = top_k_labels_encoded.astype(str)  # Convert labels to string type
    top_k_probabilities = predictions[0][top_k_indices]

    top_k_original_labels = [
        map_label.loc[map_label['KBLI'] == label, 'Original_KBLI'].values[0] for label in top_k_labels
    ]

    top_k_data = []
    
    for label in top_k_original_labels:
        # Fetch relevant information from MasterFile_KBLI
        info = masterKBLI.loc[masterKBLI['KBLI'] == label, ['Judul_KBLI', 'Uraian_KBLI']].values
        if len(info) > 0:
            top_k_data.append([label, info[0][0], info[0][1]])  # Append KBLI, Judul_KBLI, Uraian_KBLI
        else:
            top_k_data.append([label, "Not Found", "Not Found"])  # Append default values if not found

    return top_k_original_labels, top_k_probabilities, top_k_data

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form['description']
    top_labels, top_probs, top_data = predict_top_labels(data, vectorizer)
    
    return jsonify({
        'top_labels': top_labels,
        'top_probabilities': top_probs.tolist(),
        'top_data': top_data
    })

if __name__ == "__main__":
    app.run(debug=True)
