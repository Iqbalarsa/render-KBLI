from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import load_model
from joblib import load
import pandas as pd

# Load the necessary files
model = load_model('model_KBLI_v3.h5')
vectorizer = load('vectorizer.pkl')
label_encoder = load('label_encoder.pkl')
map_label = pd.read_excel('mapping.xlsx', dtype=str)   # Ensure this CSV has the columns 'KBLI' and 'Original_KBLI'
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

    # Convert the map_label['KBLI'] column to string (ensure consistent format)
    map_label['KBLI'] = map_label['KBLI'].astype(str)

    # Retrieve the original KBLI values from the map_label dataframe
    top_k_original_labels = []
    for label in top_k_labels:
        # Lookup Original_KBLI in map_label where KBLI == label
        original_kbli = map_label.loc[map_label['KBLI'] == label, 'Original_KBLI']
        
        # Check if the KBLI exists in the map_label and get the Original_KBLI
        if not original_kbli.empty:
            top_k_original_labels.append(original_kbli.values[0])  # Append the corresponding Original_KBLI
        else:
            top_k_original_labels.append(f"KBLI {label} Not Found")  # Fallback if not found

    # Fetch additional data from masterKBLI based on the original KBLI
    top_k_data = []
    for original_label in top_k_original_labels:
        # Try to fetch the relevant information from masterKBLI using the Original_KBLI
        info = masterKBLI.loc[masterKBLI['KBLI'] == original_label, ['Judul_KBLI', 'Uraian_KBLI']].values
        if len(info) > 0:
            # Append Original_KBLI, Judul_KBLI, Uraian_KBLI, converting to Python native types
            top_k_data.append([str(original_label), str(info[0][0]), str(info[0][1])])
        else:
            # Default values if no match found
            top_k_data.append([str(original_label), f"Judul_KBLI for {original_label} Not Found", f"Uraian_KBLI for {original_label} Not Found"])

    # Ensure that probabilities are in a list of floats
    top_k_probabilities = top_k_probabilities.tolist()

    # Convert top_k_original_labels to strings
    top_k_original_labels = [str(label) for label in top_k_original_labels]

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
        'top_probabilities': top_probs,
        'top_data': top_data
    })

if __name__ == "__main__":
    app.run(debug=True)
