from flask import Flask, request, render_template, redirect, url_for, session
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pandas as pd
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load ML Models
MANGO_MODEL_PATH = "mango.h5"
TOMATO_MODEL_PATH = "tomato_leaf.h5"
mango_model = load_model(MANGO_MODEL_PATH)
tomato_model = load_model(TOMATO_MODEL_PATH)

# Class labels for disease detection
MANGO_LABELS = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']
TOMATO_LABELS = ['Late Blight', 'Yellow Leaf Curl Virus', 'Leaf Mold', 'Healthy']

# Image Upload Folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Database setup
def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE,
                        password TEXT)''')
    conn.commit()
    conn.close()

init_db()

# Load Crop Dataset
data = pd.read_csv(r"C:\Users\Ankita\Desktop\combine_crop\Crop-Recomedation\Dataset\Crop_Recommendation.csv")

X = data[['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']]
y = data['Crop'].astype('category').cat.codes
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_scaled, y)
crop_mapping = dict(enumerate(data['Crop'].astype('category').cat.categories))

# Function to Predict Disease
def predict_disease(img_path, model, class_labels):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    return class_labels[predicted_class]

# 🏠 Home Page: Redirects to login first
@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('predict_crop'))  # Redirect to the crop prediction page after login
    return redirect(url_for('login'))  # Redirect to login if not logged in

# 📌 User Registration
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        try:
            cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
            conn.commit()
            return redirect(url_for('login'))
        except:
            return "Username already exists!"
        finally:
            conn.close()
    return render_template('register.html')

# 🔑 User Login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()
        if user and check_password_hash(user[2], password):
            session['username'] = username
            return redirect(url_for('predict_crop'))
        else:
            return "Invalid credentials!"
    return render_template('login.html')

# 🚪 Logout
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

# 🍃 Mango & Tomato Disease Detection (Requires Login)
@app.route('/mt_page', methods=['GET', 'POST'])
def mt_page():
    if 'username' not in session:  # Ensure user is logged in
        return redirect(url_for('login'))
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('mt.html', prediction="No file uploaded", image_path=None)
        file = request.files['file']
        plant_type = request.form.get('plant_type', None)
        if not plant_type or file.filename == '':
            return render_template('mt.html', prediction="Invalid input", image_path=None)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(file_path)
        if plant_type == 'mango':
            prediction = predict_disease(file_path, mango_model, MANGO_LABELS)
        elif plant_type == 'tomato':
            prediction = predict_disease(file_path, tomato_model, TOMATO_LABELS)
        else:
            prediction = "Invalid plant type"
        return render_template('mt.html', prediction=prediction, image_path=file_path)
    return render_template('mt.html')

# 🌾 Crop Recommendation (Requires Login)
@app.route('/predict1', methods=['GET', 'POST'])
def predict1():
    if 'username' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        features = [float(request.form[key]) for key in ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']]
        features_scaled = scaler.transform([features])
        prediction = knn.predict(features_scaled)[0]
        recommended_crop = crop_mapping[prediction]
        return render_template('predict1.html', prediction_text=f'Recommended Crop: {recommended_crop}')
    return render_template('predict1.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
