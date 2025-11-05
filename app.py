# Botnet Detection - Flask Application
# Save this as: app.py

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
from werkzeug.utils import secure_filename
import json
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'csv'}

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load trained model and preprocessing objects
print("Loading model and preprocessing objects...")
try:
    with open('botnet_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    with open('model_info.pkl', 'rb') as f:
        model_info = pickle.load(f)
    
    print("✓ Model and preprocessing objects loaded successfully!")
    
except FileNotFoundError as e:
    print(f"✗ Error loading model files: {e}")
    print("Please ensure all .pkl files are in the same directory as app.py")
    model = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_data(df):
    """Preprocess uploaded data to match training data format"""
    try:
        # Select only features used in training
        missing_features = set(feature_names) - set(df.columns)
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            # Add missing features with default values
            for feature in missing_features:
                df[feature] = 0
        
        # Select and order features
        df_processed = df[feature_names].copy()
        
        # Handle missing values
        df_processed = df_processed.fillna(df_processed.median(numeric_only=True))
        df_processed = df_processed.fillna(0)
        
        # Handle infinite values
        df_processed = df_processed.replace([np.inf, -np.inf], np.nan)
        df_processed = df_processed.fillna(0)
        
        # Scale features
        X_scaled = scaler.transform(df_processed)
        
        return X_scaled, df_processed
    
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        raise

def create_visualization(predictions, probabilities):
    """Create visualization charts for results"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Prediction distribution
    unique, counts = np.unique(predictions, return_counts=True)
    labels = ['Normal' if x == 0 else 'Botnet' for x in unique]
    colors = ['green' if x == 0 else 'red' for x in unique]
    axes[0].bar(labels, counts, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_title('Detection Results Distribution')
    axes[0].set_ylabel('Count')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Confidence distribution
    axes[1].hist(probabilities, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    axes[1].set_title('Confidence Score Distribution')
    axes[1].set_xlabel('Probability of Botnet')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Convert plot to base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return image_base64

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html', model_info=model_info if model else None)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle file upload and prediction"""
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500
    
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    # Check if file is valid
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload a CSV file'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read CSV file
        df = pd.read_csv(filepath)
        
        # Check if dataframe is empty
        if df.empty:
            return jsonify({'error': 'Uploaded file is empty'}), 400
        
        # Preprocess data
        X_scaled, df_processed = preprocess_data(df)
        
        # Make predictions
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1]  # Probability of botnet
        
        # Calculate statistics
        total_flows = len(predictions)
        botnet_count = int(np.sum(predictions))
        normal_count = total_flows - botnet_count
        botnet_percentage = (botnet_count / total_flows) * 100
        avg_confidence = float(np.mean(probabilities))
        
        # Create visualization
        viz_image = create_visualization(predictions, probabilities)
        
        # Prepare detailed results (show first 100 rows)
        detailed_results = []
        for i in range(min(100, len(predictions))):
            detailed_results.append({
                'index': i + 1,
                'prediction': 'Botnet' if predictions[i] == 1 else 'Normal',
                'confidence': f"{probabilities[i]:.2%}",
                'risk_level': 'High' if probabilities[i] > 0.8 else 'Medium' if probabilities[i] > 0.5 else 'Low'
            })
        
        # Prepare response
        response = {
            'success': True,
            'summary': {
                'total_flows': total_flows,
                'botnet_detected': botnet_count,
                'normal_traffic': normal_count,
                'botnet_percentage': f"{botnet_percentage:.2f}%",
                'average_confidence': f"{avg_confidence:.2%}",
                'status': 'THREAT DETECTED' if botnet_count > 0 else 'NETWORK SAFE'
            },
            'visualization': viz_image,
            'detailed_results': detailed_results,
            'model_performance': {
                'accuracy': f"{model_info['test_accuracy']*100:.2f}%",
                'precision': f"{model_info['precision']:.4f}",
                'recall': f"{model_info['recall']:.4f}",
                'f1_score': f"{model_info['f1_score']:.4f}"
            }
        }
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/about')
def about():
    """About page with project information"""
    return render_template('about.html')

@app.route('/model-info')
def model_information():
    """Display model information and performance metrics"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify(model_info)

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File is too large. Maximum size is 16MB'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    if model is None:
        print("\n" + "="*60)
        print("WARNING: Model not loaded!")
        print("Please run botnet_model_training.py first to train the model")
        print("="*60 + "\n")
    else:
        print("\n" + "="*60)
        print("BOTNET DETECTION SYSTEM - FLASK SERVER")
        print("="*60)
        print(f"Model loaded: {model_info['model_type']}")
        print(f"Test Accuracy: {model_info['test_accuracy']*100:.2f}%")
        print(f"Number of Features: {model_info['n_features']}")
        print("="*60 + "\n")
        print("Server starting...")
        print("Access the application at: http://localhost:5000")
        print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)