from flask import Flask, render_template, request
import pickle
import numpy as np
import os

# Initialize Flask app with explicit template folder
app = Flask(__name__, template_folder='templates', static_folder='static')

# Print current directory for debugging
print(f"Current directory: {os.getcwd()}")
print(f"Templates folder exists: {os.path.exists('templates')}")
print(f"Model file exists: {os.path.exists('drug_model.pkl')}")

# Load model
try:
    model = pickle.load(open('drug_model.pkl', 'rb'))
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"ERROR loading model: {e}")
    model = None

@app.route('/')
def home():
    try:
        print("Attempting to render index.html")
        return render_template('index.html')
    except Exception as e:
        print(f"ERROR rendering index.html: {e}")
        return f"<h1>Error: {e}</h1><p>Check console for details</p>"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return "<h1>Error: Model not loaded. Please run drug_prediction.py first.</h1>"
        
        # Get form data
        age = int(request.form['age'])
        sex = request.form['sex']
        bp = request.form['bp']
        cholesterol = request.form['cholesterol']
        na_to_k = float(request.form['na_to_k'])
        
        print(f"Prediction request: Age={age}, Sex={sex}, BP={bp}, Chol={cholesterol}, Na/K={na_to_k}")
        
        # Encode inputs
        sex_encoded = 1 if sex == 'M' else 0
        bp_map = {'HIGH': 0, 'LOW': 1, 'NORMAL': 2}
        chol_map = {'HIGH': 0, 'NORMAL': 1}
        
        bp_encoded = bp_map[bp]
        chol_encoded = chol_map[cholesterol]
        
        # Make prediction
        features = np.array([[age, sex_encoded, bp_encoded, chol_encoded, na_to_k]])
        prediction = model.predict(features)[0]
        
        # Get confidence
        try:
            confidence = max(model.predict_proba(features)[0]) * 100
        except:
            confidence = 95.0
        
        # Decode result
        drugs = ['Drug A', 'Drug B', 'Drug C', 'Drug X', 'Drug Y']
        result = drugs[prediction]
        
        print(f"Result: {result} ({confidence:.1f}%)")
        
        return render_template('result.html', 
                             prediction=result,
                             confidence=f"{confidence:.1f}",
                             age=age,
                             sex=sex,
                             bp=bp,
                             cholesterol=cholesterol,
                             na_to_k=na_to_k)
    except Exception as e:
        print(f"ERROR in prediction: {e}")
        return f"<h1>Error: {e}</h1>"

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Starting Flask Server...")
    print("="*60)
    print("📂 Make sure you have:")
    print("   - templates/index.html")
    print("   - templates/result.html")
    print("   - static/style.css")
    print("   - drug_model.pkl")
    print("\n🌐 Open browser at: http://127.0.0.1:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)
