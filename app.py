from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Model ലോഡ് ചെയ്യുന്നു
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Form-ൽ നിന്നുള്ള വാല്യൂസ് എടുക്കുന്നു
        val1 = request.form['bedrooms']
        val2 = request.form['bathrooms']
        val3 = request.form['floors']
        val4 = request.form['yr_built']
        
        # NumPy അറേയിലേക്ക് മാറ്റുന്നു
        arr = np.array([val1, val2, val3, val4], dtype=np.float64)
        
        # Prediction നടത്തുന്നു
        pred = model.predict([arr])
        output = round(float(pred[0]), 2)
        
        return render_template('index.html', data=output)

if __name__ == '__main__':
    app.run(debug=True)