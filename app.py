
from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

app = Flask(__name__)

model = joblib.load('best_svm_model.joblib')

def preprocess(data):
    num_cols = data.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = data.select_dtypes(include=['object']).columns

    num_imputer = SimpleImputer(strategy='mean')
    data[num_cols] = num_imputer.fit_transform(data[num_cols])

    scaler = StandardScaler()
    data[num_cols] = scaler.fit_transform(data[num_cols])

    cat_imputer = SimpleImputer(strategy='most_frequent')
    data[cat_cols] = cat_imputer.fit_transform(data[cat_cols])

    label_encoders = {col: LabelEncoder().fit(data[col]) for col in cat_cols}
    for col in cat_cols:
        data[col] = label_encoders[col].transform(data[col])

    return data

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    data_df = pd.DataFrame([input_data])

    processed_data = preprocess(data_df)

    prediction = model.predict(processed_data)

    return jsonify({'Credit_Score': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
