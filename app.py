# app.py
import streamlit as st
import numpy as np
import joblib
import pandas as pd
from keras.models import load_model

# ----------------------------
# Load models and preprocessing objects ONCE
# ----------------------------
@st.cache_resource
def load_resources():
    models = {
        "SVM": joblib.load("svm_model.pkl"),
        "Logistic Regression": joblib.load("lr_model.pkl"),
        "KNN": joblib.load("knn_model.pkl"),
        "Decision Tree": joblib.load("dt_model.pkl"),
    }
    
    # Load neural network and its preprocessing
    nn_model = load_model("nn_model.h5")
    
    # Load preprocessing objects for NN
    nn_scaler_tabular = joblib.load("nn_scaler_tabular.pkl")
    nn_pca_images = joblib.load("nn_pca_images.pkl")
    nn_image_cols = joblib.load("nn_image_cols.pkl")
    nn_tabular_cols = joblib.load("nn_tabular_cols.pkl")
    
    # Load preprocessing objects for KNN
    knn_scaler = joblib.load("knn_scaler.pkl")
    knn_pca = joblib.load("knn_pca.pkl")
    
    return models, nn_model, nn_scaler_tabular, nn_pca_images, nn_image_cols, nn_tabular_cols, knn_scaler, knn_pca

models, nn_model, nn_scaler_tabular, nn_pca_images, nn_image_cols, nn_tabular_cols, knn_scaler, knn_pca = load_resources()

# Load model features (for non-NN models)
with open("model_features.txt", "r", encoding="utf-8") as f:
    MODEL_FEATURES = [line.strip() for line in f if line.strip()]

# ----------------------------
# UI Inputs
# ----------------------------
FEATURE_NAMES = [
    "Ads", "Age", "Gender", "Religion", "Sport", "Medical_Condition",
    "Weight", "Nationality", "Hobby", "Field_of_Interest",
    "Salary", "Location", "Class_Level", "Dependencies"
]

CATEGORICAL_OPTIONS = {
    "Ads": ["Tech Gadget", "Organic Food", "Gym Membership", "Luxury Car",
            "Medical Insurance", "Engineering Tools", "Fashion Magazine",
            "Healthy Snacks", "Car Insurance", "Kids Toys"],
    "Gender": ["Male", "Female"],
    "Religion": ["Muslim", "Christian", "Coptic"],
    "Sport": ["Football", "Running", "No_Sport", "Swimming", "Cycling", "Dancing", "Tennis", "Yoga", "Walking", "Basketball"],
    "Medical_Condition": ["Healthy", "Diabetes", "Hypertension", "Arthritis", "Asthma"],
    "Nationality": ["Egyptian"],
    "Hobby": ["Coding", "Cooking", "Running", "Traveling", "Gardening", "Robotics",
              "Dancing", "Yoga", "Photography", "Painting", "Swimming", "Cycling",
              "Reading", "Gaming", "Skydiving", "Chess", "Singing", "Baking",
              "Fishing", "Hiking", "Knitting", "Archery", "Sculpting", "Board Games",
              "Pottery", "Volunteering", "Meditation", "Calligraphy", "Birdwatching"],
    "Field_of_Interest": ["CS", "Medical", "Engineering", "Business", "Arts", "Education"],
    "Location": ["Sheikh Zayed", "Maadi", "Dokki", "Cairo", "Giza", "6th of October",
                 "Heliopolis", "Nasr City", "London", "Paris", "New York", "Dubai"],
    "Class_Level": ["Middle", "Upper", "Lower"],
    "Dependencies": ["Spouse", "Children", "Grandma", "Mother", "Father", "Grandpa"]
}

# ----------------------------
# Encode input for non-NN models
# ----------------------------
def encode_input_for_sklearn(user_input_dict):
    sample = pd.DataFrame([user_input_dict])
    sample_encoded = pd.get_dummies(sample, columns=[
        'Ads', 'Gender', 'Religion', 'Sport', 'Medical_Condition',
        'Nationality', 'Hobby', 'Field_of_Interest', 'Location',
        'Class_Level', 'Dependencies'
    ])
    sample_encoded = sample_encoded.reindex(columns=MODEL_FEATURES, fill_value=0)
    return sample_encoded.values[0].astype(np.float32)

# ----------------------------
# Encode input for Neural Network (with PCA/scaling)
# ----------------------------
def encode_input_for_nn(user_input_dict):
    # Step 1: One-hot encode user input (same as sklearn)
    sample = pd.DataFrame([user_input_dict])
    sample_encoded = pd.get_dummies(sample, columns=[
        'Ads', 'Gender', 'Religion', 'Sport', 'Medical_Condition',
        'Nationality', 'Hobby', 'Field_of_Interest', 'Location',
        'Class_Level', 'Dependencies'
    ])
    sample_encoded = sample_encoded.reindex(columns=MODEL_FEATURES, fill_value=0)
    x_full = sample_encoded.values.astype(np.float32)
    
    # Step 2: Split into tabular and image parts
    n_image = len(nn_image_cols)  # should be 2048
    x_image = x_full[:, :n_image]
    x_tabular = x_full[:, n_image:]
    
    # Step 3: Apply preprocessing
    x_image_pca = nn_pca_images.transform(x_image)
    x_tabular_scaled = x_tabular.copy()
    x_tabular_scaled[:, :3] = nn_scaler_tabular.transform(x_tabular[:, :3])  # scale Age, Weight, Salary
    
    # Step 4: Combine
    x_nn = np.hstack([x_tabular_scaled, x_image_pca])
    return x_nn[0]

# ----------------------------
# Encode input for KNN (with scaler + PCA)
# ----------------------------
def encode_input_for_knn(user_input_dict):
    x_sklearn = encode_input_for_sklearn(user_input_dict)
    x_knn = knn_scaler.transform(x_sklearn.reshape(1, -1))
    x_knn = knn_pca.transform(x_knn)
    return x_knn[0]

# ----------------------------
# Ensemble prediction
# ----------------------------
def predict_ensemble(user_input_dict):
    weights = {
        "SVM": 0.83,
        "Neural Network": 0.79,
        "Logistic Regression": 0.82,
        "KNN": 0.72,
        "Decision Tree": 0.79,
    }
    total = sum(weights.values())
    
    weighted_prob = 0.0
    
    # SVM, LR, DT
    for name in ["SVM", "Logistic Regression", "Decision Tree"]:
        prob = models[name].predict_proba([encode_input_for_sklearn(user_input_dict)])[0][1]
        weighted_prob += prob * (weights[name] / total)
    
    # KNN: Apply scaler + PCA
    x_knn = encode_input_for_knn(user_input_dict)
    knn_prob = models["KNN"].predict_proba([x_knn])[0][1]
    weighted_prob += knn_prob * (weights["KNN"] / total)
    
    # NN model
    x_nn = encode_input_for_nn(user_input_dict)
    nn_prob = nn_model.predict(np.array([x_nn]), verbose=0)[0][0]
    weighted_prob += nn_prob * (weights["Neural Network"] / total)
    
    return 1 if weighted_prob >= 0.5 else 0

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Ad Relevance Predictor", page_icon="📊")
st.title("🎯 Ad Relevance Prediction")
st.markdown("Enter user & ad details to predict if the ad is **relevant** (1) or **not** (0).")

user_input = {}
user_input["Age"] = st.number_input("Age", min_value=0, max_value=120, value=30)
user_input["Weight"] = st.number_input("Weight (kg)", min_value=20, max_value=200, value=70)
user_input["Salary"] = st.number_input("Salary (EGP/year)", min_value=0, max_value=1000000, value=15000)

for col in ["Ads", "Gender", "Religion", "Sport", "Medical_Condition", 
            "Nationality", "Hobby", "Field_of_Interest", "Location", 
            "Class_Level", "Dependencies"]:
    user_input[col] = st.selectbox(col, CATEGORICAL_OPTIONS[col])

if st.button("🔮 Predict Ad Relevance"):
    try:
        with st.spinner("Predicting..."):
            result = predict_ensemble(user_input)
        if result == 1:
            st.success("✅ **Relevant Ad!** User is likely to engage.")
        else:
            st.error("❌ **Not Relevant** — low chance of engagement.")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {str(e)}")
        st.info("Make sure all preprocessing files (knn_scaler.pkl, knn_pca.pkl, etc.) are in the folder.")