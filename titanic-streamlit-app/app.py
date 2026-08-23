import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# Load the saved Pipeline model
@st.cache_resource
def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'titanic_pipeline_model.joblib')
    return joblib.load(model_path)

model = load_model()

# Header & UI Title
st.title("🚢 Titanic Survival Prediction App")
st.markdown("""
Predict whether a passenger would have survived the Titanic disaster based on their demographic and ticket details.
""")

st.divider()

# Input Form Layout
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class (Pclass)", options=[1, 2, 3], format_func=lambda x: f"Class {x}")
    sex = st.selectbox("Sex", options=["male", "female"])
    age = st.slider("Age", min_value=1, max_value=80, value=28, step=1)
    fare = st.number_input("Ticket Fare ($)", min_value=0.0, max_value=600.0, value=32.0, step=1.0)

with col2:
    embarked = st.selectbox("Port of Embarkation", options=["S", "C", "Q"], 
                            format_func=lambda x: {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[x])
    sibsp = st.number_input("Siblings / Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0, step=1)
    parch = st.number_input("Parents / Children Aboard (Parch)", min_value=0, max_value=10, value=0, step=1)

# Compute Engineered Features
family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0

st.divider()

# Prediction Action
if st.button("🔮 Predict Survival", type="primary", use_container_width=True):
    # Construct raw DataFrame matching the pipeline input schema
    input_data = pd.DataFrame([{
        'Pclass': pclass,
        'Sex': sex,
        'Age': float(age),
        'Fare': float(fare),
        'Embarked': embarked,
        'FamilySize': family_size,
        'IsAlone': is_alone
    }])
    
    # Run prediction through the pipeline
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    survival_prob = probabilities[1] * 100

    # Display Result Cards
    if prediction == 1:
        st.success(f"### 🎉 Result: Survived!")
        st.metric(label="Predicted Survival Probability", value=f"{survival_prob:.1f}%")
    else:
        st.error(f"### ⚠️ Result: Did Not Survive")
        st.metric(label="Predicted Survival Probability", value=f"{survival_prob:.1f}%")