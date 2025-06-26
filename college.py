import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

st.title("🎓 EAMCET College Predictor")
st.markdown("Suggesting colleges based on your rank, category, gender, and branch.")

# STEP 1: Load the CSV file
@st.cache_resource
def load_and_prepare_data():
    df = pd.read_csv("vijay.csv", header=None)

    # Set the first row as header
    df.columns = df.iloc[0]
    df = df[1:].reset_index(drop=True)
    df.columns = df.columns.map(lambda x: str(x).replace('\n', '').replace(' ', '').strip())

    # Identify rank columns
    rank_columns = [col for col in df.columns if "BOYS" in col.upper() or "GIRLS" in col.upper()]

    # Reshape data
    data = []
    for col in rank_columns:
        if col not in df.columns:
            continue
        temp = df[['inst_name', 'branch_code', col]].copy()
        temp.columns = ['inst_name', 'branch_code', 'rank']
        parts = col.upper().split('_')

        if parts[0] == 'OC' and parts[1] == 'EWS':
            temp['category'] = 'OC_EWS'
            temp['gender'] = parts[2].capitalize()
        else:
            temp['category'] = parts[0]
            temp['gender'] = parts[1].capitalize()

        data.append(temp)

    if not data:
        return None, None

    reshaped_df = pd.concat(data, ignore_index=True)
    reshaped_df.dropna(subset=['rank'], inplace=True)
    reshaped_df['rank'] = pd.to_numeric(reshaped_df['rank'], errors='coerce')
    reshaped_df.dropna(subset=['rank'], inplace=True)

    # Prepare model data
    X = reshaped_df[['rank', 'branch_code', 'category', 'gender']]
    y = reshaped_df['inst_name']

    X_encoded = pd.get_dummies(X)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return reshaped_df, model

reshaped_df, clf = load_and_prepare_data()

# STEP 2: User Inputs
with st.form("input_form"):
    student_rank = st.number_input("📥 Enter your EAMCET rank:", min_value=0, step=1)
    category = st.text_input("📥 Enter your category (e.g., BCC, OC, SC):").strip()
    gender = st.selectbox("📥 Select your gender:", ["Boys", "Girls"])
    branch_code = st.text_input("📥 (Optional) Enter branch code (e.g., CSE, ECE):").strip()

    submitted = st.form_submit_button("🔍 Show College Recommendations")

# STEP 3: Suggestion logic
def suggest_colleges(student_rank, category, gender, branch_code=None, top_n=10):
    filtered = reshaped_df[
        (reshaped_df['category'].str.upper() == category.upper()) &
        (reshaped_df['gender'] == gender.capitalize())
    ]
    if branch_code:
        filtered = filtered[filtered['branch_code'].str.upper() == branch_code.upper()]
    eligible = filtered[filtered['rank'] >= student_rank]
    recommended = eligible.sort_values(by='rank')
    return recommended[['inst_name', 'branch_code', 'rank']]

# STEP 4: Display Output
if submitted:
    if reshaped_df is None:
        st.error("❌ No valid rank columns found in CSV.")
    else:
        result = suggest_colleges(student_rank, category, gender, branch_code)
        if result.empty:
            st.warning("❗ No matching colleges found based on your input.")
        else:
            st.success("🎓 Recommended Colleges:")
            st.dataframe(result)
