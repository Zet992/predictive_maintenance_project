import re
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, roc_auc_score
)
from xgboost import XGBClassifier


def analysis_and_model_page():
    st.title("Анализ данных и модель")
    uploaded_file = st.file_uploader("Загрузите датасет (CSV)", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        return

    data = data.drop(columns=["UDI", "Product ID", "TWF", "HDF",
                              "PWF", "OSF", "RNF"])
    label_encoder = LabelEncoder()
    data["Type"] = label_encoder.fit_transform(data["Type"])

    scaler = StandardScaler()
    numerical_features = ["Air temperature [K]", "Process temperature [K]",
                          "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
    data[numerical_features] = scaler.fit_transform(data[numerical_features])
    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    data.columns = [regex.sub("_", c) for c in data.columns.values]

    X = data.drop(columns=["Machine failure"])
    y = data["Machine failure"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)

    xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb.fit(X_train, y_train)

    y_pred = xgb.predict(X_test)
    y_pred_proba = xgb.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)
    print("ROC-AUC:", roc_auc)

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    st.header("Результаты обучения модели")
    st.write(f"Accuracy: {accuracy:.2f}")

    st.subheader("Результаты обучения модели")
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    st.subheader("Classification Report")
    st.text(class_report)

    st.header("Предсказание по новым данным")
    with st.form("prediction_form"):
        st.write("Введите значения признаков для предсказания")
        product_type = st.selectbox("type", ["L", "M", "H"])
        air_temp = st.number_input("air temperature [K]")
        process_temp = st.number_input("process temperature [K]")
        rotational_speed = st.number_input("rotational speed [rpm]")
        torque = st.number_input("torque [Nm]")
        tool_wear = st.number_input("tool wear [min]")

        submit_button = st.form_submit_button("Предсказать")

        if submit_button:
            input_data = pd.DataFrame({
                "Type": [product_type],
                "Air temperature [K]": [air_temp],
                "Process temperature [K]": [process_temp],
                "Rotational speed [rpm]": [rotational_speed],
                "Torque [Nm]": [torque],
                "Tool wear [min]": [tool_wear],
            })
            input_data["Type"] = label_encoder.transform(input_data["Type"])
            input_data[numerical_features] = scaler.transform(input_data[numerical_features])
            input_data.columns = [regex.sub("_", c) for c in input_data.columns.values]
            prediction = xgb.predict(input_data)
            prediction_proba = xgb.predict_proba(input_data)[:, 1]
            st.write(f"Предсказание: {prediction[0]}")
            st.write(f"Вероятность отказа: {prediction_proba[0]:.2f}")


analysis_and_model_page()
