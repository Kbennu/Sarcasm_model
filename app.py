import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Загрузка обученной модели и токенизатора
model = torch.load("sarcasm_model.pkl")
tokenizer = torch.load("vectorizer.pkl")

def predict_sarcasm(text):
    # Токенизация текста
    inputs = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt")

    # Получение предсказания
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        prob = torch.softmax(logits, dim=1)[0, 1].item()  # вероятность сарказма

    return "Sarcasm" if predicted_class == 1 else "Not sarcasm", prob

# Настройки страницы
st.set_page_config(page_title="Sarcasm Detection", page_icon="🤖", layout="centered")

# Заголовок и описание
st.title("Sarcasm Detection App")
st.markdown("Enter a headline to check if it's sarcasm or not!")

# Ввод текста
user_input = st.text_area("Enter text here:")

if user_input:
    # Получение предсказания
    classification, prob = predict_sarcasm(user_input)

    # Отображение результата
    st.subheader(f"Prediction: {classification}")
    st.write(f"Probability: {prob * 100:.2f}%")

    # Визуальные эффекты
    if classification == "Sarcasm":
        st.markdown('<p style="color:red; font-size: 24px;">Warning: Sarcasm detected!</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:green; font-size: 24px;">No sarcasm detected.</p>', unsafe_allow_html=True)