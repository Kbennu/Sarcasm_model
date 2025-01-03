# -*- coding: utf-8 -*-
"""script

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1pck3O9KN2soQHveClo-l2wuDYzaPI1uy
"""

import os
import streamlit as st
import torch
from transformers import DistilBertModel

# Определяем кастомную модель HybridModel
class HybridModel(torch.nn.Module):
    def __init__(self, distilbert, num_features):
        super(HybridModel, self).__init__()
        self.distilbert = distilbert
        self.feature_layer = torch.nn.Sequential(
            torch.nn.Linear(num_features, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2)
        )
        self.classifier = torch.nn.Linear(64 + distilbert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask, features):
        distilbert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        bert_embeddings = distilbert_output.last_hidden_state[:, 0, :]  # [CLS] token embeddings
        feature_output = self.feature_layer(features)
        combined = torch.cat((bert_embeddings, feature_output), dim=1)
        return self.classifier(combined)

# Пути к модели и токенизатору
model_path = './HybridModel_state_dict.pth'  # Путь к state_dict
tokenizer_path = './HybridTokenizer'

try:
    # Проверка наличия файлов
    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        st.error(f"Model path contents: {os.listdir(model_path) if os.path.exists(model_path) else 'Not found'}")
        st.error(f"Tokenizer path contents: {os.listdir(tokenizer_path) if os.path.exists(tokenizer_path) else 'Not found'}")
        raise FileNotFoundError(f"Paths {model_path} or {tokenizer_path} do not exist.")

    # Загрузка модели и токенизатора
    distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
    model = HybridModel(distilbert, num_features=10)  # Укажите правильное число признаков
    model.load_state_dict(torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

    # Интерфейс приложения
    st.set_page_config(page_title="Sarcasm Detection", page_icon="🤖", layout="centered")
    st.title("Sarcasm Detection App")
    st.success("Model and tokenizer successfully loaded.")

    user_input = st.text_area("Enter text to check if it's sarcasm or not:")

    if user_input:
        if len(user_input.split()) > 128:
            st.warning("Input text exceeds the maximum token limit of 128. Please shorten your text.")
        else:
            # Пример числовых признаков (замените на реальные данные)
            dummy_features = torch.zeros(1, 10)  # Замените на фактические данные
            inputs = tokenizer(user_input, truncation=True, padding=True, max_length=128, return_tensors="pt")
            inputs = {key: val.to('cuda' if torch.cuda.is_available() else 'cpu') for key, val in inputs.items()}
            dummy_features = dummy_features.to('cuda' if torch.cuda.is_available() else 'cpu')

            # Получение предсказания
            with torch.no_grad():
                outputs = model(inputs['input_ids'], inputs['attention_mask'], dummy_features)
                logits = outputs
                predicted_class = torch.argmax(logits, dim=1).item()
                prob = torch.softmax(logits, dim=1)[0, 1].item()

            classification = "Sarcasm" if predicted_class == 1 else "Not sarcasm"
            st.subheader(f"Prediction: {classification}")
            st.write(f"Probability: {prob * 100:.2f}%")
            if classification == "Sarcasm":
                st.markdown('<p style="color:red; font-size: 24px;">Warning: Sarcasm detected!</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p style="color:green; font-size: 24px;">No sarcasm detected.</p>', unsafe_allow_html=True)

except Exception as e:
    st.error(f"Error loading model or tokenizer: {e}")