def split_data(data, test_size=0.2):
    \"\"\"
    Разделение данных на тренировочные и тестовые выборки.
    \"\"\"
    from sklearn.model_selection import train_test_split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        data['cleaned_text'], data['is_sarcastic'], test_size=test_size, random_state=42, stratify=data['is_sarcastic']
    )
    return train_texts, val_texts, train_labels, val_labels