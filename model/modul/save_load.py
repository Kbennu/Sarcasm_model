import torch
import os

def save_model(model, save_path):
    \"\"\"
    Сохранение модели на диск.
    \"\"\"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)

def load_model(model_class, model_path, *args, **kwargs):
    \"\"\"
    Загрузка модели с диска.
    \"\"\"
    model = model_class(*args, **kwargs)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    return model