import torch
from torch import nn
from transformers import DistilBertModel

class HybridModel(nn.Module):
    \"\"\"
    Гибридная модель, объединяющая DistilBERT и числовые признаки.
    \"\"\"
    def __init__(self, distilbert, num_features):
        super(HybridModel, self).__init__()
        self.distilbert = distilbert
        self.feature_layer = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Linear(64 + distilbert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask, features):
        \"\"\"
        Прямой проход через модель.
        \"\"\"
        distilbert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        bert_embeddings = distilbert_output.last_hidden_state[:, 0, :]  # [CLS] токен
        feature_output = self.feature_layer(features)
        combined = torch.cat((bert_embeddings, feature_output), dim=1)
        return self.classifier(combined)