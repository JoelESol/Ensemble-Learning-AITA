import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class VerdictEnsembleTransformer(nn.Module):
    def __init__(self, encoder_model="sentence-transformers/all-MiniLM-L6-v2", hidden_dim=512, num_agents=4, num_classes=4):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_model)
        self.encoder = AutoModel.from_pretrained(encoder_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.encoder.config.hidden_size, nhead=4),
            num_layers=2
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size * num_agents, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, agent_outputs: list[str]):
        # agent_outputs: List of strings of length num_agents
        tokenized = self.tokenizer(agent_outputs, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            embedded = self.encoder(**tokenized).last_hidden_state[:, 0, :]  # CLS token
        embedded = embedded.unsqueeze(1)  # shape: [num_agents, 1, hidden]
        transformed = self.transformer(embedded)  # optional
        flat = transformed.view(1, -1)  # Flatten sequence
        return self.classifier(flat)