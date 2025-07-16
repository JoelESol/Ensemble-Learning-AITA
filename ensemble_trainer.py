import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm


class AITADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        combined_text = self.create_combined_text(item)

        encoding = self.tokenizer(
            combined_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }

    def create_combined_text(self, item):
        """Create combined text from all agent verdicts and explanations"""
        parts = []

        for agent, agent_data in item['agents'].items():
            verdict = agent_data.get('verdict', 'unknown')
            explanation = agent_data.get('explanation', '')

            agent_text = f"Agent {agent} says {verdict}: {explanation}"
            parts.append(agent_text)

        return " [SEP] ".join(parts)


class VerdictTransformer(nn.Module):
    def __init__(self, model_name, num_classes, dropout_rate=0.3):
        super(VerdictTransformer, self).__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            last_hidden_state = outputs.last_hidden_state
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
            masked_embeddings = last_hidden_state * attention_mask_expanded
            sum_embeddings = torch.sum(masked_embeddings, dim=1)
            sum_mask = torch.sum(attention_mask, dim=1, keepdim=True)
            pooled_output = sum_embeddings / sum_mask

        output = self.dropout(pooled_output)
        return self.classifier(output)


class AITAVerdictPredictor:
    def __init__(self, model_name='distilbert-base-uncased', max_length=512):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.label_encoder = LabelEncoder()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def load_data(self, results_path):
        """Load and preprocess data from JSONL file"""
        print(f"Loading data from {results_path}")

        data = []
        with open(results_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())

                if not item.get('agents') or item.get('original_verdict') in ['unknown', 'error']:
                    continue

                data.append(item)

        print(f"Loaded {len(data)} valid samples")
        return data

    def preprocess_data(self, data):
        """Preprocess data for training"""
        labels = [item['original_verdict'] for item in data]
        encoded_labels = self.label_encoder.fit_transform(labels)

        for i, item in enumerate(data):
            item['label'] = encoded_labels[i]

        print(f"Label classes: {list(self.label_encoder.classes_)}")
        print(f"Label distribution: {pd.Series(labels).value_counts()}")

        return data

    def create_data_loaders(self, data, test_size=0.15, batch_size=16):
        """Create train/validation data loaders"""
        train_data, val_data = train_test_split(data, test_size=test_size, random_state=42,
                                                stratify=[item['label'] for item in data])

        train_dataset = AITADataset(train_data, self.tokenizer, self.max_length)
        val_dataset = AITADataset(val_data, self.tokenizer, self.max_length)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    def train_model(self, train_loader, val_loader, epochs=3, learning_rate=2e-5):
        """Train the transformer model"""
        num_classes = len(self.label_encoder.classes_)
        self.model = VerdictTransformer(self.model_name, num_classes).to(self.device)

        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        train_losses = []
        val_losses = []
        val_accuracies = []

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            self.model.train()
            train_loss = 0
            train_pbar = tqdm(train_loader, desc="Training")

            for batch in train_pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_pbar.set_postfix({'loss': loss.item()})

            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation
            self.model.eval()
            val_loss = 0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc="Validation")
                for batch in val_pbar:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    outputs = self.model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    preds = torch.argmax(outputs, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                    val_pbar.set_postfix({'loss': loss.item()})

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = accuracy_score(all_labels, all_preds)

            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)

            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val Loss: {avg_val_loss:.4f}")
            print(f"Val Accuracy: {val_accuracy:.4f}")

        return train_losses, val_losses, val_accuracies

    def evaluate_model(self, val_loader):
        """Evaluate the model and return predictions"""
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return all_preds, all_labels

    def plot_training_history(self, train_losses, val_losses, val_accuracies, save_path=None):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss plot
        ax1.plot(train_losses, label='Training Loss', marker='o')
        ax1.plot(val_losses, label='Validation Loss', marker='s')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy plot
        ax2.plot(val_accuracies, label='Validation Accuracy', marker='o', color='green')
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history saved to: {save_path}")

        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        class_names = self.label_encoder.classes_

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")

        plt.show()

    def save_model(self, save_path):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'label_encoder': self.label_encoder,
            'model_name': self.model_name,
            'max_length': self.max_length
        }, save_path)
        print(f"Model saved to: {save_path}")

    def load_model(self, load_path):
        """Load a saved model"""
        checkpoint = torch.load(load_path, map_location=self.device)

        self.label_encoder = checkpoint['label_encoder']
        num_classes = len(self.label_encoder.classes_)

        self.model = VerdictTransformer(self.model_name, num_classes).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        print(f"Model loaded from: {load_path}")


def main():
    predictor = AITAVerdictPredictor()

    data = predictor.load_data("dataset/agent_results.jsonl")
    data = predictor.preprocess_data(data)

    train_loader, val_loader = predictor.create_data_loaders(data, batch_size=16)

    print("\nStarting training...")
    train_losses, val_losses, val_accuracies = predictor.train_model(
        train_loader, val_loader, epochs=5, learning_rate=2e-6
    )

    print("\nEvaluating model...")
    predictions, true_labels = predictor.evaluate_model(val_loader)

    accuracy = accuracy_score(true_labels, predictions)
    print(f"\nFinal Validation Accuracy: {accuracy:.4f}")

    class_names = predictor.label_encoder.classes_
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=class_names))

    output_dir = Path("transformer_outputs")
    output_dir.mkdir(exist_ok=True)

    predictor.plot_training_history(
        train_losses, val_losses, val_accuracies,
        save_path=output_dir / "training_history.png"
    )

    predictor.plot_confusion_matrix(
        true_labels, predictions,
        save_path=output_dir / "confusion_matrix.png"
    )

    predictor.save_model(output_dir / "verdict_predictor.pth")

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()