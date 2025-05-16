import torch
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd

class AITADataset(Dataset):
    def __init__(self, csv_path, titles_column="title",text_column="body", label_column="verdict"):
        df = pd.read_csv(csv_path)
        self.titles = df[titles_column].fillna("").tolist()
        self.texts = df[text_column].fillna("").tolist()
        labels = df[label_column].tolist()
        self.label_map = {label: i for i, label in enumerate(sorted(set(labels)))}
        self.labels = [self.label_map[label] for label in labels]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def get_dataset(batch_size, workers):

    data = AITADataset("dataset/aita_clean.csv")
    train_len = int(len(data) * 0.8)
    test_len = len(data) - train_len
    train_dataset, test_dataset = random_split(data, [train_len, test_len])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=int(workers))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=int(workers))
    return train_dataloader, test_dataloader, data.label_map
