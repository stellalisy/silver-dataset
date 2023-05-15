#! /usr/bin/env python3

import torch
from datasets import Dataset

class SilverDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # item = {key: val[idx].clone().detach().requires_grad_(False) for key, val in self.encodings.items()}
        # item['labels'] = torch.tensor(self.labels[idx])
        # return item
        input_ids = torch.tensor(self.encodings["input_ids"][idx]).squeeze().clone().detach()
        target_ids = torch.tensor(self.labels["input_ids"][idx]).squeeze().clone().detach()
        return {"input_ids": input_ids, "labels": target_ids}

    def __len__(self):
        return len(self.encodings["input_ids"])