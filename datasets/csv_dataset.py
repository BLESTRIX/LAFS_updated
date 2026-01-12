import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

class CSVDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.root = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.root / row['image_path']).convert('RGB')
        label = int(row['label'])

        if self.transform:
            img = self.transform(img)

        return img, label
