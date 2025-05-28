import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# --- Simulate Dataset (MFCC + tabular data) ---
n_samples = 200
mfcc_dim = (13, 20)  # 13 MFCCs over 20 time frames (simulated sequence)

# Simulated MFCCs: shape (n_samples, 13, 20)
mfcc_data = np.random.normal(size=(n_samples, *mfcc_dim)).astype(np.float32)

# Tabular features: caffeine (mg), tiredness (0-10), previous sleep (hrs)
tabular_data = np.hstack([
    np.random.randint(0, 400, size=(n_samples, 1)),
    np.random.randint(0, 11, size=(n_samples, 1)),
    np.random.uniform(3, 9, size=(n_samples, 1))
]).astype(np.float32)

# Synthetic target: 0–10 sleep quality score
target = 10 - (tabular_data[:, 0] / 400) * 3 - tabular_data[:, 1] * 0.5 + tabular_data[:, 2] * 0.8
target = np.clip(target + np.random.normal(0, 0.5, size=target.shape), 0, 10).astype(np.float32)

# --- Dataset & DataLoader ---
class SleepDataset(Dataset):
    def __init__(self, mfcc, tabular, target):
        self.mfcc = torch.tensor(mfcc)
        self.tabular = torch.tensor(tabular)
        self.target = torch.tensor(target).unsqueeze(1)

    def __len__(self):
        return len(self.mfcc)

    def __getitem__(self, idx):
        return self.mfcc[idx], self.tabular[idx], self.target[idx]

X_mfcc_train, X_mfcc_test, X_tab_train, X_tab_test, y_train, y_test = train_test_split(
    mfcc_data, tabular_data, target, test_size=0.2, random_state=42
)

train_dataset = SleepDataset(X_mfcc_train, X_tab_train, y_train)
test_dataset = SleepDataset(X_mfcc_test, X_tab_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# --- Model ---
class SleepNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # (B, 1, 13, 20) → (B, 16, 13, 20)
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),                         # → (B, 16, 6, 10)
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # → (B, 32, 6, 10)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),                # → (B, 32, 1, 1)
        )

        self.fc = nn.Sequential(
            nn.Linear(32 + 3, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, mfcc, tabular):
        x = mfcc.unsqueeze(1)  # (B, 1, 13, 20)
        x = self.cnn(x).view(x.size(0), -1)  # flatten to (B, 32)
        x = torch.cat([x, tabular], dim=1)  # concat with tabular (B, 35)
        return self.fc(x)

# --- Training ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SleepNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

n_epochs = 20
for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    for mfcc, tabular, target in train_loader:
        mfcc, tabular, target = mfcc.to(device), tabular.to(device), target.to(device)

        optimizer.zero_grad()
        preds = model(mfcc, tabular)
        loss = loss_fn(preds, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {total_loss/len(train_loader):.4f}")

# --- Save Model ---
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/audio_classifier.pth")
print("PyTorch model saved to models/audio_classifier.pth")
