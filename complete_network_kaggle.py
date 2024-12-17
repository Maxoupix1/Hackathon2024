import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, Dataset, TensorDataset

class FeatureExtraction(nn.Module):
    def __init__(self, input_dim):
        super(FeatureExtraction, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return x

class ContextDetection(nn.Module):
    def __init__(self):
        super(ContextDetection, self).__init__()
        self.lstm1 = nn.LSTM(input_size=128, hidden_size=100, num_layers=2, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=100, hidden_size=200, num_layers=2, batch_first=True)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        return x

class PredictionRefinement(nn.Module):
    def __init__(self):
        super(PredictionRefinement, self).__init__()
        self.fc1 = nn.Linear(200, 128)
        self.fc2 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return x

class FullyConnectedLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FullyConnectedLayer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc(x)
        return x

class CompleteNetwork(nn.Module):
    def __init__(self, num_classes):
        super(CompleteNetwork, self).__init__()
        self.feature_extraction = FeatureExtraction(input_dim=6)  # 6 numerical features
        self.context_detection = ContextDetection()
        self.prediction_refinement = PredictionRefinement()
        self.classifier = FullyConnectedLayer(input_dim=128, output_dim=num_classes)  # Output for classification

    def forward(self, x):
        # Extract features
        features = x[:, :, :6]  # Extract numerical data (ignoring datetime column)
        x = self.feature_extraction(features)

        # Reshape for LSTM
        x = x.unsqueeze(1)  # Add sequence dimension (batch_size, seq_len, feature_dim)
        x = self.context_detection(x)

        # Refinement
        refined = self.prediction_refinement(x[:, -1, :])  # Use the last output of LSTM

        # Classification
        predictions = self.classifier(refined)
        return predictions

# Préparer les données
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# Charger et prétraiter les données d'entraînement
training_data = pd.read_csv('training_data.csv')
X_train = training_data[['acc_x_right', 'acc_y_right', 'acc_z_right', 'acc_x_left', 'acc_y_left', 'acc_z_left']].values
y_train = training_data['activity'].astype('category').cat.codes.values

train_dataset = CustomDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Définir les paramètres d'entraînement
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(set(y_train))
model = CompleteNetwork(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Entraîner le modèle
model.train()
for epoch in range(20):  # Adjust the number of epochs as necessary
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Charger et prétraiter les données de prédiction
prediction_data = pd.read_csv('prediction_data.csv')
X_pred = prediction_data[['acc_x_right', 'acc_y_right', 'acc_z_right', 'acc_x_left', 'acc_y_left', 'acc_z_left']].values

# Prédiction
model.eval()
with torch.no_grad():
    X_pred_tensor = torch.tensor(X_pred, dtype=torch.float32).to(device)
    predictions = model(X_pred_tensor.unsqueeze(0))
    predicted_labels = torch.argmax(predictions, dim=1).cpu().numpy()

# Ajouter les prédictions aux données de prédiction et sauvegarder
prediction_data['activity'] = pd.Categorical.from_codes(predicted_labels, categories=training_data['activity'].astype('category').cat.categories)
cutoff_time = pd.to_datetime('2024-07-25 19:00:00')
prediction_data = prediction_data[prediction_data['time'] <= cutoff_time]
output = prediction_data[['time', 'activity']]
output['time'] = output['time'].dt.strftime('%H:%M:%S')
output.to_csv('output_nn.csv', index=False)
print('Les prédictions ont été réalisées et le fichier output_nn.csv a été sauvegardé avec succès.')
