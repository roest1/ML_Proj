import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif

# Load and Preprocess Data


class DataPreprocessor:
    def __init__(self, filepath, scaler_type='StandardScaler'):
        self.data = pd.read_csv(filepath)
        self.numeric_features = self.data.select_dtypes(
            include=['int64', 'float64']).columns.tolist()
        self.categorical_features = self.data.select_dtypes(
            include=['object']).columns.tolist()
        self.scaler_type = scaler_type

    def normalize_data(self):
        if self.scaler_type == 'MinMaxScaler':
            scaler = MinMaxScaler()
        elif self.scaler_type == 'RobustScaler':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        self.data[self.numeric_features] = scaler.fit_transform(
            self.data[self.numeric_features])
        return self.data

    def feature_engineering(self):
        # Creating interaction features
        self.data['age_cholesterol'] = self.data['age'] * self.data['chol']
        self.data['thalach_slope'] = self.data['thalach'] * self.data['slope']
        return self.data

    def select_features(self, X, y, k=10):
        # Feature selection using SelectKBest
        selector = SelectKBest(score_func=f_classif, k=k)
        X_new = selector.fit_transform(X, y)
        selected_features = selector.get_support(indices=True)
        X = X.iloc[:, selected_features]
        return X

    def get_features_and_target(self, target_column='target'):
        X = self.data.drop(target_column, axis=1)
        y = self.data[target_column]
        return X, y

# Define Neural Network


class NeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super(NeuralNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Sigmoid for binary classification
        )

    def forward(self, x):
        return self.net(x)

# Train Function


class ModelTrainer:
    def __init__(self, model, X_train, y_train, X_val, y_val, learning_rate=0.001, num_epochs=100, weight_decay=1e-5):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(
        ), lr=self.learning_rate, weight_decay=weight_decay)
        self.train_losses = []
        self.val_losses = []

    def train(self, verbose=True):
        best_model_weights = self.model.state_dict()
        best_val_loss = float('inf')
        for epoch in range(self.num_epochs):
            # Training Phase
            self.model.train()
            y_pred = self.model(self.X_train)
            loss = self.criterion(y_pred, self.y_train)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.train_losses.append(loss.item())

            # Validation Phase
            self.model.eval()
            with torch.no_grad():
                y_val_pred = self.model(self.X_val)
                val_loss = self.criterion(y_val_pred, self.y_val)
                self.val_losses.append(val_loss.item())
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    best_model_weights = self.model.state_dict()

            if verbose:
                print(
                    f"Epoch [{epoch + 1}/{self.num_epochs}] - Loss: {loss.item():.4f} - Val Loss: {val_loss.item():.4f}")

        # Load best weights
        self.model.load_state_dict(best_model_weights)

    def plot_losses(self):
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Losses')
        plt.show()


# Main Workflow
if __name__ == "__main__":
    # Step 1: Data Loading and Preprocessing
    preprocessor = DataPreprocessor('./heart.csv', scaler_type='MinMaxScaler')
    preprocessor.normalize_data()
    preprocessor.feature_engineering()
    X, y = preprocessor.get_features_and_target()

    # Ensure target is binary (0 or 1)
    y = y.apply(lambda x: 1 if x > 0 else 0)

    # Feature Selection
    X = preprocessor.select_features(X, y, k=10)

    # Step 2: Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42)

    # Step 3: Scale Data and Convert to Tensors
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(
        y_train.values, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(
        y_test.values, dtype=torch.float32).unsqueeze(1)

    # Step 4: Define Model
    model = NeuralNet(X_train_tensor.shape[1], hidden_dim=16)

    # Step 5: Train Model
    trainer = ModelTrainer(model, X_train_tensor, y_train_tensor,
                           X_val_tensor, y_val_tensor, num_epochs=500, weight_decay=1e-5)
    trainer.train()
    trainer.plot_losses()

    # Step 6: Test Model Performance
    model.eval()
    with torch.no_grad():
        y_test_pred = model(X_test_tensor)
        y_test_pred_class = (y_test_pred >= 0.5).float()
        accuracy = accuracy_score(y_test_tensor, y_test_pred_class)
        roc_auc = roc_auc_score(y_test_tensor, y_test_pred)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test ROC AUC Score: {roc_auc:.4f}")
