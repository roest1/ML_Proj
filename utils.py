'''
Heart Disease Classification

https://www.kaggle.com/code/mragpavank/heart-disease-uci/notebook

'''

########### IMPORTS ##############
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, classification_report, log_loss, precision_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import torch

##################################

############# EDA ################
def get_data() -> pd.DataFrame:
    df = pd.read_csv('./heart.csv')
    # show that no null values exist
    print(f"There are {'' if df.isnull().values.any() else 'not'} missing values in data.")
    return df

def get_descriptive_stats(df:pd.DataFrame) -> pd.DataFrame:
    # stats will contain the 5 number summary:
    # min
    # q1 (median of 1st half)
    # median
    # q3 (median of 2nd half)
    # max
    stats = df.describe().T
    stats['variance'] = df.var() # compute variance
    return stats

# Use this function to view the distribution 
def d_plot_histogram(s:pd.Series, b:int=20):
    # plot histogram with kde overlay using b #bins to view distribution
    # kde overlay helps view the distribution
    sns.histplot(s, kde=True, bins=b)
    plt.title(f"Histogram with KDE for {s.name}")
    plt.show()

# Use this function to determine bandwidth
def c_plot_histogram(s:pd.Series, kernel: str = "epanechnikov", b:int = 20, bw: float = 1):
    fig, ax = plt.subplots(figsize=(7, 6))

    # Histogram
    sns.histplot(s, kde=False, bins=b, stat='density', label='histogram')

    # KDE
    X = s.values[:, np.newaxis]
    X_plot = np.linspace(s.min(), s.max(), len(s))[:, np.newaxis]
    kde = KernelDensity(kernel=kernel, bandwidth=bw).fit(X)
    log_dens = kde.score_samples(X_plot)
    ax.plot(X_plot[:, 0], np.exp(log_dens),
            lw=2, color='r', label=f"KDE (kernel={kernel}, bandwidth={bw})")

    ax.set_title(f"Histogram for {s.name}")
    ax.legend(loc='upper left')
    plt.show()

# Correlation coefficients
def plot_correlation_matrix(df:pd.DataFrame):
    plt.figure(figsize=(12,8))
    m = df.corr()
    sns.heatmap(
        m,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar=True
    )
    plt.title("Correlation Coefficients Heatmap")
    plt.show()
    print(m['target'].sort_values(ascending=False))

# Area under the curve (AUC) receiver operating characteristic (ROC)
# shows how well a feature does at discriminating between target = 1 and target = 0 (whether someone has heart disease)
def ROC(feature:pd.Series, target:pd.Series, plot=True) -> float:
    auc = roc_auc_score(target, feature)
    if plot:
        fpr, tpr, _ = roc_curve(target, feature)
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})', lw=2)
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel("false positive rate")
        plt.ylabel("true positive rate")
        plt.title(f"ROC Curve for {feature.name}")
        plt.legend(loc="lower right")
        plt.show()
        return auc
    else:
        return auc
##################################


######### PREPROCESSING ##########
def normalize_skewed_features(type:str, X_train: pd.DataFrame, X_test: pd.DataFrame, skewed: list) -> (pd.DataFrame, pd.DataFrame):
    if type == "z":
        scaler = StandardScaler()
    elif type == "MinMax":
        scaler = MinMaxScaler()
    else:
        print("Invalid scaler type. Please use \"z\" or \"MinMax\"")
        return (X_train, X_test)
    
    for c in skewed:
        X_train[c] = np.log1p(X_train[c])
        X_test[c] = np.log1p(X_test[c])
        # Fit on training data and just transform on testing data
        X_train[c] = scaler.fit_transform(
            X_train[c].values.reshape(-1, 1)).flatten()
        X_test[c] = scaler.transform(X_test[c].values.reshape(-1, 1)).flatten()
    return X_train, X_test


def normalize_normal_features(type:str, X_train: pd.DataFrame, X_test: pd.DataFrame, normal: list) -> (pd.DataFrame, pd.DataFrame):
    if type == "z":
        scaler = StandardScaler()
    elif type == "MinMax":
        scaler = MinMaxScaler()
    else:
        print("Invalid scaler type. Please use \"z\" or \"MinMax\"")
        return (X_train, X_test)

    for c in normal:
        X_train[c] = scaler.fit_transform(
            X_train[c].values.reshape(-1, 1)).flatten()
        X_test[c] = scaler.transform(X_test[c].values.reshape(-1, 1)).flatten()
    return X_train, X_test
##################################


########### MODELING #############
class LogisticRegressionModel:
    def __init__(self, max_iter):
        self.M = LogisticRegression(max_iter=max_iter, solver='lbfgs', verbose=0)
        self.costs = []
    
    def fit(self, X, y):
        for i in range(self.M.max_iter):
            self.M.max_iter = i + 1
            self.M.fit(X, y)
            y_probs = self.M.predict_proba(X)[:, 1]
            self.costs.append(log_loss(y, y_probs))
    
    def predict(self, X):
        return self.M.predict(X)
    
    def predict_proba(self, X):
        return self.M.predict_proba(X)
    
    def plot_cost(self):
        plt.plot(range(1, len(self.costs) + 1), self.costs)
        plt.title("Cost Function (Log-Loss) During Training")
        plt.xlabel("Iteration")
        plt.ylabel("Cost (Log-Loss)")
        plt.show()

def RunLogisticRegression(X_train, y_train, X_test, y_test, max_iter=10):
    m = LogisticRegressionModel(max_iter)
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    m.plot_cost()
    print(f"Logistic Regression Model Test Accuracy = {accuracy_score(y_test, pred):.2f}\n")
    print(classification_report(y_test, pred))

class NeuralNet(torch.nn.Module):
    def __init__(self, num_input_features, hidden_neurons=64):
        super(NeuralNet, self).__init__()
        self.first_layer = torch.nn.Linear(num_input_features, hidden_neurons)
        self.relu = torch.nn.ReLU()
        self.second_layer = torch.nn.Linear(hidden_neurons, hidden_neurons)
        self.third_layer = torch.nn.Linear(hidden_neurons, 1)
        self.sigmoid = torch.nn.Sigmoid() 

    def forward(self, x):
        x = self.first_layer(x)
        x = self.relu(x)
        x = self.second_layer(x)
        x = self.relu(x)
        x = self.third_layer(x)
        x = self.sigmoid(x)  
        return x


def train_neural_network(model, X_train, y_train, X_val, y_val, num_epochs=250, learning_rate=0.01, verbose=True):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-8)
    criterion = torch.nn.BCELoss()
    best_model_wts = model.state_dict()
    best_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            best_model_wts = model.state_dict()

        if verbose:
            print(
                f'Epoch {epoch+1}/{num_epochs} - train loss: {loss.item():.4f} - val loss: {val_loss.item():.4f}')

    model.load_state_dict(best_model_wts)

    if verbose:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_losses) + 1),
                 train_losses, label='Training Loss')
        plt.plot(range(1, len(val_losses) + 1),
                 val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.xlim(0, len(train_losses) + 1)
        plt.ylabel('Loss (Binary Cross Entropy)')
        plt.yscale('log')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.show()

    return model

def RunNeuralNetwork(X_train:pd.DataFrame, y_train:pd.Series, X_val:pd.DataFrame, y_val:pd.Series, X_test:pd.DataFrame, y_test:pd.Series, num_epochs:int=200, lr:float=0.001, verbose:bool=True):
    
    X_train_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val.to_numpy(), dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
    
    model = NeuralNet(X_train_tensor.shape[1], hidden_neurons=64)
    print("Training Neural Network...")
    best_model = train_neural_network(
        model, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor,
        num_epochs=200,
        learning_rate=0.001,
        verbose=True,
    )
    best_model.eval()
    with torch.no_grad():
        y_test_pred = best_model(X_test_tensor)
        y_test_pred_class = (y_test_pred >= 0.5).float()
        test_accuracy = (y_test_pred_class == y_test_tensor).float().mean()
        print(f'Neural Network Test Accuracy: {test_accuracy.item():.4f}')

        # For classification report
        y_test_pred_class_np = y_test_pred_class.cpu().numpy()
        y_test_tensor_np = y_test_tensor.cpu().numpy()
        print(classification_report(y_test_tensor_np, y_test_pred_class_np))


##################################
    
