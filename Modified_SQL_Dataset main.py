import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# Load your data
data = pd.read_csv('Modified_SQL_Dataset.csv', encoding='utf-8-sig')
X = data['Query']
y = data['Label']

# Handling missing values
data['Query'].fillna('', inplace=True)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data using TF-IDF vectorization
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Dimensionality reduction using TruncatedSVD
svd = TruncatedSVD(n_components=100)  # Adjust the number of components as needed
X_train_svd = svd.fit_transform(X_train_vectorized)
X_test_svd = svd.transform(X_test_vectorized)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_svd, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_svd, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Define the CNN model architecture
class CNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Instantiate the CNN model
input_dim = X_train_tensor.shape[1]
output_dim = 1
cnn_model = CNN(input_dim, output_dim)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

# Create DataLoader for training
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Training the CNN model
cnn_model.train()
start_train_time = time.time()
for epoch in range(5):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = cnn_model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
end_train_time = time.time()

# Evaluating the trained model
cnn_model.eval()
start_eval_time = time.time()
with torch.no_grad():
    outputs = cnn_model(X_test_tensor)
    cnn_predictions = torch.round(outputs).numpy().flatten()
end_eval_time = time.time()

# Calculate accuracy and classification report
cnn_accuracy = accuracy_score(y_test, cnn_predictions)
cnn_report = classification_report(y_test, cnn_predictions)
print("CNN Model Accuracy:", cnn_accuracy)
print("CNN Model Classification Report:\n", cnn_report)
print("CNN Model Training Time:", end_train_time - start_train_time, "seconds")
print("CNN Model Evaluation Time:", end_eval_time - start_eval_time, "seconds")

# Reshape the data for LSTM input
X_train_tensor_lstm = torch.tensor(X_train_vectorized.toarray(), dtype=torch.float32).unsqueeze(1)
y_train_tensor_lstm = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor_lstm = torch.tensor(X_test_vectorized.toarray(), dtype=torch.float32).unsqueeze(1)
y_test_tensor_lstm = torch.tensor(y_test.values, dtype=torch.float32)

# Define the LSTM model architecture
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        output = self.sigmoid(output)
        return output

# Instantiate the LSTM model
input_dim_lstm = X_train_tensor_lstm.shape[2]
hidden_dim_lstm = 128
output_dim_lstm = 1
lstm_model = LSTMClassifier(input_dim_lstm, hidden_dim_lstm, output_dim_lstm)

# Define loss function and optimizer for LSTM
criterion_lstm = nn.BCELoss()
optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=0.001)

# Create DataLoader for LSTM training
train_dataset_lstm = TensorDataset(X_train_tensor_lstm, y_train_tensor_lstm)
train_loader_lstm = DataLoader(train_dataset_lstm, batch_size=64, shuffle=True)

# Training the LSTM model
lstm_model.train()
start_train_time = time.time()
for epoch in range(5):
    for inputs, labels in train_loader_lstm:
        optimizer_lstm.zero_grad()
        outputs = lstm_model(inputs)
        loss = criterion_lstm(outputs.squeeze(), labels)
        loss.backward()
        optimizer_lstm.step()
end_train_time = time.time()

# Evaluating the trained LSTM model
lstm_model.eval()
start_eval_time = time.time()
with torch.no_grad():
    outputs = lstm_model(X_test_tensor_lstm)
    lstm_predictions = torch.round(outputs).numpy().flatten()
end_eval_time = time.time()

# Calculate accuracy and classification report for LSTM
lstm_accuracy = accuracy_score(y_test, lstm_predictions)
lstm_report = classification_report(y_test, lstm_predictions)
print("LSTM Model Accuracy:", lstm_accuracy)
print("LSTM Model Classification Report:\n", lstm_report)
print("LSTM Model Training Time:", end_train_time - start_train_time, "seconds")
print("LSTM Model Evaluation Time:", end_eval_time - start_eval_time, "seconds")

# Define the Bidirectional LSTM model architecture
class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BiLSTMClassifier, self).__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.bilstm(x)
        output = self.fc(lstm_out[:, -1, :])
        output = self.sigmoid(output)
        return output

# Instantiate the BiLSTM model
input_dim_bilstm = X_train_tensor_lstm.shape[2]
hidden_dim_bilstm = 128
output_dim_bilstm = 1
bilstm_model = BiLSTMClassifier(input_dim_bilstm, hidden_dim_bilstm, output_dim_bilstm)

# Define loss function and optimizer for Bidirectional LSTM
criterion_bilstm = nn.BCELoss()
optimizer_bilstm = optim.Adam(bilstm_model.parameters(), lr=0.001)

# Create DataLoader for Bidirectional LSTM training
train_dataset_bilstm = TensorDataset(X_train_tensor_lstm, y_train_tensor_lstm)
train_loader_bilstm = DataLoader(train_dataset_bilstm, batch_size=64, shuffle=True)

# Training the Bidirectional LSTM model
bilstm_model.train()
start_train_time = time.time()
for epoch in range(5):
    for inputs, labels in train_loader_bilstm:
        optimizer_bilstm.zero_grad()
        outputs = bilstm_model(inputs)
        loss = criterion_bilstm(outputs.squeeze(), labels)
        loss.backward()
        optimizer_bilstm.step()
end_train_time = time.time()

# Evaluating the trained Bidirectional LSTM model
bilstm_model.eval()
start_eval_time = time.time()
with torch.no_grad():
    outputs = bilstm_model(X_test_tensor_lstm)
    bilstm_predictions = torch.round(outputs).numpy().flatten()
end_eval_time = time.time()

# Calculate accuracy and classification report for Bidirectional LSTM
bilstm_accuracy = accuracy_score(y_test, bilstm_predictions)
bilstm_report = classification_report(y_test, bilstm_predictions)
print("Bidirectional LSTM Model Accuracy:", bilstm_accuracy)
print("Bidirectional LSTM Model Classification Report:\n", bilstm_report)
print("Bidirectional LSTM Model Training Time:", end_train_time - start_train_time, "seconds")
print("Bidirectional LSTM Model Evaluation Time:", start_eval_time - end_eval_time, "seconds")

# Define the CNN-BiLSTM hybrid model architecture
class CNNBiLSTM(nn.Module):
    def __init__(self, input_dim, cnn_output_dim, lstm_hidden_dim, output_dim):
        super(CNNBiLSTM, self).__init__()
        # CNN layers
        self.cnn_fc1 = nn.Linear(input_dim, cnn_output_dim)
        self.cnn_relu = nn.ReLU()
        self.cnn_sigmoid = nn.Sigmoid()
        # BiLSTM layers
        self.bilstm = nn.LSTM(cnn_output_dim, lstm_hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(lstm_hidden_dim * 2, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # CNN forward pass
        cnn_output = self.cnn_sigmoid(self.cnn_fc1(x))
        # Reshape output for LSTM
        cnn_output = cnn_output.unsqueeze(1)
        # BiLSTM forward pass
        lstm_out, _ = self.bilstm(cnn_output)
        output = self.fc(lstm_out[:, -1, :])
        output = self.sigmoid(output)
        return output

# Instantiate the CNN-BiLSTM hybrid model
input_dim_hybrid = X_train_tensor.shape[1]  # Use the same input dimension as the CNN model
cnn_output_dim = 128  # Adjust as needed
lstm_hidden_dim = 128  # Adjust as needed
output_dim_hybrid = 1  # Same as other models
cnn_bilstm_model = CNNBiLSTM(input_dim_hybrid, cnn_output_dim, lstm_hidden_dim, output_dim_hybrid)

# Define loss function and optimizer for CNN-BiLSTM
criterion_hybrid = nn.BCELoss()
optimizer_hybrid = optim.Adam(cnn_bilstm_model.parameters(), lr=0.001)

# Create DataLoader for CNN-BiLSTM training
train_dataset_hybrid = TensorDataset(X_train_tensor, y_train_tensor)
train_loader_hybrid = DataLoader(train_dataset_hybrid, batch_size=64, shuffle=True)

# Training the CNN-BiLSTM hybrid model
cnn_bilstm_model.train()
start_train_time = time.time()
for epoch in range(5):
    for inputs, labels in train_loader_hybrid:
        optimizer_hybrid.zero_grad()
        outputs = cnn_bilstm_model(inputs)
        loss = criterion_hybrid(outputs.squeeze(), labels)
        loss.backward()
        optimizer_hybrid.step()
end_train_time = time.time()

# Evaluating the trained CNN-BiLSTM hybrid model
cnn_bilstm_model.eval()
start_eval_time = time.time()
with torch.no_grad():
    outputs = cnn_bilstm_model(X_test_tensor)
    cnn_bilstm_predictions = torch.round(outputs).numpy().flatten()
end_eval_time = time.time()

# Calculate accuracy and classification report for CNN-BiLSTM
cnn_bilstm_accuracy = accuracy_score(y_test, cnn_bilstm_predictions)
cnn_bilstm_report = classification_report(y_test, cnn_bilstm_predictions)
print("CNN-BiLSTM Hybrid Model Accuracy:", cnn_bilstm_accuracy)
print("CNN-BiLSTM Hybrid Model Classification Report:\n", cnn_bilstm_report)
print("CNN-BiLSTM Hybrid Model Training Time:", end_train_time - start_train_time, "seconds")
print("CNN-BiLSTM Hybrid Model Evaluation Time:", end_eval_time - start_eval_time, "seconds")

# Create SVM and RF models
svm_model = SVC(probability=True)  # Enable probability estimates for SVM
rf_model = RandomForestClassifier()

# Create a voting classifier with SVM and RF using soft voting
voting_classifier = VotingClassifier(estimators=[('svm', svm_model), ('rf', rf_model)], voting='soft')

# Train the voting classifier
start_train_time = time.time()
voting_classifier.fit(X_train_svd, y_train)
end_train_time = time.time()

# Train and evaluate SVM-RF hybrid model
start_eval_time = time.time()
svm_rf_predictions = voting_classifier.predict(X_test_svd)
end_eval_time = time.time()

# Evaluate the hybrid model
svm_rf_accuracy = accuracy_score(y_test, svm_rf_predictions)
svm_rf_report = classification_report(y_test, svm_rf_predictions)
print("\nSVM-RF Hybrid Model Accuracy:", svm_rf_accuracy)
print("SVM-RF Hybrid Model Classification Report:\n", svm_rf_report)
print("SVM-RF Hybrid Model Training Time:", end_train_time - start_train_time, "seconds")
print("SVM-RF Hybrid Model Evaluation Time:", end_eval_time - start_eval_time, "seconds")
