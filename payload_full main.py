import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.modules.rnn as rnn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve
import itertools
import numpy as np

# Import your dataset
data = pd.read_csv('payload_full.csv')

# Preprocessing for textual data
textual_col = 'payload'  # Textual column
vectorizer = TfidfVectorizer()
vectorized_text = vectorizer.fit_transform(data[textual_col])

# Assigning labels to y
y = data['label']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(data[textual_col], y, test_size=0.2, random_state=42)

# Preprocess the data using TF-IDF vectorization
X_train_vectorized = vectorizer.transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Dimensionality reduction using TruncatedSVD
svd = TruncatedSVD(n_components=100)  # Adjust the number of components as needed
X_train_svd = svd.fit_transform(X_train_vectorized)
X_test_svd = svd.transform(X_test_vectorized)

# Train and evaluate Gaussian Naive Bayes model
gnb_model = GaussianNB()
gnb_model.fit(X_train_vectorized.toarray(), y_train)
gnb_predictions = gnb_model.predict(X_test_vectorized.toarray())
gnb_accuracy = accuracy_score(y_test, gnb_predictions)
gnb_report = classification_report(y_test, gnb_predictions)
print("\nGaussian Naive Bayes Accuracy:", gnb_accuracy)
print("Gaussian Naive Bayes Classification Report:\n", gnb_report)

# Train and evaluate Support Vector Machine (SVM) model
svm_model = SVC()
svm_model.fit(X_train_svd, y_train)
svm_predictions = svm_model.predict(X_test_svd)
svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_report = classification_report(y_test, svm_predictions)
print("\nSVM Accuracy:", svm_accuracy)
print("SVM Classification Report:\n", svm_report)

# Train and evaluate Random Forest (RF) model
rf_model = RandomForestClassifier()
rf_model.fit(X_train_svd, y_train)
rf_predictions = rf_model.predict(X_test_svd)
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_report = classification_report(y_test, rf_predictions)
print("\nRandom Forest Accuracy:", rf_accuracy)
print("Random Forest Classification Report:\n", rf_report)

# Train and evaluate K-Nearest Neighbors (KNN) model
knn_model = KNeighborsClassifier()
knn_model.fit(X_train_svd, y_train)
knn_predictions = knn_model.predict(X_test_svd)
knn_accuracy = accuracy_score(y_test, knn_predictions)
knn_report = classification_report(y_test, knn_predictions)
print("\nK-Nearest Neighbors Accuracy:", knn_accuracy)
print("K-Nearest Neighbors Classification Report:\n", knn_report)

# Train and evaluate Logistic Regression (LR) model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_svd, y_train)
lr_predictions = lr_model.predict(X_test_svd)
lr_accuracy = accuracy_score(y_test, lr_predictions)
lr_report = classification_report(y_test, lr_predictions)
print("\nLogistic Regression Accuracy:", lr_accuracy)
print("Logistic Regression Classification Report:\n", lr_report)

# Train and evaluate Decision Tree Classifier
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train_svd, y_train)
dt_predictions = dt_model.predict(X_test_svd)
dt_accuracy = accuracy_score(y_test, dt_predictions)
dt_report = classification_report(y_test, dt_predictions)
print("\nDecision Tree Classifier Accuracy:", dt_accuracy)
print("Decision Tree Classifier Classification Report:\n", dt_report)

# Train and evaluate AdaBoost Classifier
adb_model = AdaBoostClassifier()
adb_model.fit(X_train_svd, y_train)
adb_predictions = adb_model.predict(X_test_svd)
adb_accuracy = accuracy_score(y_test, adb_predictions)
adb_report = classification_report(y_test, adb_predictions)
print("\nAdaBoost Classifier Accuracy:", adb_accuracy)
print("AdaBoost Classifier Classification Report:\n", adb_report)

# Train and evaluate Gradient Boosting Classifier
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train_svd, y_train)
gb_predictions = gb_model.predict(X_test_svd)
gb_accuracy = accuracy_score(y_test, gb_predictions)
gb_report = classification_report(y_test, gb_predictions)
print("\nGradient Boosting Classifier Accuracy:", gb_accuracy)
print("Gradient Boosting Classifier Classification Report:\n", gb_report)

# Train and evaluate Multi-layer Perceptron (MLP) Classifier
mlp_model = MLPClassifier()
mlp_model.fit(X_train_svd, y_train)
mlp_predictions = mlp_model.predict(X_test_svd)
mlp_accuracy = accuracy_score(y_test, mlp_predictions)
mlp_report = classification_report(y_test, mlp_predictions)
print("\nMulti-layer Perceptron (MLP) Classifier Accuracy:", mlp_accuracy)
print("Multi-layer Perceptron (MLP) Classifier Classification Report:\n", mlp_report)

# Train and evaluate Linear Discriminant Analysis (LDA) Classifier
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_train_svd, y_train)
lda_predictions = lda_model.predict(X_test_svd)
lda_accuracy = accuracy_score(y_test, lda_predictions)
lda_report = classification_report(y_test, lda_predictions)
print("\nLinear Discriminant Analysis (LDA) Classifier Accuracy:", lda_accuracy)
print("Linear Discriminant Analysis (LDA) Classifier Classification Report:\n", lda_report)

# Train and evaluate Quadratic Discriminant Analysis (QDA) Classifier
qda_model = QuadraticDiscriminantAnalysis()
qda_model.fit(X_train_svd, y_train)
qda_predictions = qda_model.predict(X_test_svd)
qda_accuracy = accuracy_score(y_test, qda_predictions)
qda_report = classification_report(y_test, qda_predictions)
print("\nQuadratic Discriminant Analysis (QDA) Classifier Accuracy:", qda_accuracy)
print("Quadratic Discriminant Analysis (QDA) Classifier Classification Report:\n", qda_report)


# Mapping string classes to integers
class_mapping = {'anom': 0, 'norm': 1}

# Map true labels to integers
y_test_mapped = y_test.map(class_mapping)
y_train_mapped = y_train.map(class_mapping)

# Train and evaluate XGBoost Classifier
xgb_model = XGBClassifier()
xgb_model.fit(X_train_svd, y_train_mapped)
xgb_predictions = xgb_model.predict(X_test_svd)

# Map XGBoost predictions to integers
xgb_predictions_mapped = pd.Series(xgb_predictions).map({0: 'anom', 1: 'norm'})

# Calculate accuracy and classification report with mapped labels
xgb_accuracy = accuracy_score(y_test, xgb_predictions_mapped)
xgb_report = classification_report(y_test, xgb_predictions_mapped)
print("\nXGBoost Classifier Accuracy:", xgb_accuracy)
print("XGBoost Classifier Classification Report:\n", xgb_report)



# Train and evaluate CatBoost Classifier
catboost_model = CatBoostClassifier()
catboost_model.fit(X_train_svd, y_train)
catboost_predictions = catboost_model.predict(X_test_svd)
catboost_accuracy = accuracy_score(y_test, catboost_predictions)
catboost_report = classification_report(y_test, catboost_predictions)
print("\nCatBoost Classifier Accuracy:", catboost_accuracy)
print("CatBoost Classifier Classification Report:\n", catboost_report)

# Train and evaluate LightGBM Classifier
lgbm_model = LGBMClassifier()
lgbm_model.fit(X_train_svd, y_train)
lgbm_predictions = lgbm_model.predict(X_test_svd)
lgbm_accuracy = accuracy_score(y_test, lgbm_predictions)
lgbm_report = classification_report(y_test, lgbm_predictions)
print("\nLightGBM Classifier Accuracy:", lgbm_accuracy)
print("LightGBM Classifier Classification Report:\n", lgbm_report)

# Mapping string classes to integers
class_mapping = {'anom': 0, 'norm': 1}

# Map true labels to integers
y_train_mapped = y_train.map(class_mapping)
y_test_mapped = y_test.map(class_mapping)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_svd, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_mapped.values, dtype=torch.float32)  # Use mapped labels here
X_test_tensor = torch.tensor(X_test_svd, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_mapped.values, dtype=torch.float32)  # Use mapped labels here


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
for epoch in range(5):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = cnn_model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

# Evaluating the trained model
cnn_model.eval()
with torch.no_grad():
    outputs = cnn_model(X_test_tensor)
    cnn_predictions = torch.round(outputs).numpy().flatten()

# Convert predicted labels to strings
cnn_predictions_strings = ['anom' if pred == 0 else 'norm' for pred in cnn_predictions]

# Calculate accuracy and classification report with string labels
cnn_accuracy = accuracy_score(y_test, cnn_predictions_strings)
cnn_report = classification_report(y_test, cnn_predictions_strings)
print("CNN Model Accuracy:", cnn_accuracy)
print("CNN Model Classification Report:\n", cnn_report)


# Reshape the data for LSTM input
X_train_tensor_lstm = torch.tensor(X_train_vectorized.toarray(), dtype=torch.float32).unsqueeze(1)
y_train_tensor_lstm = torch.tensor(y_train_mapped.values, dtype=torch.float32)  # Use mapped labels here
X_test_tensor_lstm = torch.tensor(X_test_vectorized.toarray(), dtype=torch.float32).unsqueeze(1)
y_test_tensor_lstm = torch.tensor(y_test_mapped.values, dtype=torch.float32)  # Use mapped labels here

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
for epoch in range(5):
    for inputs, labels in train_loader_lstm:
        optimizer_lstm.zero_grad()
        outputs = lstm_model(inputs)
        loss = criterion_lstm(outputs.squeeze(), labels)
        loss.backward()
        optimizer_lstm.step()

# Evaluating the trained LSTM model
lstm_model.eval()
with torch.no_grad():
    outputs = lstm_model(X_test_tensor_lstm)
    lstm_predictions = torch.round(outputs).numpy().flatten()

# Convert numeric predictions to string labels
lstm_predictions_strings = ['anom' if pred == 0 else 'norm' for pred in lstm_predictions]

# Calculate accuracy and classification report with string labels
lstm_accuracy = accuracy_score(y_test, lstm_predictions_strings)
lstm_report = classification_report(y_test, lstm_predictions_strings)
print("LSTM Model Accuracy:", lstm_accuracy)
print("LSTM Model Classification Report:\n", lstm_report)



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

# Instantiate the Bidirectional LSTM model
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
for epoch in range(5):
    for inputs, labels in train_loader_bilstm:
        optimizer_bilstm.zero_grad()
        outputs = bilstm_model(inputs)
        loss = criterion_bilstm(outputs.squeeze(), labels)
        loss.backward()
        optimizer_bilstm.step()

# Evaluating the trained Bidirectional LSTM model
bilstm_model.eval()
with torch.no_grad():
    outputs = bilstm_model(X_test_tensor_lstm)
    bilstm_predictions = torch.round(outputs).numpy().flatten()

# Convert numeric predictions to string labels
bilstm_predictions_strings = ['anom' if pred == 0 else 'norm' for pred in bilstm_predictions]

# Calculate accuracy and classification report with string labels
bilstm_accuracy = accuracy_score(y_test, bilstm_predictions_strings)
bilstm_report = classification_report(y_test, bilstm_predictions_strings)
print("Bidirectional LSTM Model Accuracy:", bilstm_accuracy)
print("Bidirectional LSTM Model Classification Report:\n", bilstm_report)

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
 
# Training the CNN-BiLSTM hybrid model
cnn_bilstm_model.train()
for epoch in range(5):
    for inputs, labels in train_loader:
        optimizer_hybrid.zero_grad()
        outputs = cnn_bilstm_model(inputs)
        loss = criterion_hybrid(outputs.squeeze(), labels)
        loss.backward()
        optimizer_hybrid.step()
 
# Evaluating the trained CNN-BiLSTM hybrid model
cnn_bilstm_model.eval()
with torch.no_grad():
    outputs = cnn_bilstm_model(X_test_tensor)
    cnn_bilstm_predictions = torch.round(outputs).numpy().flatten()
 
# Convert numeric predictions to string labels
cnn_bilstm_predictions_strings = ['anom' if pred == 0 else 'norm' for pred in cnn_bilstm_predictions]
 
# Calculate accuracy and classification report with string labels
cnn_bilstm_accuracy = accuracy_score(y_test, cnn_bilstm_predictions_strings)
cnn_bilstm_report = classification_report(y_test, cnn_bilstm_predictions_strings)
print("CNN-BiLSTM Hybrid Model Accuracy:", cnn_bilstm_accuracy)
print("CNN-BiLSTM Hybrid Model Classification Report:\n", cnn_bilstm_report)
 
from sklearn.ensemble import VotingClassifier

# Define the ensemble model with hard voting
# Hard voting takes a majority vote from the individual models
ensemble_model = VotingClassifier(estimators=[('svm', svm_model), ('rf', rf_model)], voting='hard')

# Train the ensemble model
ensemble_model.fit(X_train_svd, y_train)

# Make predictions using the ensemble model
ensemble_predictions = ensemble_model.predict(X_test_svd)

# Calculate accuracy and classification report for the ensemble model
ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
ensemble_report = classification_report(y_test, ensemble_predictions)

print("\nEnsemble (SVM + Random Forest) Accuracy:", ensemble_accuracy)
print("Ensemble (SVM + Random Forest) Classification Report:\n", ensemble_report)

# Define a function to plot the confusion matrix
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Define a function to convert integer predictions to string labels
def map_predictions_to_labels(predictions):
    return np.array(['anom' if pred == 0 else 'norm' for pred in predictions])

# Confusion matrices
classifiers = {
    'Gaussian Naive Bayes': gnb_predictions,
    'Support Vector Machine (SVM)': svm_predictions,
    'Random Forest': rf_predictions,
    'K-Nearest Neighbors':  knn_predictions,
    'Logistic Regression':  lr_predictions,
    'Decision Tree':dt_predictions,
    'AdaBoost':  adb_predictions,
    'Gradient Boosting':  gb_predictions,
    'Multi-layer Perceptron (MLP)':  mlp_predictions,
    'Linear Discriminant Analysis (LDA)': lda_predictions,
    'Quadratic Discriminant Analysis (QDA)': qda_predictions,
    'XGBoost':  xgb_predictions,
    'CatBoost':  catboost_predictions,
    'LightGBM':  lgbm_predictions,
    'CNN': cnn_predictions,
    'LSTM': lstm_predictions,
    'Bidirectional LSTM':  bilstm_predictions,
    'CNN-BiLSTM Hybrid' : cnn_bilstm_predictions,
    'SVM-RandomForest Hybrid': ensemble_predictions
}

# Plot confusion matrices
plt.figure(figsize=(15, 12))
num_plots = len(classifiers)
num_rows = num_plots // 4 + (1 if num_plots % 4 > 0 else 0)  # Calculate the number of rows needed
for i, (clf_name, clf_preds) in enumerate(classifiers.items(), 1):
    # Convert integer predictions to string labels
    clf_preds_labels = map_predictions_to_labels(clf_preds)
    # Compute confusion matrix
    cm = confusion_matrix(y_test, clf_preds_labels)
    # Plot confusion matrix
    plt.subplot(num_rows, 4, i)
    plot_confusion_matrix(cm, classes=['anom', 'norm'], title=f'Confusion Matrix - {clf_name}')

plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

def plot_roc_curve(y_true, y_pred_proba, model_name):
    # Manually encode string labels to binary format
    y_true_binary = np.where(y_true == 'norm', 1, 0)
    y_pred_proba = np.where(y_pred_proba == 'norm', 1, 0)  # Adjust this line accordingly if y_pred_proba is not in string format
    fpr, tpr, thresholds = roc_curve(y_true_binary, y_pred_proba)
    auc = roc_auc_score(y_true_binary, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')


# Plot ROC curves
plt.figure(figsize=(10, 8))
for model_name, clf_preds in classifiers.items():
    plot_roc_curve(y_test, clf_preds, model_name)  # Pass predicted class labels directly

plt.legend(loc='lower right')
plt.show()

from sklearn.calibration import calibration_curve


# Plot reliability curves
plt.figure(figsize=(10, 8))
for model_name, clf_preds in classifiers.items():
    # Convert string labels to binary format
    y_true_binary = np.where(y_test == 'norm', 1, 0)
    y_pred_proba = np.where(clf_preds == 'norm', 1, 0)  # Adjust this line accordingly if clf_preds is not in string format
    
    # Plot reliability curve
    prob_true, prob_pred = calibration_curve(y_true_binary, y_pred_proba, n_bins=10, strategy='uniform')
    plt.plot(prob_pred, prob_true, marker='o', label=model_name)

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Reliability Curve')
plt.legend(loc='upper left')
plt.show()


print("All classifiers trained and evaluated successfully!")


