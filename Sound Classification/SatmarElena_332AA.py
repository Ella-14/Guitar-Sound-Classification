import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Function to extract MFCC features from audio files
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return mfccs_processed
    except Exception as e:
        print("Error encountered while parsing file:", file_path)
        print(e)
        return None

# Directory paths for minor and major chord audio files
minor_dir = 'Audio_Files/Training/Minor'
major_dir = 'Audio_Files/Training/Major'

# Lists to hold features and corresponding labels
X = []  # Features
y = []  # Labels (0 for minor, 1 for major)

# Extract features and labels for minor chords
for filename in os.listdir(minor_dir):
    if filename.endswith('.wav'):
        file_path = os.path.join(minor_dir, filename)
        features = extract_features(file_path)
        if features is not None:
            X.append(features)
            y.append(0)  # Assign label 0 for minor chords

# Extract features and labels for major chords
for filename in os.listdir(major_dir):
    if filename.endswith('.wav'):
        file_path = os.path.join(major_dir, filename)
        features = extract_features(file_path)
        if features is not None:
            X.append(features)
            y.append(1)  # Assign label 1 for major chords

# Convert lists to NumPy arrays
X = np.array(X)
y = np.array(y)

# Splitting data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Perform cross-validation on the training set
svm_model = SVC(C=10000000, kernel="poly", gamma="scale")
cv_scores = cross_val_score(svm_model, X_train, y_train, cv=5)  # Perform 5-fold cross-validation

# Print cross-validation scores
print("Cross-Validation Scores:", cv_scores)
print("Mean Cross-Validation Score:", np.mean(cv_scores))

# Train the SVM model on the entire training set
best_svm = svm_model.fit(X_train, y_train)  # Assign the trained model to 'best_svm'

# Make predictions on the test set
best_predictions = best_svm.predict(X_test)
# Train the SVM model on the entire training set
svm_model.fit(X_train, y_train)

# Evaluate model performance on the test set
accuracy = accuracy_score(y_test, best_predictions)
print("Accuracy on Test Set:", accuracy)

# Classification report on test set
print("Classification Report on Test Set:")
print(classification_report(y_test, best_predictions))

# Confusion matrix on test set
cm_test = confusion_matrix(y_test, best_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, cmap='Reds', fmt='d', xticklabels=['Minor', 'Major'], yticklabels=['Minor', 'Major'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix on Test Set')
plt.show()

# 1. ROC Curve and AUC Score
from sklearn.metrics import roc_curve, auc

# Compute ROC curve and AUC for the best model
fpr, tpr, _ = roc_curve(y_test, best_svm.decision_function(X_test))
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
from sklearn.metrics import precision_recall_curve, auc
precision, recall, thresholds = precision_recall_curve(y_test, best_svm.decision_function(X_test))

# Compute Area Under the Curve (AUC) for Precision-Recall curve
pr_auc = auc(recall, precision)

# Plot Precision-Recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='green', lw=2, label=f'Precision-Recall curve (AUC = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.show()
# 3. Display Misclassified Samples (if applicable)
misclassified_indices = np.where(y_test != best_predictions)[0]
print("Misclassified values:")
# Display some misclassified samples
for idx in misclassified_indices[:5]:
    print(f"Sample: {idx}, Predicted Label: {best_predictions[idx]}, True Label: {y_test[idx]}")
    # Display or visualize the misclassified sample if applicable

print("\nCorectly classified values:")
classified_indices = np.where(y_test == best_predictions)[0]
# Display some classified samples
for idx in classified_indices[:5]:
    print(f"Sample: {idx}, Predicted Label: {best_predictions[idx]}, True Label: {y_test[idx]}")
    # Display or visualize the misclassified sample if applicable
# 4. Class Distribution with Counts for Training and Test Sets
# Calculate class distribution for the whole dataset
unique_classes, class_counts = np.unique(y, return_counts=True)
class_distribution = dict(zip(unique_classes, class_counts))

# Calculate class distribution for the training set
unique_train, train_counts = np.unique(y_train, return_counts=True)
train_distribution = dict(zip(unique_train, train_counts))

# Calculate class distribution for the test set
unique_test, test_counts = np.unique(y_test, return_counts=True)
test_distribution = dict(zip(unique_test, test_counts))

# Plot class distribution for the whole dataset
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
sns.countplot(x=y)
plt.title('Class Distribution (Whole Dataset)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(list(class_distribution.keys()))

# Show class distribution for training and test sets in numbers
plt.subplot(1, 2, 2)
plt.bar(class_distribution.keys(), class_distribution.values(), label='Whole Dataset', alpha=0.6)
plt.bar(train_distribution.keys(), train_distribution.values(), label='Training Set', alpha=0.8)
plt.bar(test_distribution.keys(), test_distribution.values(), label='Test Set', alpha=0.8)
plt.title('Class Distribution: Training vs Test')
plt.xlabel('Class')
plt.ylabel('Count')
plt.legend()
plt.xticks(list(class_distribution.keys()))

# Annotate bars with values
for i, count in enumerate(class_distribution.values()):
    plt.text(i, count + 5, str(count), ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()

# Hyperparameters to test
param_grid = [
    {'C': [1, 10, 100, 1000, 10000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000, 10000], 'gamma': ['scale', 'auto'], 'kernel': ['rbf', 'poly']}
]

# Perform GridSearchCV to find the best model
grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best model and its hyperparameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Visualize model performance with varying hyperparameters
results = grid_search.cv_results_
param_values = results['param_C']
param_gamma = results['param_gamma']
mean_test_scores = results['mean_test_score']

plt.figure(figsize=(10, 6))
for i, (param, gamma) in enumerate(zip(param_values, param_gamma)):
    plt.scatter(i, mean_test_scores[i], label=f'C={param}, Gamma={gamma}' if gamma else f'C={param}', s=100)

plt.title('Model Performance with Different Hyperparameters')
plt.xlabel('Parameter Combinations')
plt.ylabel('Mean Test Accuracy')
plt.xticks(np.arange(len(mean_test_scores)), labels=np.arange(len(mean_test_scores)) + 1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print best hyperparameters and corresponding accuracy
print("\nBest Hyperparameters:", best_params)
print("Corresponding Accuracy:", grid_search.best_score_)

