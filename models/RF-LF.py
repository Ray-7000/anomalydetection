import pandas as pd
from scipy.io import arff
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Load ARFF files
train_data, train_meta = arff.loadarff('/Users/carlostorres/Documents/Python Project/Data Sets/Annthyroid/Annthyroid_02_v10.arff')
test_data, test_meta = arff.loadarff('/Users/carlostorres/Documents/Python Project/Data Sets/Annthyroid/Annthyroid_withoutdupl_norm_07.arff')

# Convert to DataFrames
train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

# Convert byte strings to regular strings for train set
for col in train_df.columns:
    if train_df[col].dtype == 'object':
        train_df[col] = train_df[col].str.decode('utf-8')

# Convert byte strings to regular strings for test set
for col in test_df.columns:
    if test_df[col].dtype == 'object':
        test_df[col] = test_df[col].str.decode('utf-8')

# Convert 'yes'/'no' to 1/0 in the outlier column for train and test sets
train_df['outlier'] = train_df['outlier'].map({'yes': 1, 'no': 0})
test_df['outlier'] = test_df['outlier'].map({'yes': 1, 'no': 0})

# Drop ID column if it exists
if 'id' in train_df.columns:
    train_df = train_df.drop(columns=['id'])
if 'id' in test_df.columns:
    test_df = test_df.drop(columns=['id'])

# Separate features and labels for train set
X_train = train_df.drop(columns=['outlier'])  # Drop outlier column
y_train = train_df['outlier']

# Separate features and labels for test set
X_test = test_df.drop(columns=['outlier'])  # Drop outlier column
y_test = test_df['outlier']

# Standardize the features for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Start measuring time
start_time = time.time()

# Initialize classifiers
rf = RandomForestClassifier(n_estimators=100, random_state=42)
lr = LogisticRegression(max_iter=1000, random_state=42)

# Create cross-validated predictions for stacking
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf_cv_predictions = cross_val_predict(rf, X_train, y_train, cv=cv, method='predict_proba')[:, 1]
lr_cv_predictions = cross_val_predict(lr, X_train_scaled, y_train, cv=cv, method='predict_proba')[:, 1]

# Combine cross-validated predictions as new features
stacked_features_train = pd.DataFrame({
    'rf_pred': rf_cv_predictions,
    'lr_pred': lr_cv_predictions
})

# Train the meta-model (Logistic Regression) on the stacked features
stacked_lr = LogisticRegression(max_iter=1000, random_state=42)
stacked_lr.fit(stacked_features_train, y_train)

# Fit the base classifiers on the entire training set
rf.fit(X_train, y_train)
lr.fit(X_train_scaled, y_train)

# Make predictions on the test set using base classifiers
rf_test_predictions = rf.predict_proba(X_test)[:, 1]
lr_test_predictions = lr.predict_proba(X_test_scaled)[:, 1]

# Combine test set predictions as new features
stacked_features_test = pd.DataFrame({
    'rf_pred': rf_test_predictions,
    'lr_pred': lr_test_predictions
})

# Make final predictions using the meta-model
final_predictions = stacked_lr.predict(stacked_features_test)

# End measuring time
end_time = time.time()
runtime = end_time - start_time

# Add predictions to the test dataframe for visualization
test_df['anomaly'] = final_predictions

# Evaluate the model
print(classification_report(y_test, final_predictions, target_names=['Normal', 'Anomaly']))
print(confusion_matrix(y_test, final_predictions))

# Print runtime
print(f"Runtime for anomaly detection: {runtime:.2f} seconds")

# Visualize the anomalies using scatter plot (example using the first two features)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=test_df, x=test_df.columns[0], y=test_df.columns[1], hue='anomaly', palette={0: 'blue', 1: 'red'})
plt.title("Anomaly Detection")
plt.xlabel(test_df.columns[0])
plt.ylabel(test_df.columns[1])
plt.legend(title='Anomaly', loc='upper left', labels=['Normal', 'Anomaly'])
plt.show()

# Obtain anomaly scores from the second-stage model (in this case, probabilities of being an anomaly)
anomaly_scores = stacked_lr.predict_proba(stacked_features_test)[:, 1]

# Calculate AUC score
auc_score = roc_auc_score(y_test, anomaly_scores)

# Print AUC score
print("AUC Score:", auc_score)
