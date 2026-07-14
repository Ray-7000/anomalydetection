import pandas as pd
from scipy.io import arff
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Load ARFF file
data, meta = arff.loadarff('/Users/carlostorres/Documents/Python Project/Data Sets/WPBC/WPBC_withoutdupl_norm.arff')
df = pd.DataFrame(data)

# Convert byte strings to strings (if needed)
df = df.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

# Assuming the last column is the 'outlier' column
df.columns = [f'feature_{i}' for i in range(df.shape[1] - 1)] + ['outlier']

# Convert 'yes'/'no' to 1/0 in the outlier column if necessary
df['outlier'] = df['outlier'].map({'yes': 1, 'no': 0})

# Separate features and labels
X = df.drop(columns=['outlier'])  # Drop outlier column
y = df['outlier'].astype(int)  # Ensure y is integer type

# Start measuring time
start_time = time.time()

# Apply Isolation Forest for anomaly detection
iso_forest = IsolationForest(n_estimators=100, max_samples=256, random_state=42)
iso_forest.fit(X)

# Predict anomalies
predictions = iso_forest.predict(X)
predictions = pd.Series(predictions).map({1: 0, -1: 1})  # 1 for anomaly, 0 for normal

# End measuring time
end_time = time.time()
runtime = end_time - start_time

# Add predictions to the dataframe for visualization
df['anomaly'] = predictions

# Evaluate the model
print(classification_report(y, predictions, target_names=['Normal', 'Anomaly']))
print(confusion_matrix(y, predictions))

# Print runtime
print(f"Runtime for anomaly detection: {runtime:.2f} seconds")

# Visualize the anomalies using scatter plot (example using the first two features)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x=df.columns[0], y=df.columns[1], hue='anomaly', palette={0: 'blue', 1: 'red'})
plt.title("Anomaly Detection")
plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])
plt.legend(title='Anomaly', loc='upper left', labels=['Normal', 'Anomaly'])
plt.show()

# Obtain anomaly scores
anomaly_scores = iso_forest.decision_function(X)

# Calculate AUC score
auc_score = roc_auc_score(y, -anomaly_scores)  # The '-' sign is used because Isolation Forest returns negative anomaly scores

# Print AUC score
print("AUC Score:", auc_score)
