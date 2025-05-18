import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc

# Read the data from the CSV file
df = pd.read_csv("train.csv")

# Select only numerical columns
numerical_cols = [col for col in df.columns if df[col].dtype != 'object']

# Calculate correlation matrix
correlation_matrix = df[numerical_cols].corr()

#Missing Values
missing_values = df.isnull().sum()
print("Missing values per column:")
print(missing_values)

# Calculate the average missingness across all columns
average_missingness = missing_values.mean()
print("Average missingness per column:")
print(average_missingness)

# Create a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True)
plt.show()

# List of variables to keep for the model
selected_vars = ['Churn', 'SubscriptionType', 'PaymentMethod', 'ContentType', 'GenrePreference',
                'Gender', 'ParentalControl', 'SubtitlesEnabled', 'AccountAge', 'TotalCharges',
                'AverageViewingDuration', 'ContentDownloadsPerMonth', 'SupportTicketsPerMonth',
                'ViewingHoursPerWeek', 'UserRating']

# Select only the columns we want to keep
df = df[selected_vars]

# Preprocess 'Gender' column
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

# Convert categorical variables to factors
categorical_vars = ['SubscriptionType', 'PaymentMethod', 'ContentType', 'GenrePreference', 'ParentalControl', 'SubtitlesEnabled']
df = pd.get_dummies(df, columns=categorical_vars)

# Split the data into predictor variables (X) and the target variable (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Sample 5% of the data for training
sample_size = round(0.05 * len(df))
train_data = df.sample(n=sample_size, random_state=123)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.02, random_state=123)

# Build logistic regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)

# Predict on test data
predictions = logistic_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Plot the logistic model coefficients
plt.plot(logistic_model.coef_.flatten())
plt.xlabel('Coefficients')
plt.ylabel('Values')
plt.title('Logistic Regression Coefficients')
plt.show()

# Compute predicted probabilities for X_test
probabilities = logistic_model.predict_proba(X_test)
# Keep probabilities of positive class only
probabilities = probabilities[:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, probabilities)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

print("AUC:", roc_auc)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:\n", conf_matrix)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Compute additional metrics
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)

print("\nPrecision:", precision)
print("Recall:", recall)