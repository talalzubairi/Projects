import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, r2_score
import numpy as np
import gradio as gr
from sklearn.tree import export_graphviz
import graphviz

# Read the data from the CSV file
df = pd.read_csv("train.csv")

# Select only numerical columns
numerical_cols = [col for col in df.columns if df[col].dtype != 'object']

# Calculate correlation matrix
correlation_matrix = df[numerical_cols].corr()

# Create a heatmap
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
sns.heatmap(correlation_matrix, annot=True)  # Add annotations to show values on the heatmap
plt.show()

# Categorical Variables
categorical_vars = ['SubscriptionType', 'PaymentMethod', 'PaperlessBilling',
                   'ContentType', 'MultiDeviceAccess', 'DeviceRegistered',
                   'GenrePreference', 'Gender', 'ParentalControl', 'SubtitlesEnabled']

for var in categorical_vars:
   contingency_table = pd.crosstab(df[var], df['Churn'])  # Create contingency table
   chi2, pval, deg_of_freedom, expected_freq = chi2_contingency(contingency_table.values)

   # Check the p-value. A value below a threshold (e.g., 0.05) indicates a significant association.
   print("Chi-Square Test for", var, ":")
   if pval < 0.05:
       print("  - There is a significant association between", var, "and 'churn' (p-value:", pval, ")")
   else:
       print("  - There is no significant association between", var, "and 'churn' (p-value:", pval, ")")

# List of variables to keep for the model
selected_vars = ['Churn', 'SubscriptionType', 'PaymentMethod', 'ContentType', 'GenrePreference',
                'Gender', 'ParentalControl', 'SubtitlesEnabled', 'AccountAge', 'TotalCharges',
                'AverageViewingDuration', 'ContentDownloadsPerMonth', 'SupportTicketsPerMonth',
                'ViewingHoursPerWeek', 'UserRating']

# Select only the columns we want to keep
df = df[selected_vars]

# Convert categorical variables to factors (i.e., one-hot encoding)
categorical_vars = ['Gender', 'SubscriptionType', 'PaymentMethod', 'ContentType', 'GenrePreference', 'ParentalControl', 'SubtitlesEnabled']
df = pd.get_dummies(df, columns=categorical_vars)
# print(df.columns)
# Split the data into predictor variables (X) and the target variable (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Sample 6% of the data for training
sample_size = round(0.06 * len(df))
train_data = df.sample(n=sample_size, random_state=123)
print(len(train_data))

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.02, random_state=123)

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=123)

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train)

# Predict on the test data
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

from sklearn.metrics import precision_score
# Calculate precision
precision = precision_score(y_test, y_pred)
print("Precision:", precision)

# Calculate predicted probabilities
y_probs = rf_classifier.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
# Calculate AUC-ROC score
auc_score = roc_auc_score(y_test, y_probs)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Calculate prediction errors
errors = y_test != y_pred

# Convert boolean errors to integers (False -> 0, True -> 1)
errors_int = errors.astype(int)

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix as heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

print(len(y_test), len(y_pred))

# Getting our columns
print(df.columns)

# Define the churn prediction function
def Churn(AccountAge, TotalCharges, AverageViewingDuration, ContentDownloadsPerMonth, SupportTicketsPerMonth,
          ViewingHoursPerWeek, UserRating, Gender_Female, Gender_Male, SubscriptionType_Basic, SubscriptionType_Premium,
          SubscriptionType_Standard, PaymentMethod_Bank_transfer, PaymentMethod_Credit_card,
          PaymentMethod_Electronic_check, PaymentMethod_Mailed_check, ContentType_Both, ContentType_Movies,
          ContentType_TV_Shows, GenrePreference_Action, GenrePreference_Comedy, GenrePreference_Drama,
          GenrePreference_Fantasy, GenrePreference_Sci_Fi, ParentalControl_No, ParentalControl_Yes, SubtitlesEnabled_No,
          SubtitlesEnabled_Yes):
    # Prepare input features as a DataFrame
    input_data = pd.DataFrame({
        'AccountAge': [AccountAge],
        'TotalCharges': [TotalCharges],
        'AverageViewingDuration': [AverageViewingDuration],
        'ContentDownloadsPerMonth': [ContentDownloadsPerMonth],
        'SupportTicketsPerMonth': [SupportTicketsPerMonth],
        'ViewingHoursPerWeek': [ViewingHoursPerWeek],
        'UserRating': [UserRating],
        'Gender_Female': [Gender_Female],
        'Gender_Male': [Gender_Male],
        'SubscriptionType_Basic': [SubscriptionType_Basic],
        'SubscriptionType_Premium': [SubscriptionType_Premium],
        'SubscriptionType_Standard': [SubscriptionType_Standard],
        'PaymentMethod_Bank_transfer': [PaymentMethod_Bank_transfer],
        'PaymentMethod_Credit_card': [PaymentMethod_Credit_card],
        'PaymentMethod_Electronic_check': [PaymentMethod_Electronic_check],
        'PaymentMethod_Mailed_check': [PaymentMethod_Mailed_check],
        'ContentType_Both': [ContentType_Both],
        'ContentType_Movies': [ContentType_Movies],
        'ContentType_TV_Shows': [ContentType_TV_Shows],
        'GenrePreference_Action': [GenrePreference_Action],
        'GenrePreference_Comedy': [GenrePreference_Comedy],
        'GenrePreference_Drama': [GenrePreference_Drama],
        'GenrePreference_Fantasy': [GenrePreference_Fantasy],
        'GenrePreference_Sci_Fi': [GenrePreference_Sci_Fi],
        'ParentalControl_No': [ParentalControl_No],
        'ParentalControl_Yes': [ParentalControl_Yes],
        'SubtitlesEnabled_No': [SubtitlesEnabled_No],
        'SubtitlesEnabled_Yes': [SubtitlesEnabled_Yes]
    })

    # Make prediction using the churn model
    prediction = rf_classifier.predict(input_data)
    return prediction[0]

# Define Gradio inputs
inputs = [
    gr.Slider(0, 200, label="Account Age"),
    gr.Number(label="Total Charges"),
    gr.Number(label="Average Viewing Duration"),
    gr.Slider(0, 25, label="Content Downloads Per Month"),
    gr.Slider(0, 15, label="Support Tickets Per Month"),
    gr.Number(label="Viewing Hours Per Week"),
    gr.Number(label="User Rating"),
    gr.Checkbox(label="Female"),
    gr.Checkbox(label="Male"),
    gr.Checkbox(label="SubscriptionType Basic"),
    gr.Checkbox(label="SubscriptionType Premium"),
    gr.Checkbox(label="SubscriptionType Standard"),
    gr.Checkbox(label="PaymentMethod Bank transfer"),
    gr.Checkbox(label="PaymentMethod Credit card"),
    gr.Checkbox(label="PaymentMethod Electronic check"),
    gr.Checkbox(label="PaymentMethod Mailed check"),
    gr.Checkbox(label="ContentType Both"),
    gr.Checkbox(label="ContentType Movies"),
    gr.Checkbox(label="ContentType TV Shows"),
    gr.Checkbox(label="GenrePreference Action"),
    gr.Checkbox(label="GenrePreference Comedy"),
    gr.Checkbox(label="GenrePreference Drama"),
    gr.Checkbox(label="GenrePreference Fantasy"),
    gr.Checkbox(label="GenrePreference Sci-Fi"),
    gr.Checkbox(label="ParentalControl No"),
    gr.Checkbox(label="ParentalControl Yes"),
    gr.Checkbox(label="SubtitlesEnabled No"),
    gr.Checkbox(label="SubtitlesEnabled Yes")
]

# Define Gradio output
output = gr.Textbox(label="Churn Prediction")

# Create Gradio Interface
app = gr.Interface(fn=Churn, inputs=inputs, outputs=output, title="Customer Churn Prediction", description="Please only select one option for each category in the checkboxes.")
app.launch(share=True)