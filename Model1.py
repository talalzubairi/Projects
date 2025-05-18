import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, chi2_contingency
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import gradio as gr

data = pd.read_csv('TaxiRideShare .csv')
print("Dataset Info:")
print(data.info())

# Count missing values in each column
missing_values = data.isnull().sum()

# Display the count of missing values
print("Missing Values:")
print(missing_values)
print("\nFirst few rows of the dataset:")
print(data.head())

# Get the shape of the data
data_shape = data.shape

# Display the size of the data
print("Size of the data (rows, columns):", data_shape)
# Count missing values in each column
missing_values = data.isnull().sum()

# Calculate average missingness across all columns combined
average_missingness_combined = (data.isnull().mean().mean() * 100).round(2)

# Display the result
print("Average Missingness across all Columns Combined:", average_missingness_combined, "%")

# Include NA counts if 'NA' is present in the data
if 'NA' in data.values:
    missing_values['NA'] = len(data[data.isin(['NA']).any(axis=1)])

# Create a bar plot
plt.figure(figsize=(10, 6))
missing_values.plot(kind='bar')
plt.title('Missing Values Count')
plt.xlabel('Columns')
plt.ylabel('Number of Missing Values')
plt.xticks(rotation=45)
plt.show()
# Remove rows with any missing values
data_cleaned = data.dropna()
# Get the shape of the data
data_shape = data_cleaned.shape

# Display the size of the cleaned data
print("Size of the data (rows, columns):", data_shape)
# Count missing values in each column
missing_values = data.isnull().sum()

# Replace 'NA' and '?' with NaN
data_cleaned.replace(['NA', '?'], np.nan, inplace=True)

missing_values = data_cleaned.isnull().sum()

# Display the count of missing values
print("Missing Values (including 'NA'):")
print(missing_values)

# Get the shape of the data
data_shape = data_cleaned.shape

data_cleaned.dropna(inplace=True)

try:
  data_cleaned['price'] = pd.to_numeric(data_cleaned['price'], errors='coerce')
except:
  print("Error: Object column contains non-numeric values.")

data_type=data.dtypes
print(data_type)

print(data_cleaned.info())

# Display the size of the cleaned data
print("Size of the data (rows, columns):", data_shape)

data_cleaned['price'] = data_cleaned['price'].astype(float)

timestamp_values = data_cleaned['timestamp'].values.reshape(-1, 1)

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the data
scaled_timestamp = scaler.fit_transform(timestamp_values)

# Visualizing each column against 'price'
plt.subplot(3, 3, 1)
sns.scatterplot(data=data_cleaned, x='distance', y='price')
plt.title('Distance vs Price')

plt.subplot(3, 3, 2)
sns.scatterplot(data=data_cleaned, x='temperature', y='price')
plt.title('Temperature vs Price')

plt.subplot(3, 3, 3)
sns.scatterplot(data=data_cleaned, x='precipProbability', y='price')
plt.title('Precipitation Probability vs Price')

plt.subplot(3, 3, 4)
sns.scatterplot(data=data_cleaned, x='windSpeed', y='price')
plt.title('Wind Speed vs Price')

plt.subplot(3, 3, 5)
sns.scatterplot(data=data_cleaned, x='visibility', y='price')
plt.title('Visibility vs Price')

plt.subplot(3, 3, 6)
sns.scatterplot(data=data_cleaned, x='pressure', y='price')
plt.title('Pressure vs Price')

plt.subplot(3, 3, 7)
sns.scatterplot(data=data_cleaned, x='surge_multiplier', y='price')
plt.title('Surge Multiplier vs Price')

plt.tight_layout()
plt.show()

# Add the scaled timestamp values to data_cleaned DataFrame
data_cleaned['timestamp_scaled'] = scaled_timestamp

# Select only numeric columns
numeric_columns = data_cleaned.select_dtypes(include=['float64', 'int64'])

# Calculate the correlation matrix
correlation_matrix = numeric_columns.corr()

# Create a heatmap for the correlation matrix
plt.figure(figsize=(15, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=1.5)
plt.title('Correlation Matrix of Cleaned Data')
plt.show()

categorical_column = data_cleaned['datetime']
continuous_column = data_cleaned['timestamp']

# Perform ANOVA test
f_statistic, p_value = f_oneway(*[continuous_column[categorical_column == category] for category in categorical_column.unique()])

print(f"F-statistic: {f_statistic}")
print(f"p-value: {p_value}")

categorical_vars = ['datetime', 'source', 'destination', 'cab_type', 'name', 'short_summary' ]

# Chi test for categorical variables
for var in categorical_vars:
   contingency_table = pd.crosstab(data_cleaned[var], data_cleaned['price'])
   chi2, pval, deg_of_freedom, expected_freq = chi2_contingency(contingency_table.values)

   # Check the p-value. A value below a threshold (e.g., 0.05) indicates a significant association.
   print("Chi-Square Test for", var, ":")
   if pval < 0.05:
       print("  - There is a significant association between", var, "and 'price' (p-value:", pval, ")")
   else:
       print("  - There is no significant association between", var, "and 'price' (p-value:", pval, ")")


selected_vars = ['timestamp', 'source', 'destination', 'cab_type', 'name', 'price', 'distance', 'temperature',
                 'precipProbability', 'windSpeed', 'visibility', 'pressure', 'surge_multiplier']
df = data_cleaned[selected_vars]

# Convert categorical variables to factors (i.e., one-hot encoding)
categorical_vars = ['source', 'destination', 'cab_type', 'name']
df = pd.get_dummies(df, columns=categorical_vars)

# Splitting the data into features (X) and target variable (y)
X = df.drop(columns=['price'])
y = df['price']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the Random Forest regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Making predictions on the test set
y_pred = rf_regressor.predict(X_test)

# Evaluating the model
mse_rf = mean_squared_error(y_test, y_pred)
print("Mean Squared Error(Random Forest):", mse_rf)
r_squared_rf = r2_score(y_test, y_pred)
print("R-squared Score:(Random Forest)", r_squared_rf)

# Creating and training the Decision Tree regressor
dt_regressor = DecisionTreeRegressor(random_state=42)
dt_regressor.fit(X_train, y_train)

# Making predictions on the test set
y_pred_dt = dt_regressor.predict(X_test)

# Evaluating the model
mse_dt = mean_squared_error(y_test, y_pred_dt)
print("Mean Squared Error (Decision Tree):", mse_dt)
r_squared_dt = r2_score(y_test, y_pred_dt)
print("R-squared Score (Decision Tree):", r_squared_dt)

# Scatter plot for Random Forest model
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Actual vs. Predicted (Random Forest)')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs. Predicted Prices (Random Forest)')
plt.legend()
plt.show()

# Scatter plot for Decision Tree model
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_dt, color='red', label='Actual vs. Predicted (Decision Tree)')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs. Predicted Prices (Decision Tree)')
plt.legend()
plt.show()

# Create a dictionary to store the results
results = {
    'Model': ['Random Forest', 'Decision Tree'],
    'Mean Squared Error': [mse_rf, mse_dt],
    'R-squared Score': [r_squared_rf, r_squared_dt]
}

# Convert the dictionary to a pandas DataFrame
results_df = pd.DataFrame(results)

# Display the DataFrame
print(results_df)

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='neg_mean_squared_error')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# Plot learning curves for Random Forest
plot_learning_curve(rf_regressor, "Learning Curve (Random Forest)", X_train, y_train, cv=5)
plt.show()

# Plot learning curves for Decision Tree
plot_learning_curve(dt_regressor, "Learning Curve (Decision Tree)", X_train, y_train, cv=5)
plt.show()

print(df.columns)
def price(timestamp, distance, temperature, precipProbability, windSpeed, visibility, pressure,
          surge_multiplier, source_Back_Bay, source_Beacon_Hill, source_Boston_University, source_Fenway,
          source_Financial_District, source_Haymarket_Square, source_North_End, source_North_Station,
          source_Northeastern_University, source_South_Station, source_Theatre_District, source_West_End,
          destination_Back_Bay, destination_Beacon_Hill, destination_Boston_University, destination_Fenway,
          destination_Financial_District, destination_Haymarket_Square, destination_North_End,
          destination_North_Station, destination_Northeastern_University, destination_South_Station,
          destination_Theatre_District, destination_West_End, cab_type_Lyft, cab_type_Uber, name_Black,
          name_Black_SUV, name_Lux, name_Lux_Black, name_Lux_Black_XL, name_Lyft, name_Lyft_XL, name_Shared,
          name_UberPool, name_UberX, name_UberXL, name_WAV):
    # Define Gradio Inputs
    # Prepare input features as a DataFrame
    input_data = pd.DataFrame({
        'timestamp': [timestamp],
        'distance': [distance],
        'temperature': [temperature],
        'precipProbability': [precipProbability],
        'windSpeed': [windSpeed],
        'visibility': [visibility],
        'pressure': [pressure],
        'surge_multiplier': [surge_multiplier],
        'source_Back Bay': [source_Back_Bay],
        'source_Beacon Hill': [source_Beacon_Hill],
        'source_Boston University': [source_Boston_University],
        'source_Fenway': [source_Fenway],
        'source_Financial District': [source_Financial_District],
        'source_Haymarket Square': [source_Haymarket_Square],
        'source_North End': [source_North_End],
        'source_North Station': [source_North_Station],
        'source_Northeastern University': [source_Northeastern_University],
        'source_South Station': [source_South_Station],
        'source_Theatre District': [source_Theatre_District],
        'source_West End': [source_West_End],
        'destination_Back Bay': [destination_Back_Bay],
        'destination_Beacon Hill': [destination_Beacon_Hill],
        'destination_Boston University': [destination_Boston_University],
        'destination_Fenway': [destination_Fenway],
        'destination_Financial District': [destination_Financial_District],
        'destination_Haymarket Square': [destination_Haymarket_Square],
        'destination_North End': [destination_North_End],
        'destination_North Station': [destination_North_Station],
        'destination_Northeastern University': [destination_Northeastern_University],
        'destination_South Station': [destination_South_Station],
        'destination_Theatre District': [destination_Theatre_District],
        'destination_West End': [destination_West_End],
        'cab_type_Lyft': [cab_type_Lyft],
        'cab_type_Uber': [cab_type_Uber],
        'name_Black': [name_Black],
        'name_Black SUV': [name_Black_SUV],
        'name_Lux': [name_Lux],
        'name_Lux Black': [name_Lux_Black],
        'name_Lux Black XL': [name_Lux_Black_XL],
        'name_Lyft': [name_Lyft],
        'name_Lyft XL': [name_Lyft_XL],
        'name_Shared': [name_Shared],
        'name_UberPool': [name_UberPool],
        'name_UberX': [name_UberX],
        'name_UberXL': [name_UberXL],
        'name_WAV': [name_WAV]
    })
    # Make prediction using the churn model
    prediction = rf_regressor.predict(input_data)
    return prediction[0]
# Define the churn prediction function
inputs = [
    gr.Number(label="TimeStamp"),
    gr.Number(label="Distance"),
    gr.Number(label="Temperature"),
    gr.Number(label="Precip Probability"),
    gr.Number(label="Wind Speed"),
    gr.Number(label="Visibility"),
    gr.Number(label="Pressure"),
    gr.Number(label="Surge Multiplier"),
    gr.Checkbox(label="Source Back Bay"),
    gr.Checkbox(label="Source Beacon Hill"),
    gr.Checkbox(label="Source Boston University"),
    gr.Checkbox(label="Source Fenway"),
    gr.Checkbox(label="Source Financial District"),
    gr.Checkbox(label="Source Haymarket Square"),
    gr.Checkbox(label="Source North End"),
    gr.Checkbox(label="Source North Station"),
    gr.Checkbox(label="Source Northeastern University"),
    gr.Checkbox(label="Source South Station"),
    gr.Checkbox(label="Source Theatre District"),
    gr.Checkbox(label="Source West End"),
    gr.Checkbox(label="Destination Back Bay"),
    gr.Checkbox(label="Destination Beacon Hill"),
    gr.Checkbox(label="Destination Boston University"),
    gr.Checkbox(label="Destination Fenway"),
    gr.Checkbox(label="Destination Financial District"),
    gr.Checkbox(label="Destination Haymarket Square"),
    gr.Checkbox(label="Destination North End"),
    gr.Checkbox(label="Destination North Station"),
    gr.Checkbox(label="Destination Northeastern University"),
    gr.Checkbox(label="Destination South Station"),
    gr.Checkbox(label="Destination Theatre District"),
    gr.Checkbox(label="Destination West End"),
    gr.Checkbox(label="Cab Type Lyft"),
    gr.Checkbox(label="Cab Type Uber"),
    gr.Checkbox(label="Name Black"),
    gr.Checkbox(label="Name Black SUV"),
    gr.Checkbox(label="Name Lux"),
    gr.Checkbox(label="Name Lux Black"),
    gr.Checkbox(label="Name Lux Black XL"),
    gr.Checkbox(label="Name Lyft"),
    gr.Checkbox(label="Name Lyft XL"),
    gr.Checkbox(label="Name Shared"),
    gr.Checkbox(label="Name UberPool"),
    gr.Checkbox(label="Name UberX"),
    gr.Checkbox(label="Name UberXL"),
    gr.Checkbox(label="Name WAV")
]

#Define Gradio Output
output = gr.Textbox(label="Predicted Price")

#Create Gradio Interface
app = gr.Interface(fn=price, inputs=inputs, outputs=output, title="Price Prediction", description="Please only select one option for each category in the checkboxes.")
app.launch(share=True)