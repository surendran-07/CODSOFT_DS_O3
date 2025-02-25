import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load the sales dataset
sales_file_path = "advertising (2).csv"  # Update this with the correct path
sales_df = pd.read_csv(sales_file_path)

# Remove duplicates and handle missing values
sales_df = sales_df.drop_duplicates()
sales_df = sales_df.dropna()

# Save the cleaned sales dataset
sales_df.to_csv("advertising_cleaned.csv", index=False)

# Define features and target variable
X_sales = sales_df[['TV', 'Radio', 'Newspaper']]
y_sales = sales_df['Sales']

# Split the sales data into training and testing sets
X_train_sales, X_test_sales, y_train_sales, y_test_sales = train_test_split(X_sales, y_sales, test_size=0.2, random_state=42)

# Train a Random Forest Regressor model for sales prediction
sales_model = RandomForestRegressor(n_estimators=100, random_state=42)
sales_model.fit(X_train_sales, y_train_sales)

# Make predictions for sales
y_pred_sales = sales_model.predict(X_test_sales)

# Evaluate the sales model
mae_sales = mean_absolute_error(y_test_sales, y_pred_sales)
r2_sales = r2_score(y_test_sales, y_pred_sales)

print(f"Sales Prediction - Mean Absolute Error: {mae_sales}")
print(f"Sales Prediction - R² Score: {r2_sales}")

# Print actual vs predicted values for sales
predictions_df_sales = pd.DataFrame({"Actual": y_test_sales.values, "Predicted": y_pred_sales})
print("Sales - Actual vs Predicted:")
print(predictions_df_sales.head(10))
