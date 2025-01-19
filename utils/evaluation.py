import pandas as pd
import numpy as np

# Load the truth and generated CSV files
truth_path = '../data/yelp_val.csv'
prediction_path = '../data/hybrid_pred.csv'  # Change the file path to the path of your generated file

truth_df = pd.read_csv(truth_path)
generated_df = pd.read_csv(prediction_path)

# Merge the data on user_id and business_id to align the predicted and actual ratings
merged_df = pd.merge(truth_df, generated_df, on=['user_id', 'business_id'])

# Calculate RMSE using the correct column names
rmse = np.sqrt(((merged_df['stars'] - merged_df['prediction']) ** 2).mean())
print("RMSE:", rmse)