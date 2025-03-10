import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os


# Function to load and preprocess data
def load_data(file_path):
    """
    Load and preprocess the dataset.
    - Reads the CSV file.
    - Sets the 'Date Time (UTC)' column as datetime and index.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    df = pd.read_csv(file_path)
    df['Date Time (UTC)'] = pd.to_datetime(df['Date Time (UTC)'])
    df.set_index('Date Time (UTC)', inplace=True)
    return df

def bin_wind_directions(df, direction_column='nmb Dir', bin_size=5):
    dir_bins = np.arange(0, 360 + bin_size, bin_size)
    dir_labels = [(dir_bins[i] + dir_bins[i + 1]) / 2 for i in range(len(dir_bins) - 1)]
    df['dir_bin'] = pd.cut(df[direction_column], bins=dir_bins, labels=dir_labels, right=False)
    return df



# Function to add direction_bias feature
def add_direction_bias(df, direction_column='nmb Dir', threshold=10):
    """
    Add a direction bias feature based on observed wind speed differences.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        direction_column (str): Name of the column for wind direction.
        threshold (int): Range around 200° where bias is set to 0.

    Returns:
        pd.DataFrame: DataFrame with the new 'direction_bias' column.
    """
    df['direction_bias'] = 5.0
    mask = (df[direction_column] >= 200 - threshold) & (df[direction_column] <= 200 + threshold)
    df.loc[mask, 'direction_bias'] = 0
    return df


# Function to remove repeated Crescent data
def remove_repeated_crescent_data(df, ws_col='cres WS', gust_col='cres Gust', dir_col='cres Dir', max_repeats=2):
    """
    Remove Crescent data rows where the same wind speed, gust, and direction 
    are repeated more than `max_repeats` times consecutively.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        ws_col (str): Column name for Crescent wind speed.
        gust_col (str): Column name for Crescent wind gust.
        dir_col (str): Column name for Crescent wind direction.
        max_repeats (int): Maximum allowed consecutive repeats before dropping rows.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    repeat_counter = 0
    keep_rows = [True]

    for i in range(1, len(df)):
        is_repeated = (
            df.iloc[i][ws_col] == df.iloc[i - 1][ws_col] and
            df.iloc[i][gust_col] == df.iloc[i - 1][gust_col] and
            df.iloc[i][dir_col] == df.iloc[i - 1][dir_col]
        )
        if is_repeated:
            repeat_counter += 1
        else:
            repeat_counter = 0
        keep_rows.append(repeat_counter <= max_repeats)

    cleaned_df = df[keep_rows].copy()
    return cleaned_df


'''# Function to train Random Forest model
def train_random_forest_model(df):
    """
    Train a Random Forest model with the direction_bias feature.

    Parameters:
        df (pd.DataFrame): Input DataFrame with features and target.

    Returns:
        RandomForestRegressor: Trained Random Forest model.
    """
    df = add_direction_bias(df)
    X = df[['cres WS', 'nmb Dir', 'direction_bias']]
    y = df['nmb WS']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"Random Forest RMSE with direction_bias: {rmse:.2f}")
    joblib.dump(rf_model, "wind_model_with_bias.pkl")
    return rf_model'''

# Function to analyze wind speed differences by direction bins
def analyze_wind_speed_difference_by_bins(df, dir_bin_column='dir_bin', ws_diff_column='WS Difference'):
    """
    Analyze wind speed differences by direction bins.

    Parameters:
        df (pd.DataFrame): DataFrame containing binned wind data.
        dir_bin_column (str): Column for binned wind directions.
        ws_diff_column (str): Column for wind speed difference.

    Returns:
        pd.DataFrame: Summary statistics for wind speed differences by direction bins.
    """
    # Group by direction bins and compute statistics
    stats = df.groupby(dir_bin_column)[ws_diff_column].agg(['mean', 'std', 'count']).reset_index()
    stats.rename(columns={'mean': 'Mean WS Difference', 'std': 'Std Dev WS Difference', 'count': 'Count'}, inplace=True)

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.errorbar(
        stats[dir_bin_column], stats['Mean WS Difference'], 
        yerr=stats['Std Dev WS Difference'], fmt='o', capsize=4, label='Mean WS Difference (±1 Std Dev)'
    )
    plt.axhline(0, color='red', linestyle='--', label='Zero Difference')
    plt.title('Mean Wind Speed Difference vs NMB Wind Direction (Binned by 5°)')
    plt.xlabel('NMB Wind Direction Bins (°)')
    plt.ylabel('Mean Wind Speed Difference (kt)')
    plt.legend()
    plt.grid()
    plt.show()

    return stats


# Main Workflow
file_path = '/Users/jayriihiluoma/Documents/python/scrapers/NMB_Crescent_comparisons/NMB_Crescent_data/combined_data_cleaned.csv'
output_dir = '/Users/jayriihiluoma/Documents/python/scrapers/NMB_Crescent_comparisons/NMB_Crescent_data/'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load the data
df = load_data(file_path)

# Remove repeated Crescent data
cleaned_df = remove_repeated_crescent_data(df)

# Drop rows where `nmb WS` or `cres WS` is missing
cleaned_df = cleaned_df.dropna(subset=['nmb WS', 'cres WS'])

# Add wind speed difference
cleaned_df["WS Difference"] = cleaned_df["nmb WS"] - cleaned_df["cres WS"]

# Bin wind directions
cleaned_df = bin_wind_directions(cleaned_df, direction_column='nmb Dir', bin_size=5)

# Compute bin statistics
bin_stats = cleaned_df.groupby('dir_bin', observed=False)['WS Difference'].agg(['mean', 'std', 'count']).reset_index()

# Filter bins with fewer than 20 observations
valid_bins = bin_stats[bin_stats['count'] >= 20]

# Apply dynamic thresholds to filter extreme WS differences
valid_bins = valid_bins[(valid_bins['mean'] <= 10) & (valid_bins['mean'] >= -10)]

# Filter cleaned_df based on valid bins
filtered_df = cleaned_df[cleaned_df['dir_bin'].isin(valid_bins['dir_bin'])]

# Analyze wind speed differences by direction bins (using filtered data)
stats = analyze_wind_speed_difference_by_bins(filtered_df)

# Save filtered data for further analysis
filtered_data_path = os.path.join(output_dir, 'filtered_combined_data.csv')
filtered_df.to_csv(filtered_data_path, index=True)

print(f"Filtered data saved to '{filtered_data_path}'")



'''# Train Random Forest with direction_bias
rf_model = train_random_forest_model(cleaned_df)

# Feature Importance Analysis
feature_importances = rf_model.feature_importances_
feature_names = ['cres WS', 'nmb Dir', 'direction_bias']
for name, importance in zip(feature_names, feature_importances):
    print(f"Feature: {name}, Importance: {importance:.2f}")

# Save cleaned data for further analysis
cleaned_data_path = os.path.join(output_dir, 'combined_data_cleaned_with_bias_and_filtered.csv')
cleaned_df.to_csv(cleaned_data_path, index=True)

# Save summary statistics to CSV
summary = cleaned_df.describe()
summary_path = os.path.join(output_dir, 'summary_statistics_filtered.csv')
summary.to_csv(summary_path)

# Save the trained Random Forest model
model_path = os.path.join(output_dir, 'wind_model_with_bias.pkl')
joblib.dump(rf_model, model_path)

# Print confirmation messages
print(f"Cleaned data saved to '{cleaned_data_path}'")
print(f"Summary statistics saved to '{summary_path}'")
print(f"Model saved to '{model_path}'")'''

