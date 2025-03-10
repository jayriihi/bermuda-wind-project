import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt


def train_models_for_bins(df, dir_bin_column='dir_bin', output_dir='bin_models/'):
    """
    Train a separate model for each wind direction bin.

    Parameters:
        df (pd.DataFrame): Filtered data.
        dir_bin_column (str): Column for wind direction bins.
        output_dir (str): Directory to save the trained models.

    Returns:
        dict: A dictionary with bin labels as keys and model RMSE as values.
    """
    os.makedirs(output_dir, exist_ok=True)
    bin_models = {}
    bin_metrics = {}

    for bin_label in df[dir_bin_column].unique():
        # Filter data for the current bin
        bin_data = df[df[dir_bin_column] == bin_label]

        if len(bin_data) < 20:  # Skip bins with insufficient data
            print(f"Skipping bin {bin_label} due to insufficient data.")
            continue

        # Define features and target
        X = bin_data[['cres WS']]
        y = bin_data['WS Difference']

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        # Save the model
        model_path = os.path.join(output_dir, f'bin_{bin_label}_model.pkl')
        joblib.dump(model, model_path)

        # Log metrics
        bin_models[bin_label] = model
        bin_metrics[bin_label] = rmse

        #print(f"Model for bin {bin_label}: RMSE = {rmse:.2f} (saved to {model_path})")

    return bin_metrics


if __name__ == "__main__":
    # Define paths
    filtered_data_path = '/Users/jayriihiluoma/Documents/python/scrapers/NMB_Crescent_comparisons/NMB_Crescent_data/filtered_combined_data.csv'
    output_dir = '/Users/jayriihiluoma/Documents/python/scrapers/NMB_Crescent_comparisons/NMB_Crescent_data/bin_models/'

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load filtered data
    df = pd.read_csv(filtered_data_path)

    # Ensure 'dir_bin' column is properly formatted
    df['dir_bin'] = df['dir_bin'].astype(str)

    # Calculate variability metrics for WS Difference
    ws_diff_std = df['WS Difference'].std()
    ws_diff_range = df['WS Difference'].max() - df['WS Difference'].min()
    print(f"Standard Deviation of WS Difference: {ws_diff_std:.2f}")
    print(f"Range of WS Difference: {ws_diff_range:.2f}")
    print("Interpreting RMSE of 2.18:")
    if ws_diff_std > 5:
        print("The RMSE is quite acceptable compared to the high variability (std > 5 knots).")
    else:
        print("The RMSE is relatively high compared to the low variability in WS Difference.")

    # Train models for each bin
    metrics = train_models_for_bins(df, output_dir=output_dir)

    '''
    This section gives an analysis of the model 
    # Analyze RMSE distribution
    rmse_values = np.array([rmse for rmse in metrics.values() if not np.isnan(rmse)])
    #print(f"\nMean RMSE: {rmse_values.mean():.2f}, Std Dev RMSE: {rmse_values.std():.2f}")

    # Save RMSE summary
    summary_df = pd.DataFrame(list(metrics.items()), columns=['Bin', 'RMSE'])
    summary_path = os.path.join(output_dir, 'bin_model_performance_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Bin model performance summary saved to '{summary_path}'")

    # Plot RMSE across bins
    plt.figure(figsize=(12, 6))
    plt.bar(summary_df['Bin'], summary_df['RMSE'])
    plt.xlabel('Direction Bin')
    plt.ylabel('RMSE')
    plt.title('Model Performance by Bin')
    plt.xticks(rotation=90)
    plt.grid(axis='y')
    plt.show()

    # Print summary of model performance
    print("\nBin Model Performance Summary:")
    for bin_label, rmse in metrics.items():
        print(f"Bin {bin_label}: RMSE = {rmse:.2f}")

    # Plot the histogram of WS Difference
    plt.hist(df['WS Difference'], bins=30, color='blue', alpha=0.7)
    plt.axvline(2.18, color='red', linestyle='--', label='RMSE = 2.18')
    plt.title('Distribution of WS Difference')
    plt.xlabel('WS Difference (knots)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()'''


