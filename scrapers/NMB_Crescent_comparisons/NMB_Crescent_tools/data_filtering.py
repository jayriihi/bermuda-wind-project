import numpy as np
import pandas as pd
import os

# Function to load and preprocess data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Date Time (UTC)'] = pd.to_datetime(df['Date Time (UTC)'])
    df.set_index('Date Time (UTC)', inplace=True)
    return df

# Function to bin wind directions
def bin_wind_directions(df, direction_column='nmb Dir', bin_size=5):
    dir_bins = np.arange(0, 360 + bin_size, bin_size)
    dir_labels = [(dir_bins[i] + dir_bins[i + 1]) / 2 for i in range(len(dir_bins) - 1)]
    df['dir_bin'] = pd.cut(df[direction_column], bins=dir_bins, labels=dir_labels, right=False)
    return df

# Function to remove repeated Crescent data
def remove_repeated_crescent_data(df, ws_col='cres WS', gust_col='cres Gust', dir_col='cres Dir', max_repeats=2):
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
    return df[keep_rows].copy()

# Function to filter outliers
def filter_outliers(df):
    outlier_threshold = 10  # Define acceptable WS Difference range
    df = df[(df['WS Difference'] >= -outlier_threshold) & (df['WS Difference'] <= outlier_threshold)]
    return df

# Main Workflow
if __name__ == "__main__":
    file_path = '/Users/jayriihiluoma/Documents/python/scrapers/NMB_Crescent_comparisons/NMB_Crescent_data/combined_data_cleaned.csv'
    output_dir = '/Users/jayriihiluoma/Documents/python/scrapers/NMB_Crescent_comparisons/NMB_Crescent_data/'

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load and preprocess data
    df = load_data(file_path)

    # Remove repeated Crescent data
    cleaned_df = remove_repeated_crescent_data(df)

    # Drop rows where `nmb WS` or `cres WS` is missing
    cleaned_df = cleaned_df.dropna(subset=['nmb WS', 'cres WS'])

    # Add wind speed difference
    cleaned_df["WS Difference"] = cleaned_df["nmb WS"] - cleaned_df["cres WS"]

    # Bin wind directions
    cleaned_df = bin_wind_directions(cleaned_df, direction_column='nmb Dir', bin_size=5)

    # Remove rows with NaN in dir_bin or invalid wind direction values
    cleaned_df = cleaned_df.dropna(subset=['dir_bin'])



    # Filter outliers
    filtered_df = filter_outliers(cleaned_df)

    # Save the filtered data
    filtered_data_path = os.path.join(output_dir, 'filtered_combined_data.csv')
    filtered_df.to_csv(filtered_data_path, index=True)
    print(f"Filtered data saved to '{filtered_data_path}'")

    # Print summary
    print(f"Original data points: {len(df)}")
    print(f"Filtered data points: {len(filtered_df)}")
    print(f"Data points removed: {len(df) - len(filtered_df)}")
