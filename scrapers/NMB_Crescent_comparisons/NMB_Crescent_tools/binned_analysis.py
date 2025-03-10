import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to bin directions and wind speeds
def bin_wind_data(df, direction_column='nmb Dir', speed_column='cres WS', 
                  dir_bin_size=10, speed_bin_size=5):
    """
    Bin wind directions and wind speeds into specified intervals.

    Parameters:
        df (pd.DataFrame): DataFrame containing wind data.
        direction_column (str): Column for wind direction.
        speed_column (str): Column for wind speed.
        dir_bin_size (int): Bin size for wind directions (degrees).
        speed_bin_size (int): Bin size for wind speeds (knots).

    Returns:
        pd.DataFrame: DataFrame with added binned columns for directions and speeds.
    """
    # Bin directions
    dir_bins = np.arange(0, 360 + dir_bin_size, dir_bin_size)
    dir_labels = [(dir_bins[i] + dir_bins[i + 1]) / 2 for i in range(len(dir_bins) - 1)]
    df['dir_bin'] = pd.cut(df[direction_column], bins=dir_bins, labels=dir_labels, right=False)

    # Bin wind speeds
    speed_bins = np.arange(0, df[speed_column].max() + speed_bin_size, speed_bin_size)
    speed_labels = [(speed_bins[i] + speed_bins[i + 1]) / 2 for i in range(len(speed_bins) - 1)]
    df['speed_bin'] = pd.cut(df[speed_column], bins=speed_bins, labels=speed_labels, right=False)

    return df

# Function to analyze correlation between binned directions and binned speeds
def analyze_binned_correlation(df, direction_bin_column='dir_bin', speed_bin_column='speed_bin'):
    """
    Analyze correlation between binned wind directions and binned wind speeds.

    Parameters:
        df (pd.DataFrame): DataFrame with binned wind data.
        direction_bin_column (str): Column for binned wind directions.
        speed_bin_column (str): Column for binned wind speeds.

    Returns:
        pd.DataFrame: Pivot table showing mean wind speed for each direction bin.
    """
    # Create a pivot table
    pivot_table = df.pivot_table(
        values='WS Difference',
        index=direction_bin_column,
        columns=speed_bin_column,
        aggfunc='mean'
    )

    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, cmap='coolwarm', annot=True, fmt='.2f', cbar_kws={'label': 'Mean WS Difference'})
    plt.title('Mean Wind Speed Difference by Direction and Speed Bins')
    plt.xlabel('Crescent Wind Speed Bins (kt)')
    plt.ylabel('NMB Wind Direction Bins (°)')
    plt.xticks(rotation=45)
    plt.show()

    return pivot_table

# Main Workflow
file_path = '/Users/jayriihiluoma/Documents/python/scrapers/NMB_Crescent_comparisons/NMB_Crescent_data/binned_combined_data.csv'
df = pd.read_csv(file_path)
df['WS Difference'] = df['nmb WS'] - df['cres WS']

# Bin wind directions and speeds
df_binned = bin_wind_data(df, direction_column='nmb Dir', speed_column='cres WS')

# Analyze binned correlation
pivot_table = analyze_binned_correlation(df_binned)

# Print pivot table
print(pivot_table)

