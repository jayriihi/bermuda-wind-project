import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to bin wind directions
def bin_wind_directions(df, direction_column='nmb Dir', bin_size=5):
    """
    Bin wind directions into specified intervals.

    Parameters:
        df (pd.DataFrame): DataFrame containing wind data.
        direction_column (str): Column for wind direction.
        bin_size (int): Bin size for wind directions (degrees).

    Returns:
        pd.DataFrame: DataFrame with added binned direction column.
    """
    # Define bins and labels
    dir_bins = np.arange(0, 360 + bin_size, bin_size)
    dir_labels = [(dir_bins[i] + dir_bins[i + 1]) / 2 for i in range(len(dir_bins) - 1)]
    df['dir_bin'] = pd.cut(df[direction_column], bins=dir_bins, labels=dir_labels, right=False)
    return df

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
file_path = '/Users/jayriihiluoma/Documents/python/scrapers/NMB_Crescent_data /combined_data_cleaned_no_repeats.csv'

# Load data
df = pd.read_csv(file_path)

# Calculate wind speed difference
df['WS Difference'] = df['nmb WS'] - df['cres WS']

# Bin wind directions
df_binned = bin_wind_directions(df, direction_column='nmb Dir', bin_size=5)

# Analyze wind speed differences by direction bins
stats = analyze_wind_speed_difference_by_bins(df_binned)

# Print summary statistics
print(stats)
