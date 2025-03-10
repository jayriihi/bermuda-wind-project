'''import pandas as pd

# Load the text file into a DataFrame
input_file = '/Users/jayriihiluoma/Documents/python/scrapers/NMB Crescent data /2023.09.01-2024.08.31-CresentAWOS.txt'  # Replace with your actual file name

headers = ['Date Time (UTC)', 'Wind Speed', 'Wind Gust', 'Wind Direction']  # Replace with actual column names
df = pd.read_csv(input_file, sep="\t", header=None, names=headers)

df = pd.read_csv(input_file, sep="\t")  # Use "\t" for tab-separated or adjust if space-separated

# Show all columns
#pd.set_option('display.max_columns', None)

# This gives details of the data frame
#print(df.info())

headers = ['Date Time (UTC)', 'Wind Speed', 'Wind Gust', 'Wind Direction']
df = pd.read_csv(input_file, sep="\t", header=None, names=headers, skiprows=1)  # Adjust skiprows if necessary

# Drop unnecessary columns like "index" (if exists)
if 'index' in df.columns:
    df.drop(columns=['index'], inplace=True)

df.reset_index(inplace=True)

# Parse the datetime column
df['Date Time (UTC)'] = pd.to_datetime(df['Date Time (UTC)'], errors='coerce')

# Set the datetime column as the DataFrame index
df.set_index('Date Time (UTC)', inplace=True)

# Align timestamps to the nearest 5-minute interval
df.index = df.index.round('5min')

# Resample the data to 5-minute intervals, taking the mean for numerical columns
df_resampled = df.resample('5min').mean()

#print(df.columns)
#print(type(df.index))




# Resample the data to 5-minute intervals, using the mean for numerical columns
#df_resampled = df.resample('5T').mean()

# Optional: Reset the index if you want the datetime column back as a regular column
#df_resampled.reset_index(inplace=True)

# Display the first few rows to confirm the data is loaded correctly
print(df.head(30))

#with open(input_file, 'r') as f:
#    for _ in range(5):
#        print(f.readline())'''

'''import pandas as pd

# Load the text file into a DataFrame
input_file = '/Users/jayriihiluoma/Documents/python/scrapers/NMB Crescent data /2023.09.01-2024.08.31-CresentAWOS.txt'

# Define headers for the columns
headers = ['Date Time (UTC)', 'Wind Speed', 'Wind Gust', 'Wind Direction']

# Load the file with the correct separator and headers
df = pd.read_csv(input_file, sep="\t", header=None, names=headers, skiprows=1, nrows=100000)

# Drop unnecessary "index" column if it exists
if 'index' in df.columns:
    df.drop(columns=['index'], inplace=True)

# Parse the datetime column, ensuring invalid entries are dropped
df['Date Time (UTC)'] = pd.to_datetime(df['Date Time (UTC)'], errors='coerce')
df.dropna(subset=['Date Time (UTC)'], inplace=True)

# Set the datetime column as the DataFrame index
df.set_index('Date Time (UTC)', inplace=True)

# Resample the data to 5-minute intervals
# Aggregate with the mean for numerical columns
df_resampled = df.resample('5min').mean()

# Optional: Reset the index if you want the datetime back as a column
df_resampled.reset_index(inplace=True)

# Round Wind Speed and Wind Gust to one decimal place
df_resampled['Wind Speed'] = df_resampled['Wind Speed'].round(1)
df_resampled['Wind Gust'] = df_resampled['Wind Gust'].round(1)

# Round Wind Direction to the nearest degree
df_resampled['Wind Direction'] = df_resampled['Wind Direction'].round(0)

# This gives details of the data frame
print(df.info())

# Display the first 30 rows of the resampled data
print(df_resampled.head(30))

# Save the cleaned and resampled data to a CSV file (optional)
df_resampled.to_csv('crescent_data_cleaned.csv')'''

import pandas as pd
from fpdf import FPDF

# File paths
nmb_file = '/Users/jayriihiluoma/Documents/python/scrapers/NMB Crescent data /2023.09.01-2024.08.31-Cresent-NMB/2023.09.01-2024.08.31-NMBAWOS.txt'
crescent_file = '/Users/jayriihiluoma/Documents/python/scrapers/NMB Crescent data /2023.09.01-2024.08.31-Cresent-NMB/2023.09.01-2024.08.31-CresentAWOS.txt'

# NMB dataset
nmb_headers = [
    'Date Time (UTC)', 'Wind Speed Avg (kt)', 'Wind Gust Avg (kt)', 'Wind Direction (Deg True)',
    'Pressure (hPa)', 'Temperature (C)', 'RH (%)', 'Dew Point (C)', 'Heat Index (C)',
    'Wet Bulb (C)', 'Precipitation (mm)'
]
df_nmb = pd.read_csv(nmb_file, sep="\t", header=None, names=nmb_headers, skiprows=1, low_memory=False)

# Ensure numeric columns
numeric_columns_nmb = ['Wind Speed Avg (kt)', 'Wind Gust Avg (kt)', 'Wind Direction (Deg True)']
for col in numeric_columns_nmb:
    df_nmb[col] = pd.to_numeric(df_nmb[col], errors='coerce')

# Drop rows with missing numeric data
df_nmb.dropna(subset=numeric_columns_nmb, inplace=True)

# Parse the datetime column
df_nmb['Date Time (UTC)'] = pd.to_datetime(df_nmb['Date Time (UTC)'], errors='coerce')
df_nmb.dropna(subset=['Date Time (UTC)'], inplace=True)
df_nmb.set_index('Date Time (UTC)', inplace=True)

# Resample to 5-minute intervals
df_nmb = df_nmb[['Wind Speed Avg (kt)', 'Wind Gust Avg (kt)', 'Wind Direction (Deg True)']]
df_nmb = df_nmb.resample('5min').mean()

# Crescent dataset
crescent_headers = [
    'Date Time (UTC)', 'Wind Speed Avg (kt)', 'Wind Gust Avg (kt)', 'Wind Direction (Deg True)'
]
df_crescent = pd.read_csv(crescent_file, sep="\t", header=None, names=crescent_headers, skiprows=1, low_memory=False)

# Ensure numeric columns
for col in ['Wind Speed Avg (kt)', 'Wind Gust Avg (kt)', 'Wind Direction (Deg True)']:
    df_crescent[col] = pd.to_numeric(df_crescent[col], errors='coerce')

# Drop rows with missing numeric data
df_crescent.dropna(subset=['Wind Speed Avg (kt)', 'Wind Gust Avg (kt)', 'Wind Direction (Deg True)'], inplace=True)

# Parse the datetime column
df_crescent['Date Time (UTC)'] = pd.to_datetime(df_crescent['Date Time (UTC)'], errors='coerce')
df_crescent.dropna(subset=['Date Time (UTC)'], inplace=True)
df_crescent.set_index('Date Time (UTC)', inplace=True)
df_crescent = df_crescent.resample('5min').mean()

# Combine datasets
combined_df = pd.merge(df_nmb, df_crescent, on='Date Time (UTC)', how='outer', suffixes=('_nmb', '_crescent'))

# Rename columns for simplicity
combined_df.columns = ['nmb WS', 'nmb Gust', 'nmb Dir', 'cres WS', 'cres Gust', 'cres Dir']

# Drop rows where Crescent data is missing
combined_df = combined_df.dropna(subset=['cres WS', 'cres Gust', 'cres Dir'])

# Round numeric columns
columns_to_round = ['nmb WS', 'nmb Gust', 'nmb Dir', 'cres WS', 'cres Gust', 'cres Dir']
combined_df[columns_to_round] = combined_df[columns_to_round].round({'nmb WS': 1, 'nmb Gust': 1, 
                                                                      'nmb Dir': 0, 'cres WS': 1, 
                                                                      'cres Gust': 1, 'cres Dir': 0})

# Save to CSV for local inspection
combined_df.to_csv('combined_data_cleaned.csv')

# Export to PDF
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Combined Dataset (NMB and Crescent)', 0, 1, 'C')

pdf = PDF()
pdf.add_page()
pdf.set_font('Arial', size=10)

# Add data to PDF
pdf.cell(0, 10, f"Date Time (UTC) | {', '.join(combined_df.columns)}", 0, 1)
for i, row in combined_df.iterrows():  # Export all rows
    pdf.cell(0, 10, f"{i} | {', '.join(row.astype(str).tolist())}", 0, 1)

pdf.output('combined_data_cleaned.pdf')




