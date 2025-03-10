import json
import pandas as pd

# Loading the data from the sleep data files provided for comprehensive analysis
# File paths
files = [
    "2024-07-31_2024-11-08_1715014_sleepData.json",

]

# Initializing a DataFrame to collect all sleep data
all_sleep_data = pd.DataFrame()

# Parsing each file
for file in files:
    with open(file, 'r') as f:
        data = json.load(f)
        df = pd.json_normalize(data)
        all_sleep_data = pd.concat([all_sleep_data, df], ignore_index=True)

# Converting time in seconds to hours for sleep stages
all_sleep_data['deepSleepHours'] = all_sleep_data['deepSleepSeconds'] / 3600
all_sleep_data['lightSleepHours'] = all_sleep_data['lightSleepSeconds'] / 3600
all_sleep_data['remSleepHours'] = all_sleep_data['remSleepSeconds'] / 3600
all_sleep_data['awakeHours'] = all_sleep_data['awakeSleepSeconds'] / 3600

# Converting timestamp to a readable date format
all_sleep_data['sleepStart'] = pd.to_datetime(all_sleep_data['sleepStartTimestampGMT'])
all_sleep_data['sleepEnd'] = pd.to_datetime(all_sleep_data['sleepEndTimestampGMT'])
all_sleep_data['date'] = all_sleep_data['calendarDate']

# Summary statistics for sleep stages
summary_stats = all_sleep_data[['deepSleepHours', 'lightSleepHours', 'remSleepHours', 'awakeHours']].describe()

# Aggregating average sleep duration by date for each stage
average_sleep_data = all_sleep_data.groupby('date')[['deepSleepHours', 'lightSleepHours', 'remSleepHours', 'awakeHours']].mean()

# Displaying processed data to the user
print("Comprehensive Sleep Analysis")
print(average_sleep_data)

# Print to csv
average_sleep_data.to_csv("comprehensive_sleep_analysis.csv", index=False)

average_rem_sleep_hours = all_sleep_data['remSleepHours'].mean()
print("Average REM Sleep Hours:", average_rem_sleep_hours)


summary_stats

