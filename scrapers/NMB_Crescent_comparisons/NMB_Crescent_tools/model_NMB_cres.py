import os
import joblib
import numpy as np
from oauth2client.service_account import ServiceAccountCredentials
import gspread
import pandas as pd


# Function to predict Crescent wind speed
def predict_cres_ws(nmb_ws, nmb_dir, output_dir='bin_models/'):
    """
    Predict Crescent wind speed (cres WS) given NMB wind speed and direction.

    Parameters:
        nmb_ws (float): Wind speed at NMB.
        nmb_dir (float): Wind direction at NMB.
        output_dir (str): Directory where the models are stored.

    Returns:
        float: Predicted wind speed at Crescent.
    """
    bin_size = 5
    dir_bins = np.arange(0, 360 + bin_size, bin_size)
    dir_labels = [(dir_bins[i] + dir_bins[i + 1]) / 2 for i in range(len(dir_bins) - 1)]
    dir_bin = dir_labels[int(nmb_dir / bin_size)]

    model_path = os.path.join(output_dir, f'bin_{dir_bin}_model.pkl')
    if not os.path.exists(model_path):
        raise ValueError(f"No model found for direction bin {dir_bin}. Ensure the model is trained.")

    model = joblib.load(model_path)
    X = pd.DataFrame([[nmb_ws]], columns=['cres WS'])
    ws_diff_pred = model.predict(X)[0]
    cres_ws_pred = nmb_ws - ws_diff_pred
    cres_ws_pred = max(cres_ws_pred, 0)  # Ensure wind speed is not negative
    return cres_ws_pred


# Main execution
if __name__ == "__main__":
    # Define paths
    creds_path = '/Users/jayriihiluoma/Documents/python/scrapers/crescent_scraper/creds.json'  # Update with the actual path
    spreadsheet_name = 'crescent_data'  # Update with the spreadsheet name
    nmb_sheet_name = 'NMB_data'  # Update with the NMB sheet name
    pred_sheet_name = 'pred_cresc'
    output_dir = '/Users/jayriihiluoma/Documents/python/scrapers/NMB_Crescent_comparisons/NMB_Crescent_data/bin_models'  # Update with the correct path

    # Authenticate with Google Sheets
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
    client = gspread.authorize(creds)

    # Open the spreadsheet and access sheets
    spreadsheet = client.open(spreadsheet_name)
    nmb_sheet = spreadsheet.worksheet(nmb_sheet_name)
    try:
        pred_sheet = spreadsheet.worksheet(pred_sheet_name)
    except gspread.exceptions.WorksheetNotFound:
        pred_sheet = spreadsheet.add_worksheet(title=pred_sheet_name, rows='1000', cols='10')

    # Write header to pred_cresc if empty
    header = ["Date/Time", "Cresc Avg WS", "Cresc Max WS", "NMB Dir"]
    if len(pred_sheet.get_all_values()) == 0:
        pred_sheet.append_row(header)

    # Fetch the most recent row from the NMB sheet
    latest_row = nmb_sheet.row_values(4)  # Assuming the latest data is always in row 4

    # Extract values from the row (update indices based on column positions)
    date_time = latest_row[0]
    nmb_avg_ws = float(latest_row[1])
    nmb_max_ws = float(latest_row[2])
    nmb_dir = float(latest_row[3])

    # Predict Crescent wind speeds
    cres_avg_ws = round(predict_cres_ws(nmb_avg_ws, nmb_dir, output_dir=output_dir),1)
    cres_max_ws = round(predict_cres_ws(nmb_max_ws, nmb_dir, output_dir=output_dir),1)

    # Write predictions to the pred_cresc sheet
    pred_sheet.insert_row([date_time, cres_avg_ws, cres_max_ws, nmb_dir], 4)  # Insert in row 4
    print("Prediction data written to 'pred_cresc' sheet.")
