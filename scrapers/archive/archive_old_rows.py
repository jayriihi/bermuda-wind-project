import gspread
from oauth2client.service_account import ServiceAccountCredentials
from pathlib import Path
from datetime import datetime
import pandas as pd

# Define scope
scope = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

# Point to your creds file (works whether you're local or moving to VM later)
creds_path = Path(__file__).resolve().parents[1] / "crescent_scraper" / "creds.json"

# Authenticate
creds = ServiceAccountCredentials.from_json_keyfile_name(str(creds_path), scope)
client = gspread.authorize(creds)

# Open spreadsheet
spreadsheet = client.open("crescent_data")

# Access tabs by name
crescent_sheet = spreadsheet.worksheet("Sheet1")
nmb_sheet = spreadsheet.worksheet("NMB_data")
pred_sheet = spreadsheet.worksheet("pred_cresc")  # etc.

# Test auth and sheet access only
if __name__ == "__main__":
    print("Authenticating and accessing spreadsheet...")
    print("Client email:", creds.service_account_email)
    print("Successfully accessed:", spreadsheet.title)
    print("Sheet1 first row:", crescent_sheet.row_values(1))

    values = crescent_sheet.get_all_values()
    data_rows = values[3:]  # skip headers
    df = pd.DataFrame(data_rows, columns=values[2])  # use row 3 as headers
    df['Date/Time'] = pd.to_datetime(df['Date/Time'], errors='coerce')
    to_archive = df[df['Date/Time'].dt.year < 2025]

    print(f"\n✅ Rows before 2025: {len(to_archive)}")
    print(f"📋 First archive row: {to_archive.head(1)}")

    # Save to CSV
    to_archive.to_csv("archive_Sheet1_pre_2025.csv", index=False)
    print("📁 Archived rows saved to 'archive_Sheet1_pre_2025.csv'")

    # Keep only post-2024 rows
    keep_df = df[df['Date/Time'].dt.year >= 2025]
    new_sheet_values = values[:3] + keep_df.astype(str).values.tolist()

    # Update sheet
    print(f"🧹 Replacing sheet data with {len(keep_df)} rows from 2025 and beyond...")
    crescent_sheet.clear()
    crescent_sheet.update(new_sheet_values)
    print("✅ Sheet1 archive complete and cleaned up.")


