import gspread
from oauth2client.service_account import ServiceAccountCredentials
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import List

# -------- Config --------
SPREADSHEET_NAME = "crescent_data"
TABS: List[str] = ["NMB_data", "pred_cresc", "Pearl", "Sheet1"]  # order doesn't matter
HEADER_ROWS = 3                 # you keep headers in rows 1–3
DATE_COL = "Date/Time"         # exact header text in row 3
KEEP_YEAR = 2026  # keep rows from this year and newer; archive anything older
DRY_RUN = True                # True = no writes, just CSVs + console
CHUNK_ROWS = 1000              # chunk writes to avoid API size limits
# ------------------------

# Auth
scope = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
creds_path = Path(__file__).resolve().parents[1] / "scrapers" / "crescent_scraper" / "creds.json"
print("Using creds path:", creds_path.resolve())
if not creds_path.exists():
    raise FileNotFoundError(f"Missing Google service account credentials file: {creds_path.resolve()}")
creds = ServiceAccountCredentials.from_json_keyfile_name(str(creds_path), scope)
client = gspread.authorize(creds)

def chunked(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i+size]

def process_tab(ss, tab_name: str):
    print(f"\n=== Processing tab: {tab_name} ===")
    sh = ss.worksheet(tab_name)

    values = sh.get_all_values()  # list[list[str]]
    if len(values) < HEADER_ROWS:
        print("Tab has fewer than header rows; skipping.")
        return

    headers = values[HEADER_ROWS-1]  # row 3 (0-indexed)
    data_rows = values[HEADER_ROWS:]  # after headers

    if DATE_COL not in headers:
        print(f"⚠️ '{DATE_COL}' not found in header row; skipping this tab.")
        return

    # Frame the data for filtering
    df = pd.DataFrame(data_rows, columns=headers)
    if df.empty:
        print("No data rows; nothing to do.")
        return

    # Parse Date/Time
    raw_dt = df[DATE_COL].astype(str).str.strip()
    df[DATE_COL] = pd.to_datetime(raw_dt, errors="coerce")

    year_counts = df[DATE_COL].dt.year.value_counts(dropna=False).sort_index()
    print("Year counts:", year_counts.to_dict())
    print("Unparseable Date/Time rows (NaT):", int(df[DATE_COL].isna().sum()))

    # 👉 ADD THIS BLOCK HERE
    bad_dt = df[df[DATE_COL].isna()]
    if len(bad_dt) > 0:
        bad_csv = f"bad_datetime_{tab_name}.csv"
        bad_dt.to_csv(bad_csv, index=False)
        print(f"⚠️ Wrote {len(bad_dt)} rows with unparseable Date/Time to {bad_csv}")

    # Now filter
    year = df[DATE_COL].dt.year
    to_archive = df[year < KEEP_YEAR]
    keep_df    = df[(year >= KEEP_YEAR) | (df[DATE_COL].isna())]


    # Keep rows from KEEP_YEAR and newer; archive anything older
    to_archive = df[df[DATE_COL].dt.year < KEEP_YEAR]
    keep_df    = df[df[DATE_COL].dt.year >= KEEP_YEAR]


    print(f"Rows total: {len(df)} | archived: {len(to_archive)} | kept: {len(keep_df)}")

    # Save archive CSV (even in dry run—it’s safe and handy)
    out_csv = f"archive_{tab_name}_pre_{KEEP_YEAR}.csv"
    if len(to_archive) > 0:
        to_archive.reindex(columns=headers).to_csv(out_csv, index=False)
        print(f"📁 Wrote {len(to_archive)} rows to {out_csv}")
    else:
        print("Nothing to archive for this tab.")

    if DRY_RUN:
        print("🧪 DRY RUN: skipping sheet writes.")
        return

    # Re-build values matrix: first 3 header rows + kept rows
    new_values = values[:HEADER_ROWS] + keep_df.astype(str).values.tolist()

    # Clear data region only (preserves header rows and formatting), then chunked update.
    used_col_count = max((len(r) for r in values), default=len(headers))
    used_col_count = max(used_col_count, len(headers), 1)
    first_data_row = HEADER_ROWS + 1
    last_used_row = len(values)
    if last_used_row >= first_data_row:
        clear_range = f"A{first_data_row}:{gspread.utils.rowcol_to_a1(last_used_row, used_col_count)}"
        print(f"🧹 Clearing data values in '{tab_name}' range {clear_range} and writing {len(keep_df)} kept rows...")
        sh.batch_clear([clear_range])
    else:
        print(f"🧹 No existing data rows to clear in '{tab_name}'; writing {len(keep_df)} kept rows...")

    # Write kept data in chunks starting after header rows
    start_row = HEADER_ROWS + 1
    data_matrix = new_values[HEADER_ROWS:]
    col_count = len(headers)

    for idx, block in enumerate(chunked(data_matrix, CHUNK_ROWS), start=1):
        end_row = start_row + len(block) - 1
        rng = f"A{start_row}:{gspread.utils.rowcol_to_a1(end_row, col_count)}"
        sh.update(rng, block)
        print(f"   → wrote chunk {idx}: rows {start_row}-{end_row}")
        start_row = end_row + 1

    print(f"✅ Finished '{tab_name}'")

if __name__ == "__main__":
    ss = client.open(SPREADSHEET_NAME)
    print("Authenticated as:", creds.service_account_email)
    print("Spreadsheet:", ss.title)
    for tab in TABS:
        try:
            process_tab(ss, tab)
        except gspread.exceptions.WorksheetNotFound:
            print(f"⚠️ Tab '{tab}' not found; skipping.")
