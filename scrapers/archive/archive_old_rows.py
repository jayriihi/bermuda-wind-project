import gspread
from oauth2client.service_account import ServiceAccountCredentials
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import List

# -------- Config --------
SPREADSHEET_NAME = "crescent_data"
TABS: List[str] = ["NMB_data", "pred_cresc", "Pearl"]  # order doesn't matter
HEADER_ROWS = 3                 # you keep headers in rows 1–3
DATE_COL = "Date/Time"         # exact header text in row 3
CUTOFF_YEAR = 2025             # keep rows >= this year
DRY_RUN = False                # True = no writes, just CSVs + console
CHUNK_ROWS = 1000              # chunk writes to avoid API size limits
# ------------------------

# Auth
scope = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
creds_path = Path(__file__).resolve().parents[1] / "crescent_scraper" / "creds.json"
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
    # guard against entirely empty sheets
    if df.empty:
        print("No data rows; nothing to do.")
        return

    # Parse Date/Time
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    # Archive < CUTOFF_YEAR
    to_archive = df[df[DATE_COL].dt.year < CUTOFF_YEAR]
    keep_df    = df[df[DATE_COL].dt.year >= CUTOFF_YEAR]

    print(f"Rows total: {len(df)} | archive: {len(to_archive)} | keep: {len(keep_df)}")

    # Save archive CSV (even in dry run—it’s safe and handy)
    out_csv = f"archive_{tab_name}_pre_{CUTOFF_YEAR}.csv"
    if len(to_archive) > 0:
        to_archive.to_csv(out_csv, index=False)
        print(f"📁 Wrote {len(to_archive)} rows to {out_csv}")
    else:
        print("📁 Nothing to archive for this tab.")

    if DRY_RUN:
        print("🧪 DRY RUN: skipping sheet writes.")
        return

    # Re-build values matrix: first 3 header rows + kept rows
    new_values = values[:HEADER_ROWS] + keep_df.astype(str).values.tolist()

    # Clear + chunked update (avoid 10MB payload issues)
    print(f"🧹 Clearing '{tab_name}' and writing {len(keep_df)} kept rows...")
    sh.clear()

    if not new_values:
        print("Nothing to write after clear (unexpected).")
        return

    # Write headers first (HEADER_ROWS rows)
    sh.update(f"A1:{gspread.utils.rowcol_to_a1(HEADER_ROWS, len(new_values[0]))}", new_values[:HEADER_ROWS])

    # Then write data in chunks starting after header rows
    start_row = HEADER_ROWS + 1
    data_matrix = new_values[HEADER_ROWS:]
    col_count = len(new_values[0])

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
