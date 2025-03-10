import gspread
from oauth2client.service_account import ServiceAccountCredentials

scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('/Users/jayriihiluoma/Documents/python/scrapers/crescent_scraper/creds.json', scope)
client = gspread.authorize(creds)

spreadsheet = client.open("crescent_data")
try:
    sheet = spreadsheet.worksheet('pred_cresc')
except gspread.exceptions.WorksheetNotFound:
    sheet = spreadsheet.add_worksheet(title='pred_cresc', rows='1000', cols='10')

sheet.append_row(["Test", "Row", "For", "Google", "Sheets"])
print("Test row written.")
