# This version installed on to VM 21 January 2025
from datetime import datetime,timedelta #for setting times
import pytz  # Import pytz for timezone handling
import csv #to export to csv file
import requests #to get web pages
from PIL import Image  #to manipulate image
import PIL.ImageOps #to manipulate image
import cv2 #to manipulate image
import pytesseract #to ocr read from image
import re #to use filtering numbers from string
from bs4 import BeautifulSoup #for parsing website
import time
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import hashlib #for creating md5 hash
import os # os to allow for dir/folder management

# Define the base directory for the scraper
# Base dir for local
base_dir = "/Users/jayriihiluoma/Documents/python/scrapers/crescent_scraper"
# Base dir for VM
#base_dir = "/home/jayriihi/scrapers/crescent_scraper"

# Create an 'images' folder inside the scraper folder
images_dir = os.path.join(base_dir, "images")
os.makedirs(images_dir, exist_ok=True)

# Define the timezone for Bermuda (you can use "America/Halifax" as it's the same)
timezone = pytz.timezone("America/Halifax")

# Get the current time in the Bermuda timezone
now_time_bda = datetime.now(timezone)
now_time = now_time_bda.strftime("%Y-%m-%d-%H%M")  # For filenames or identifiers
#print (("writing_crescent_v1 {}").format(now_time))


URLC = 'http://weather.bm/tools/graphics.asp?name=CRESCENT%20GRAPH&user='
page = requests.get(URLC)
#print (type(page))
#if (type(page)) == requests.models.Response:
    #print ('page type ok')
#else:
    #print ('page type wrong')

soup = BeautifulSoup(page.content, 'html.parser')


images = soup.find(id="image")
src = images.get('src')
    #print(src)
url = ('http://weather.bm/{}').format(src)
url1 = url.replace(" ", "%20")

filename = os.path.join(images_dir, "windc.png")
#filename = 'windv3 {}.png'.format(now_time)
r = requests.get(url1)
open(filename, 'wb').write(r.content)

im = Image.open(filename, mode='r')
#im.show()

# Setting the points for cropped image wind speed
left = 980
top = 82
right =1140
bottom = 112

# Cropped image of above dimension 
# (It will not change orginal image) 
im1 = im.crop((left, top, right, bottom)) 
# Save the cropped image in the nmb_scraper folder
crop_wspc_path = os.path.join(images_dir, "crop_wspc.png")
im1.save(crop_wspc_path)

# Open the cropped image
cropwspc = Image.open(crop_wspc_path)
# Uncomment if you want to display the image
# cropwspc.show()

# Convert the cropped image to grayscale
bw_crop_wspc_path = os.path.join(images_dir, "bw_crop_wspc.png")
image_file = cropwspc.convert('L')  # convert image to black and white
image_file.save(bw_crop_wspc_path)

# Invert the grayscale image to create a black-and-white inverted image
bw_crop_inv_wspc_path = os.path.join(images_dir, "bw_crop_inv_wspc.png")
image = Image.open(bw_crop_wspc_path)
inverted_image = PIL.ImageOps.invert(image)
inverted_image.save(bw_crop_inv_wspc_path)

# Uncomment if you want to display the inverted image
# inverted_image.show()

# Read the inverted image with OpenCV
img = cv2.imread(bw_crop_inv_wspc_path)

# Perform OCR using pytesseract
text_ws = pytesseract.image_to_string(img)

p = re.compile(r'\d+\.\d+')  # Compile a pattern to capture float values
num_ws = [round(float(i), 1) for i in p.findall(text_ws)]  # Convert strings to float and round to 1 decimal
recent_ws = num_ws[0]  # Assign the first value to recent_ws
#print(recent_ws)




#Setting the points for cropped image max wind speed
left = 980
top = 315
right =1140
bottom = 338

# Cropped image of above dimension 
# (It will not change orginal image) 
im1 = im.crop((left, top, right, bottom)) 
# Save the cropped image in the nmb_scraper folder
crop_mwspc_path = os.path.join(images_dir, "crop_mwspc.png")
im1.save(crop_mwspc_path)

# Open the cropped image
cropmwspc = Image.open(crop_mwspc_path)
# Uncomment if you want to display the image
# cropmwspc.show()

# Convert the cropped image to grayscale
bw_crop_mwspc_path = os.path.join(images_dir, "bw_crop_mwspc.png")
image_file = cropmwspc.convert('L')  # convert image to black and white
image_file.save(bw_crop_mwspc_path)

# Invert the grayscale image to create a black-and-white inverted image
bw_crop_inv_mwspc_path = os.path.join(images_dir, "bw_crop_inv_mwspc.png")
image = Image.open(bw_crop_mwspc_path)
inverted_image = PIL.ImageOps.invert(image)
inverted_image.save(bw_crop_inv_mwspc_path)

# Uncomment if you want to display the inverted image
# inverted_image.show()

# Read the inverted image with OpenCV
img = cv2.imread(bw_crop_inv_mwspc_path)

# Perform OCR using pytesseract
text_mws = pytesseract.image_to_string(img)

# Compile a pattern to capture float values
p = re.compile(r'\d+\.\d+')

# Convert strings to float and round to 1 decimal
num_mws = [round(float(i), 1) for i in p.findall(text_mws)]
#print(num_mws)

# Assign the first value to recent_mws
recent_mws = num_mws[0]
#print(recent_mws)

#print(recent_mws)

# Setting the points for cropped image wind direction 
left = 980
top = 535
right =1140
bottom = 565

# Cropped image of above dimension 
# (It will not change orginal image) 
im1 = im.crop((left, top, right, bottom))
# Save the cropped image in the nmb_scraper folder
crop_wdc_path = os.path.join(images_dir, "crop_wdc.png")
im1.save(crop_wdc_path)

# Open the cropped image
cropwdc = Image.open(crop_wdc_path)
# Uncomment if you want to display the image
# cropwdc.show()

# Convert the cropped image to grayscale
bw_crop_wdc_path = os.path.join(images_dir, "bw_crop_wdc.png")
image_file = cropwdc.convert('L')  # convert image to black and white
image_file.save(bw_crop_wdc_path)

# Invert the grayscale image to create a black-and-white inverted image
bw_crop_inv_wdc_path = os.path.join(images_dir, "bw_crop_inv_wdc.png")
image = Image.open(bw_crop_wdc_path)
inverted_image = PIL.ImageOps.invert(image)
inverted_image.save(bw_crop_inv_wdc_path)

# Uncomment if you want to display the inverted image
#inverted_image.show()

# Read the inverted image with OpenCV
img = cv2.imread(bw_crop_inv_wdc_path)

# Perform OCR using pytesseract
text_wd = pytesseract.image_to_string(img)

# Compile a pattern to capture float values
p = re.compile(r'\d+\.\d+')

# Convert strings to float and round to the nearest whole number
num_wd = [round(float(i)) for i in p.findall(text_wd)]
#print(num_wd)
# Assign the first value to recent_wd
recent_wd = num_wd[0]
#print(recent_wd)

#adding date format that GSheets can read with date/time value
#this sets the time to BDA from UTC use timedelta -180 for daylight savings and -240 for no daylight savings
# Format the time for Google Sheets with proper timezone adjustment
now_time_gsheet = now_time_bda.strftime("%Y/%m/%d %H:%M")

#print("This is the time for gsheet recording", now_time_gsheet)	

# Values fetched from Crescent scraping process
print(recent_ws, recent_mws, recent_wd)

# Gsheet APIs
scope = ['https://www.googleapis.com/auth/spreadsheets', 
         "https://www.googleapis.com/auth/drive.file", 
         "https://www.googleapis.com/auth/drive"]

# Define the absolute path to creds.json
# Creds_path for local
creds_path = "/Users/jayriihiluoma/Documents/python/scrapers/crescent_scraper/creds.json"

# CredsPath for VM
#creds_path = "/home/jayriihi/scrapers/crescent_scraper/creds.json"

# Authenticate with Google Sheets
creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
client = gspread.authorize(creds)

# Access the Crescent data sheet
sheet = client.open("crescent_data").sheet1

# Fetch the latest three rows from Crescent sheet (rows 4, 5, and 6)
latest_rows = [sheet.row_values(i) for i in range(4, 7)]

# Ensure all fetched rows have valid data before proceeding
valid_rows = [row for row in latest_rows if len(row) >= 4 and all(row[1:])]

# Simulate using pred_cresc for testing (Set to True to force pred_cresc)
force_pred_cresc = True  # Change to True to force the script to act as if Crescent is offline

# Fetch the latest three rows from Crescent sheet (rows 4, 5, and 6)
latest_rows = [sheet.row_values(i) for i in range(4, 7)]

# Ensure all fetched rows have valid data before proceeding
valid_rows = [row for row in latest_rows if len(row) >= 4 and all(row[1:])]

# Function to check if all three rows match recent_ws, recent_mws, recent_wd
def is_crescent_offline(rows, recent_ws, recent_mws, recent_wd):
    return all(
        len(row) >= 4 and
        float(row[1]) == recent_ws and
        float(row[2]) == recent_mws and
        float(row[3]) == recent_wd
        for row in rows
    )

# Check if Crescent data is offline or if we are forcing pred_cresc
crescent_is_offline = is_crescent_offline(valid_rows, recent_ws, recent_mws, recent_wd) or force_pred_cresc

if crescent_is_offline:
    print("Crescent data appears to be offline OR force_pred_cresc is enabled. Fetching pred_cresc data.")

    # Access pred_cres sheet and fetch the latest row
    pred_sheet = client.open("crescent_data").worksheet("pred_cresc")
    latest_pred_row = pred_sheet.row_values(4)

    # Ensure pred_cresc row contains enough data before using it
    if len(latest_pred_row) < 4 or not all(latest_pred_row[1:]):
        print("⚠️ Error: pred_cresc row 4 is missing or incomplete. Skipping Windguru update.")
        pred_ws, pred_mws, pred_wd = None, None, None
    else:
        # Use pred_cresc data for Windguru API ONLY
        pred_ws = float(latest_pred_row[1])
        pred_mws = float(latest_pred_row[2])
        pred_wd = float(latest_pred_row[3])

        print(f"Using pred_cresc data for Windguru: {pred_ws}, {pred_mws}, {pred_wd}")

    # 🚨 Ensure Windguru receives pred_cresc values when Crescent is offline
    windguru_ws = pred_ws if pred_ws is not None else recent_ws
    windguru_mws = pred_mws if pred_mws is not None else recent_mws
    windguru_wd = pred_wd if pred_wd is not None else recent_wd

    # 🚨 STILL WRITE CRESCENT DATA TO SHEET1 (duplicate entry) for views.py outage detection
    data_row_add = [now_time_gsheet, recent_ws, recent_mws, recent_wd]
    sheet.insert_row(data_row_add, 4)  # ✅ Ensures repeated data is written for `views.py` to detect
    print("Offline Crescent data written to Sheet1 for redundancy.")
    
else:
    print("Crescent data is online. Storing in Crescent sheet.")

    # **Only insert real Crescent data into Google Sheets**
    data_row_add = [now_time_gsheet, recent_ws, recent_mws, recent_wd]
    sheet.insert_row(data_row_add, 4)  # ✅ Writes only once when Crescent is online.

    # 🚨 Ensure Windguru receives Crescent values when online
    windguru_ws = recent_ws
    windguru_mws = recent_mws
    windguru_wd = recent_wd

# 🚨 Print the final values being sent to Windguru
print(f"Data being sent to Windguru (Crescent or pred_cresc): Avg Wind Speed: {windguru_ws}, Max Wind Speed: {windguru_mws}, Wind Direction: {windguru_wd}")


'''# Function to check if all three rows match recent_ws, recent_mws, recent_wd
def is_crescent_offline(rows, recent_ws, recent_mws, recent_wd):
    return all(
        len(row) >= 4 and
        float(row[1]) == recent_ws and
        float(row[2]) == recent_mws and
        float(row[3]) == recent_wd
        for row in rows
    )

crescent_is_offline = is_crescent_offline(valid_rows, recent_ws, recent_mws, recent_wd)

if crescent_is_offline:
    print("Crescent data appears to be offline. Fetching pred_cresc data.")

    # Access pred_cres sheet and fetch the latest row
    pred_sheet = client.open("crescent_data").worksheet("pred_cresc")
    latest_pred_row = pred_sheet.row_values(4)

    # Use pred_cresc data for Windguru API ONLY
    pred_ws = float(latest_pred_row[1])
    pred_mws = float(latest_pred_row[2])
    pred_wd = float(latest_pred_row[3])

    print(f"Using pred_cresc data for Windguru: {pred_ws}, {pred_mws}, {pred_wd}")

    # 🚨 Ensure Windguru receives pred_cresc values when Crescent is offline
    windguru_ws = pred_ws
    windguru_mws = pred_mws
    windguru_wd = pred_wd

    # 🚨 STILL WRITE CRESCENT DATA TO SHEET1 (duplicate entry) for views.py outage detection
    data_row_add = [now_time_gsheet, recent_ws, recent_mws, recent_wd]
    sheet.insert_row(data_row_add, 4)  # ✅ Ensures repeated data is written for `views.py` to detect
    print("Offline Crescent data written to Sheet1 for redundancy.")
    
else:
    print("Crescent data is online. Storing in Crescent sheet.")

    # **Only insert real Crescent data into Google Sheets**
    data_row_add = [now_time_gsheet, recent_ws, recent_mws, recent_wd]
    sheet.insert_row(data_row_add, 4)  # ✅ Writes only once when Crescent is online.

    # 🚨 Ensure Windguru receives Crescent values when online
    windguru_ws = recent_ws
    windguru_mws = recent_mws
    windguru_wd = recent_wd

# 🚨 Print the final values being sent to Windguru
print(f"Data being sent to Windguru (Crescent or pred_cresc): Avg Wind Speed: {windguru_ws}, Max Wind Speed: {windguru_mws}, Wind Direction: {windguru_wd}")
'''


# Creating URL for Windguru API
str2hash = f"{now_time}crescent_bermudacrescentstation*"
result = hashlib.md5(str2hash.encode())
hash_value = result.hexdigest()

# Send data to Windguru
URL = (
    f"http://www.windguru.cz/upload/api.php?"
    f"uid=crescent_bermuda&salt={now_time}&hash={hash_value}&"
    f"wind_avg={windguru_ws}&wind_max={windguru_mws}&wind_direction={windguru_wd}"
)

try:
    response = requests.get(URL)
    print(f"Windguru API Response: {response.status_code}")
except Exception as e:
    print(f"Error occurred while sending data to Windguru: {e}")


