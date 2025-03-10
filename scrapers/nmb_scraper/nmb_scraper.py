
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
import os #to allow temp files to be stoed in the local dir

# Define the base directory for the scraper
base_dir = "/Users/jayriihiluoma/Documents/python/scrapers/nmb_scraper"

# Create an 'images' folder inside the scraper folder
images_dir = os.path.join(base_dir, "images")
os.makedirs(images_dir, exist_ok=True)

# Define the timezone for Bermuda (you can use "America/Halifax" as it's the same)
timezone = pytz.timezone("America/Halifax")

# Get the current time in the Bermuda timezone
now_time_bda = datetime.now(timezone)
now_time = now_time_bda.strftime("%Y-%m-%d-%H%M")  # For filenames or identifiers
#print (("writing_NMB {}").format(now_time))


URLC = 'http://weather.bm/tools/graphics.asp?name=NMB%20GRAPH&user='
page = requests.get(URLC)
#print (type(page))
#if (type(page)) == requests.models.Response:
    #print ('page type ok')
#else:
    #print ('page type wrong')

soup = BeautifulSoup(page.content, 'html.parser')

# Find the image with id="Img_1"
image = soup.find('img', id="Img_1")  # Adjusted to select the most recent slide
if image:
    src = image.get('src')  # Get the source of the image
    url = f"http://weather.bm/{src}"
    url = url.replace(" ", "%20")  # Replace spaces with %20 for proper URL encoding

    filename = os.path.join(images_dir, "windc.png")
    response = requests.get(url)

    with open(filename, 'wb') as file:
        file.write(response.content)  # Save the image locally

    # Open and process the image
    im = Image.open(filename, mode='r')
    #im.show()
else:
    print("Image with id='Img_1' not found!")


# Setting the points for cropped image wind speed
left = 978
top = 75
right =1115
bottom = 100

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

# Output the OCR result
print(text_ws)

#print(text_ws)
p = re.compile(r'\d+\.\d+')  # Compile a pattern to capture float values
num_ws = [float(i) for i in p.findall(text_ws)]  # Convert strings to float
recent_ws = num_ws [0]
#print(recent_ws)



#Setting the points for cropped image max wind speed
left = 978
top = 265
right =1115
bottom = 298

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

# Output the OCR result
print(text_mws)

#print(text_mws)

p = re.compile(r'\d+\.\d+')  # Compile a pattern to capture float values
num_mws = [float(i) for i in p.findall(text_mws)]  # Convert strings to float
#print(num_mws)


recent_mws = num_mws[0]

#print(recent_mws)

# Setting the points for cropped image wind direction 
left = 978
top = 475
right =1140
bottom = 498
# Cropped image of above dimension 
# (It will not change orginal image) 
im1 = im.crop((left, top, right, bottom)) 
# Crop the image
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
# inverted_image.show()

# Read the inverted image with OpenCV
img = cv2.imread(bw_crop_inv_wdc_path)

# Perform OCR using pytesseract
text_wd = pytesseract.image_to_string(img)

# Output the OCR result
print(text_wd)


#returning only floating numbers from wd string
p = re.compile(r'\d+\.\d+')  # Compile a pattern to capture float values
num_wd = [float(i) for i in p.findall(text_wd)]  # Convert strings to float
#print (num_wd)

#slice for output
recent_wd = num_wd[0]

#print(recent_wd)

    #the 'a' says to append where as a 'w' would write (from scratch)
    #for textLine in text:
    #f.write(textLine) # write data line to the open file 
    # with closes file automatically on exiting block


'''with open('jdatap3.csv', 'a', newline='') as file:  
    writer = csv.writer(file)
    writer.writerow([now_time,recent_ws,recent_mws,recent_wd])
    #print ("finished_writing_pearl V3")'''

#adding date format that GSheets can read with date/time value


#this sets the time to BDA from UTC use timedelta -180 for daylight savings and -240 for no daylight savings
# Format the time for Google Sheets with proper timezone adjustment
now_time_gsheet = now_time_bda.strftime("%Y/%m/%d %H:%M")

#print("This is the time for gsheet recording", now_time_gsheet)	


#Gsheet APIs commented out for working on locally
scope = ['https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]

creds = ServiceAccountCredentials.from_json_keyfile_name(
    "/Users/jayriihiluoma/Documents/python/scrapers/nmb_scraper/creds.json", scope
)


client = gspread.authorize(creds)

spreadsheet = client.open("crescent_data")  # Open the spreadsheet by name
sheet = spreadsheet.worksheet("NMB_data")       # Access the worksheet named "NMB"

data = sheet.get_all_records(head=3)

data_row_add = [now_time_gsheet,recent_ws,recent_mws,recent_wd]
#print("Data to insert:", data_row_add)

sheet.insert_row(data_row_add,4)

'''# Creating url for windguru get API # initializing string 
str2hash = (("{}crescent_bermudacrescentstation*").format(now_time))
#print(("{}crescent_bermudacrescentstation*").format(now_time))
# encoding Salt using encode() 
# then sending to md5() 
result = hashlib.md5(str2hash.encode()) 
  
# printing the equivalent hexadecimal value. 
#print("The hexadecimal equivalent of hash is : ", end ="") 
#print(result.hexdigest())

#print(("windguru.cz/upload/api.php?uid=crescent_bermuda&salt={}&hash={}&wind_avg={}&wind_max={}&wind_direction={}").format(now_time,result.hexdigest(),recent_ws,recent_mws,recent_wd))

#send windguru pearl data via get
URL = ("http://www.windguru.cz/upload/api.php?uid=crescent_bermuda&salt={}&hash={}&wind_avg={}&wind_max={}&wind_direction={}").format(now_time,result.hexdigest(),recent_ws,recent_mws,recent_wd)
page = requests.get(URL)'''
