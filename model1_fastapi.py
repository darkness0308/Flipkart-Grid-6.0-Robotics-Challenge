import cv2
import torch
from ultralytics import YOLO
import easyocr
import re
import threading
import time
import numpy as np
import pandas as pd
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect,BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import sqlite3
import time
import cv2
import base64
import html
from starlette.websockets import WebSocketState
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import torch
from ultralytics import YOLO
import easyocr
import os
import time
from concurrent.futures import ThreadPoolExecutor
import numpy
import re
import threading
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from datetime import datetime

app = FastAPI()
# Global lists to store detected texts for each label
global valid_result
valid_result=None
brand_name_list = []
date_list = []
due_list = []
flavour_list = []
mrp_list = []
net_list = []
product_name_list = []
# Current date
current_date = datetime.now()
# Dictionary to convert month abbreviations to numbers
month_map = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
# Dictionary to correct incorrect month abbreviations
incorrect_month_map = {'JAN': 'JAN', 'JAM': 'JAN', 'JANU': 'JAN', 'JAU': 'JAN', 'FEB': 'FEB', 'FEE': 'FEB', 'FE8': 'FEB', 'FEP': 'FEB', 'MAR': 'MAR', 'MAA': 'MAR', 'MARCH': 'MAR', 'APR': 'APR', 'FIPR': 'APR', 'FipR': 'APR', 'APRL': 'APR', 'FFR':'APR','FPR':'APR','APRIL': 'APR', 'MAY': 'MAY','MAT':'MAY', 'MAN': 'MAY','MFM':'MAY','MIFN':'MAY','MFN':'MAY','MFN': 'MAY', 'MIFM': 'MAY', 'MIFN': 'MAY', 'JUN': 'JUN', 'JUNE': 'JUN', 'JWN': 'JUN', 'JUL': 'JUL', 'JULY': 'JUL', 'JLI': 'JUL', 'JLY': 'JUL', 'AUG': 'AUG', 'AU6': 'AUG', 'AUCT': 'AUG', 'AUGUST': 'AUG', 'AU8': 'AUG', 'AUC': 'AUG', 'AUQ': 'AUG', 'AU9': 'AUG', 'SEP': 'SEP', 'SEPT': 'SEP', 'SEPTEMBER': 'SEP', '5EP': 'SEP', '5EPT': 'SEP', 'SFP': 'SEP', 'SFP7': 'SEP', 'SPT': 'SEP', 'S9P': 'SEP', 'OCT': 'OCT', 'OCTOBER': 'OCT', '0CT': 'OCT', '0C7': 'OCT', 'OCt': 'OCT', 'OCTO': 'OCT', 'OQT': 'OCT', 'OC0': 'OCT', 'NOV': 'NOV', 'NOVEMBER': 'NOV', 'NQV': 'NOV', 'MOV': 'NOV', 'NOVEM': 'NOV', 'NOY': 'NOV', 'NOVB': 'NOV', 'NOVF': 'NOV', 'DEC': 'DEC', 'DECEMBER': 'DEC', 'DE0': 'DEC', 'D6C': 'DEC', 'OEC': 'DEC', 'DECEM': 'DEC', 'DECMB': 'DEC', 'DLC': 'DEC', 'D8C': 'DEC', 'DEC2': 'DEC','Aut': 'AUG', 'Aui': 'AUG', 'Auf': 'AUG'}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   # Initialize the IP webcam
ip_webcam_url = "http://10.151.8.128.8080/video" 
cap = cv2.VideoCapture(ip_webcam_url)
if not cap.isOpened():
    print("Error: Could not open IP webcam.")
    exit()
    # Function to rotate the image by a given angle

    # Function to rotate the image by a given angle

reader = easyocr.Reader(['en'])
MONTH_CORRECTIONS = {'JAN': 'JAN', 'JAM': 'JAN', 'JANU': 'JAN', 'JAU': 'JAN', 'FEB': 'FEB', 'FEE': 'FEB', 'FE8': 'FEB', 'FEP': 'FEB', 'MAR': 'MAR', 'MAA': 'MAR', 'MARCH': 'MAR', 'APR': 'APR', 'FIPR': 'APR', 'FipR': 'APR', 'APRL': 'APR', 'FFR':'APR','FPR':'APR','APRIL': 'APR', 'MAY': 'MAY','MAT':'MAY', 'MAN': 'MAY','MFM':'MAY','MIFN':'MAY','MFN':'MAY','MFN': 'MAY', 'MIFM': 'MAY', 'MIFN': 'MAY', 'JUN': 'JUN', 'JUNE': 'JUN', 'JWN': 'JUN', 'JUL': 'JUL', 'JULY': 'JUL', 'JLI': 'JUL', 'JLY': 'JUL', 'AUG': 'AUG', 'AU6': 'AUG', 'AUCT': 'AUG', 'AUGUST': 'AUG', 'AU8': 'AUG', 'AUC': 'AUG', 'AUQ': 'AUG', 'AU9': 'AUG', 'SEP': 'SEP', 'SEPT': 'SEP', 'SEPTEMBER': 'SEP', '5EP': 'SEP', '5EPT': 'SEP', 'SFP': 'SEP', 'SFP7': 'SEP', 'SPT': 'SEP', 'S9P': 'SEP', 'OCT': 'OCT', 'OCTOBER': 'OCT', '0CT': 'OCT', '0C7': 'OCT', 'OCt': 'OCT', 'OCTO': 'OCT', 'OQT': 'OCT', 'OC0': 'OCT', 'NOV': 'NOV', 'NOVEMBER': 'NOV', 'NQV': 'NOV', 'MOV': 'NOV', 'NOVEM': 'NOV', 'NOY': 'NOV', 'NOVB': 'NOV', 'NOVF': 'NOV', 'DEC': 'DEC', 'DECEMBER': 'DEC', 'DE0': 'DEC', 'D6C': 'DEC', 'OEC': 'DEC', 'DECEM': 'DEC', 'DECMB': 'DEC', 'DLC': 'DEC', 'D8C': 'DEC', 'DEC2': 'DEC'}

# Load the dataset
df = pd.read_csv('comdat.csv')

def correct_month_abbreviation(date_str):
    try:
        for incorrect, correct in incorrect_month_map.items():
            if incorrect in date_str:
                return date_str.replace(incorrect, correct)
        return date_str
    except Exception as e:
        print(f"Error correcting month abbreviation for {date_str}: {e}")
        return date_str


def check_brand_product_net_mrp(brand_name_list, product_name_list, net_list, mrp_list):
    # Variables to store the matched values
    matched_product_name = None
    matched_brand_name = None
    matched_net = None
    matched_mrp = None

    # Step 1: Check for product_name match first
    for product in product_name_list:
        product_matches = df[df['product_name'].str.lower() == product.lower()]  # Case-insensitive match
        if not product_matches.empty:
            matched_product_name = product
            matched_row = product_matches.iloc[0]  # Take the first matching row
            break

    # Step 2: Wait for brand name to match or use the corresponding brand name of the matched product name
    if matched_product_name:
        for brand in brand_name_list:
            if brand.lower() == matched_row['brand_name'].lower():
                matched_brand_name = brand
                break

        # If no brand name matched, take the brand name from the matched product
        if not matched_brand_name:
            matched_brand_name = matched_row['brand_name']

    # Step 3: Check for net and mrp matches within the matched row (if available)
    if matched_product_name:
        # Check Net Weight
        for net in net_list:
            if str(net).lower() == str(matched_row['net']).lower():
                matched_net = net
                break

        # If no net weight matched, take the one from the dataset
        if not matched_net:
            matched_net = matched_row['net']

        # Check MRP
        for mrp in mrp_list:
            if str(mrp).lower() == str(matched_row['mrp']).lower():
                matched_mrp = mrp
                break

        # If no MRP matched, take the one from the dataset
        if not matched_mrp:
            matched_mrp = matched_row['mrp']

    return matched_brand_name, matched_product_name, matched_net, matched_mrp

def correct_month_in_text(text):
    """Correct OCR misinterpretations of months."""
    corrected_text = text
    for wrong, correct in MONTH_CORRECTIONS.items():
        corrected_text = re.sub(rf'\b{wrong}\b', correct, corrected_text)
        return corrected_text

def clean_date_element(date_str):
    """Remove unwanted characters except for letters, numbers, spaces, and /, -, . from the date string."""
    # Escape the dash (-) to avoid regex range issues
    cleaned_date = re.sub(r"[^a-zA-Z0-9\s/\-\.]", "", date_str)
    return cleaned_date.strip()


def clean_date_list(dates_list):
    """Clean the date list by removing unwanted characters from each date."""
    return [clean_date_element(date) for date in dates_list]

def extract_dates(text):
    """Extract valid date patterns (DD/MM/YYYY, DD/MM/YY, MMM YYYY, MMM-YYYY) and correct month misinterpretations."""
    corrected_text = correct_month_in_text(' '.join(text))
    
    # Date patterns: DD/MM/YYYY, DD/MM/YY, MMM YYYY, MMM-YYYY
    date_pattern = re.compile(
        r'(\b[A-Z]{3}[-\s]?\d{4}\b)|'  # Matches MMM YYYY or MMM-YYYY
        r'(\d{2}/\d{2}/\d{4})|'         # Matches DD/MM/YYYY
        r'(\d{2}/\d{2}/\d{2})'          # Matches DD/MM/YY
    )
    
    # Find all matching date patterns
    matches = date_pattern.findall(corrected_text)
    
    # Flatten the matches into a list and remove empty strings
    cleaned_dates = [match for group in matches for match in group if match]
    
    return cleaned_dates

def clean_mrp_text(text):
    """ Clean and retain only valid MRP values (whole numbers or numbers with two decimal places). """
    # Remove any characters that are not digits or decimal points
    text = re.sub(r'[^0-9.]', '', text).strip()
    
    # Match valid rounded values
    if re.match(r'^\d+(\.00)?$', text):  # Matches whole numbers or numbers with ".00"
        return text
    else:
        return None

def clean_text(text):
    """ Remove unwanted punctuation and special characters from the text. """
    return re.sub(r'[^a-zA-Z0-9\s]', '', text).strip()

# Function to convert a date string into a datetime object
def parse_date(date_str):
    try:
        # For 'MM/DD/YY' format
        if '/' in date_str:
            return datetime.strptime(date_str, '%m/%d/%y')
        # For 'MMM-YYYY' format
        elif '-' in date_str and len(date_str) == 8:
            month_abbr = date_str[:3].upper()
            year = int(date_str[-4:])
            month = month_map.get(month_abbr, None)
            if month:
                return datetime(year, month, 1)
            else:
                raise ValueError(f"Invalid month abbreviation: {month_abbr}")
    except ValueError as ve:
        print(f"ValueError: {ve} for date: {date_str}")
        return None
    except Exception as e:
        print(f"Unexpected error parsing date {date_str}: {e}")
        return None


def remove_redundancy_and_print():
    print("_____________________________________________",date_list)
    """ Remove redundancy and apply filtering for specific fields. """
    unique_brand_names = list(set([clean_text(x) for x in brand_name_list]))
    #cleaned_dates_list = clean_date_list(date_list)
    # Regular expression patterns for two date formats: 'MMM-YYYY' and 'MM/DD/YY'
    pattern1 = re.compile(r'^[A-Za-z]{3}-\d{4}$')  # Example: JUL-2025
    pattern2 = re.compile(r'^\d{2}/\d{2}/\d{2}$')  # Example: 07/02/25
    # Filter valid dates and remove duplicates
    cleaned_dates_list = list(set([item for item in date_list if pattern1.match(item) or pattern2.match(item)]))
    # Sort the cleaned list (optional)
    cleaned_dates_list.sort()

    # unique_dates = [date for date in cleaned_dates_list if parse_date(date) and parse_date(date) > current_date]
    unique_dates = []
    # Step 1: Correct the month abbreviations in the cleaned_dates_list
    corrected_dates_list = []
    for date in cleaned_dates_list:
        try:
            corrected_date = correct_month_abbreviation(date)
            corrected_dates_list.append(corrected_date)
        except Exception as e:
            print(f"Error processing date {date}: {e}")

    print("Corrected Dates List:", corrected_dates_list)

    for date in corrected_dates_list:
        try:
            parsed_date = parse_date(date)
            if parsed_date and parsed_date > current_date:
                unique_dates.append(date)
        except Exception as e:
            print(f"Error filtering date {date}: {e}")


    if unique_dates:
        valid_result="not expired"
        print("Not yet Expired")
    else:
        valid_result="expired"
        print("Expired")

        
    #unique_dates = list(set([date for date in cleaned_dates_list if extract_dates([date])]))
    unique_due_dates = list(set(due_list))
    unique_flavours = list(set(flavour_list))
    unique_mrp_values = list(filter(None, set([clean_mrp_text(x) for x in mrp_list])))
    unique_net_weights = list(set(net_list))
    unique_product_names = list(set([clean_text(x) for x in product_name_list]))


    # Print the final lists without redundancy
    print("Brand Names:", unique_brand_names)
    print("Dates:", unique_dates)
    print("Due Dates:", unique_due_dates)
    print("Flavours:", unique_flavours)
    print("MRP Values:", unique_mrp_values)
    print("Net Weights:", unique_net_weights)
    print("Product Names:", unique_product_names)

    print("--------------comparison-------------")
    # Perform brand, product, net, and MRP matching
    matched_brand, matched_product, matched_net, matched_mrp = check_brand_product_net_mrp(
        unique_brand_names, unique_product_names, unique_net_weights, unique_mrp_values
    )

    # Print matched values
    print(f"\nMatched Brand Name: {matched_brand}")
    print(f"Matched Product Name: {matched_product}")
    print(f"Matched Net Weight: {matched_net}")
    print(f"Matched MRP: {matched_mrp}")
    print(f"valid result {valid_result}")
    print("-----------------------------")
    if str(unique_dates)!='[]':
        insert_data(matched_product, matched_brand, str(matched_net), str(matched_mrp), str(unique_dates),str(valid_result))
    else:
         pass
    #cleaning the list for next iterations
    brand_name_list.clear()
    date_list.clear()
    due_list.clear()
    flavour_list.clear()
    mrp_list.clear()
    net_list.clear()
    valid_result=None
    product_name_list.clear()
    unique_brand_names.clear()
    unique_dates.clear()
    unique_due_dates.clear()
    unique_flavours.clear()
    unique_mrp_values.clear()
    unique_net_weights.clear()
    unique_product_names.clear()

def get_dominant_color(image, k=4):
    """Extract the dominant color of the image using K-means clustering."""
    # Reshape image to be a list of pixels
    pixels = image.reshape((-1, 3))

    # If the image has fewer pixels than clusters, reduce k to the number of unique colors
    if len(pixels) < k:
        k = len(pixels)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=k, n_init='auto')
    kmeans.fit(pixels)

    # Get the most common label
    labels, counts = np.unique(kmeans.labels_, return_counts=True)

    # Find the most frequent label (cluster)
    dominant_label = labels[counts.argmax()]

    # Ensure the dominant_label index does not exceed the number of cluster centers
    if dominant_label < len(kmeans.cluster_centers_):
        dominant_color = kmeans.cluster_centers_[dominant_label]
    else:
        # In case of mismatch, return the average color as a fallback
        dominant_color = pixels.mean(axis=0)

    return tuple(map(int, dominant_color))  # Return as (B, G, R) integer tuple


def insert_data(product_name, brand_name, net, mrp, expiry_date,valid_result):
    # Connect to the SQLite database
    conn = sqlite3.connect('example.db')

    # Check if any of the fields are missing
    if not product_name or not brand_name or not net or not mrp or not expiry_date or not valid_result:
        print("Error: All fields must be provided and not empty.")
        return

    with conn:
        try:
            # Insert data into the products table
            conn.execute('''
                INSERT INTO products (product_name, brand_name, net, mrp, expiry_date,valid)
                VALUES (?, ?, ?, ?, ?,?)
            ''', (product_name, brand_name, net, mrp, expiry_date,valid_result))
            
            print("Data inserted:", {
                'product_name': product_name,
                'brand_name': brand_name,
                'net': net,
                'mrp': mrp,
                'expiry_date': expiry_date
            })
        except sqlite3.Error as e:
            print(f"An error occurred while inserting data: {e}")
        except sqlite3.IntegrityError:
         # Handle the case where the product already exists
            print(f'Product {product_name} already exists in the database.')
    conn.close()
# WebSocket endpoint to start capturing video frames
def extract_best_text_4_directions(image):
    results = reader.readtext(image, detail=1, rotation_info=[0, 90, 180, 270])
    best_text = None
    highest_confidence = 0
    for (bbox, text, confidence) in results:
        if confidence > highest_confidence:
            best_text = text
            highest_confidence = confidence
    return best_text

def extract_best_text_360(image):
    results = reader.readtext(image, detail=1, rotation_info=[0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330])
    best_text = None
    highest_confidence = 0
    for (bbox, text, confidence) in results:
        if confidence > highest_confidence:
            best_text = text
            highest_confidence = confidence
    return best_text

# Function to filter specific patterns using regular expressions
def extract_dates(text):

    """Extract valid date patterns (DD/MM/YYYY, MMM YYYY) and correct month misinterpretations."""
    # Correct any common month OCR errors in the text
    corrected_text = correct_month_in_text(' '.join(text))
    
    # Date patterns: DD/MM/YYYY or Month YYYY
    date_pattern = re.compile(r'(\b[A-Z]{3}[-\s]?\d{4}\b)|'  # Matches MMM YYYY or MMM-YYYY
                            r'(\d{2}[./-]\d{2}[./-]\d{4})|'  # Matches DD/MM/YYYY
                            r'(\d{3}[./-]\d{2}[./-]\d{2})')
    
    matches = date_pattern.findall(corrected_text)

    # Flatten the matches into a list and remove empty strings
    return [match for group in matches for match in group if match]
custom_labels = {
    0: 'Bottle',
    1: 'Brand_name',
    2: 'Date',
    3: 'Due',
    4: 'Flavour',
    5: 'MRP',
    6: 'Net',
    7: 'Pack',
    8: 'Product_name',
    9: 'Flavour'
}

async def capture_video(websocket: WebSocket):
    
    await websocket.accept()
    global processed_frame
    model = YOLO('best.pt').to(device)
 
    # Custom labels for the detection (update as per your dataset)
   

    # Use IP webcam (replace with your IP webcam URL)
        
    # Main loop to process frames
    while True:
        ret, frame = cap.read()  # Capture frame-by-frame
        
        if not ret:
            break

        # Process frame at 1920x1080 for OCR and YOLO detection
        frame_ocr = cv2.resize(frame, (640, 480))

        # Perform object detection using YOLOv8 (on GPU)
        results = model(frame_ocr, stream=True)
        pack_bottle_count = 0
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                # Shrink the bounding box slightly to make it tighter
                padding_x = int((x2 - x1) * 0.02)  # Reduce width by 5%
                padding_y = int((y2 - y1) * 0.02)  # Reduce height by 5%

                # Apply the padding to tighten the bounding box
                x1 += padding_x
                y1 += padding_y
                x2 -= padding_x
                y2 -= padding_y
                class_idx = int(box.cls[0])
                label = custom_labels.get(class_idx, 'Unknown')

                cropped_img = frame_ocr[y1:y2, x1:x2]
                if label in ['Pack', 'Bottle']:
                    pack_bottle_count += 1  # Increment the count
                        
                    # Calculate the width and height of the bounding box
                    width = x2 - x1
                    height = y2 - y1
                        
                    # Display the dimensions on the frame
                    cv2.putText(frame_ocr, f"{label} (W: {width}, H: {height})", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (0, 255, 255), 1)
                    dominant_color = get_dominant_color(cropped_img)
                    # Draw a filled rectangle for the dominant color below the detected object
                    cv2.rectangle(frame_ocr, (x1, y2), (x2, y2 + 30), dominant_color, -1)  # Draw filled rectangle for dominant color
                    # Add the "Dominant Color" text just below the color rectangle
                    cv2.putText(frame_ocr, "Dominant Color", (x1, y2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (255, 255, 255), 1)  # White text-1)  # Black rectangle

                    # Add text on top of the rectangle
                    #cv2.putText(frame_ocr, f"Dominant Color", (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # White text


                if label in ['MRP', 'Net']:
                    best_text = extract_best_text_360(cropped_img)
                elif label in ['Date', 'Flavour']:
                    best_text = extract_best_text_4_directions(cropped_img)
                elif label in ['Product_name', 'Brand_name']:
                    best_text = reader.readtext(cropped_img, detail=0)
                    best_text = ' '.join(best_text) if best_text else None
                else:
                    best_text = None

                if best_text:
                    extracted_dates = extract_dates([best_text])
                    print(f"Detected {label}: {best_text}")

                    # Append the text to the corresponding list based on the label
                    if label == 'Brand_name':
                        brand_name_list.append(best_text)
                    elif label == 'Date':
                        date_list.append(best_text)
                        date_list.append(str(extracted_dates))
                    elif label == 'Due':
                        due_list.append(best_text)
                    elif label == 'Flavour':
                        flavour_list.append(best_text)
                    elif label == 'MRP':
                        mrp_list.append(best_text)
                    elif label == 'Net':
                        net_list.append(best_text)
                    elif label == 'Product_name':
                        product_name_list.append(best_text)

                    if extracted_dates:
                        print(f"Detected Dates_test_______________: {', '.join(extracted_dates)}")
                        date_list.append(str(extracted_dates))

                cv2.rectangle(frame_ocr, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(frame_ocr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (0, 255, 0), 1)
        
        # Display the count of "Pack" and "Bottle" on the frame
        cv2.putText(frame_ocr, f"Pack/Bottle Count: {pack_bottle_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0), 2)
            
        # Resize frame for display to 640x480
        #frame_display = cv2.resize(frame_ocr, (640, 480))
        _, buffer = cv2.imencode('.jpg', frame_ocr)
        frame_data = base64.b64encode(buffer).decode('utf-8')

            # Send the frame to the WebSocket client
        await websocket.send_text(frame_data)

        await asyncio.sleep(0.05)

    cap.release()
    cv2.destroyAllWindows()

def return_detected_lists_every_cum_seconds():
    while True:
        time.sleep(10)
        print("\n--- Lists every for a product ---")
        remove_redundancy_and_print()
        print("-----------------------------")
        

@app.websocket("/video_stream")
async def video_stream(websocket: WebSocket):
    try:
        thread = threading.Thread(target=return_detected_lists_every_cum_seconds)
        thread.daemon = True
        thread.start()
        await capture_video(websocket)
        
    except WebSocketDisconnect:
        print("Client disconnected")
        thread._stop() 


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="127.0.0.2", port=8001)