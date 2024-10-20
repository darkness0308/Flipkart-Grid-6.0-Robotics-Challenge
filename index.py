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
app = FastAPI()
latest_data = None
latest_fruit_data=None
executor = ThreadPoolExecutor(max_workers=5)
# Mount static files (for serving HTML)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

MONTH_CORRECTIONS = {'JAN': 'JAN', 'JAM': 'JAN', 'JANU': 'JAN', 'JAU': 'JAN', 'FEB': 'FEB', 'FEE': 'FEB', 'FE8': 'FEB', 'FEP': 'FEB', 'MAR': 'MAR', 'MAA': 'MAR', 'MARCH': 'MAR', 'APR': 'APR', 'FIPR': 'APR', 'FipR': 'APR', 'APRL': 'APR', 'FFR':'APR','FPR':'APR','APRIL': 'APR', 'MAY': 'MAY','MAT':'MAY', 'MAN': 'MAY','MFM':'MAY','MIFN':'MAY','MFN':'MAY','MFN': 'MAY', 'MIFM': 'MAY', 'MIFN': 'MAY', 'JUN': 'JUN', 'JUNE': 'JUN', 'JWN': 'JUN', 'JUL': 'JUL', 'JULY': 'JUL', 'JLI': 'JUL', 'JLY': 'JUL', 'AUG': 'AUG', 'AU6': 'AUG', 'AUCT': 'AUG', 'AUGUST': 'AUG', 'AU8': 'AUG', 'AUC': 'AUG', 'AUQ': 'AUG', 'AU9': 'AUG', 'SEP': 'SEP', 'SEPT': 'SEP', 'SEPTEMBER': 'SEP', '5EP': 'SEP', '5EPT': 'SEP', 'SFP': 'SEP', 'SFP7': 'SEP', 'SPT': 'SEP', 'S9P': 'SEP', 'OCT': 'OCT', 'OCTOBER': 'OCT', '0CT': 'OCT', '0C7': 'OCT', 'OCt': 'OCT', 'OCTO': 'OCT', 'OQT': 'OCT', 'OC0': 'OCT', 'NOV': 'NOV', 'NOVEMBER': 'NOV', 'NQV': 'NOV', 'MOV': 'NOV', 'NOVEM': 'NOV', 'NOY': 'NOV', 'NOVB': 'NOV', 'NOVF': 'NOV', 'DEC': 'DEC', 'DECEMBER': 'DEC', 'DE0': 'DEC', 'D6C': 'DEC', 'OEC': 'DEC', 'DECEM': 'DEC', 'DECMB': 'DEC', 'DLC': 'DEC', 'D8C': 'DEC', 'DEC2': 'DEC'}
df = pd.read_csv('comdat.csv')


# Check if CUDA is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_data():
    conn = sqlite3.connect('example.db')
    cursor = conn.cursor()

    # Fetch all products from the products table
    cursor.execute("""
        SELECT id, product_name, brand_name, net, mrp, expiry_date, valid
        FROM products
    """)
    products = cursor.fetchall()

    # Fetch all descriptions from the description table
    cursor.execute("""
        SELECT product_name, brand_name, description
        FROM description
    """)
    descriptions = cursor.fetchall()

    conn.close()

    # Convert descriptions to a dictionary for quick lookup
    description_map = {(d[0], d[1]): d[2] for d in descriptions}  # {(product_name, brand_name): description}

    # Prepare the data by comparing products and descriptions
    escaped_data = []
    for product in products:
        product_id = product[0]
        product_name = product[1]
        brand_name = product[2]
        net = product[3]
        mrp = product[4]
        expiry_date = product[5]
        valid=product[6]

        # Check if there is a matching description for the product
        description = description_map.get((product_name, brand_name), None)

        # Escape all fields and ensure description is escaped if available
        escaped_product = {
            "product_id": product_id,
            "product_name": product_name or '',
            "brand_name": brand_name or '',
            "net": net or '',
            "mrp": mrp or '',
            "expiry_date": expiry_date or '',
            "valid":valid or '',
            "description": html.escape(description or '', quote=True) if description else None
        }

        # Add the escaped product data to the list
        escaped_data.append(escaped_product)

    return escaped_data# Function to capture video frames and send to client


# Home route
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
    # with open("static/index.html") as f:
    #     return f.read().replace("{{ current_time }}", str(current_time))

@app.get("/get_data")
async def get_latest_data(background_tasks: BackgroundTasks):
    global latest_data
    current_data = get_data()
    loop = asyncio.get_event_loop()
    # If the data has changed (new products), send the updated data to the client
    if current_data != latest_data:
        new_data = []

        # Check for new products by comparing with the previously stored latest_data
        if latest_data is not None:
            old_products = set((prod["product_id"]) for prod in latest_data)  # Set of product_ids in latest_data
        else:
            old_products = set()

        # Identify new products and get their details including the description
        for product in current_data:
            product_id = product["product_id"]

            # Only consider new products (not in old_products)
            if product_id not in old_products:
                new_data.append(product)

        # Update the latest_data to include the newly fetched products
        latest_data = current_data

        # Send new product data (including description if available)
        if new_data:
            return new_data

    # If no new data is found, return a message
    return {"message": "no new data found"}

def get_fruit_data():
    conn = sqlite3.connect('example.db')
    cursor = conn.cursor()

    # Fetch all products from the products table
    cursor.execute("""
        SELECT id, fruit, freshness_index, estimated_shelf_time
        FROM fruit
    """)
    fruits = cursor.fetchall()
    print(fruits)

    # Fetch all descriptions from the description table
    cursor.execute("""
        SELECT fruit, fruit_description
        FROM fruit_description
    """)
    fruit_description = cursor.fetchall()

    conn.close()

    # Convert descriptions to a dictionary for quick lookup
    description_map = {(d[0]): d[1] for d in fruit_description}  # {(product_name, brand_name): description}

    # Prepare the data by comparing products and descriptions
    escaped_data = []
    for fruit in fruits:
        fruit_id = fruit[0]
        fruit_name = fruit[1]
        freshness_index = fruit[2]
        estimated_shelf_time=fruit[3]
        # Check if there is a matching description for the product
        description = description_map.get((fruit_name), None)

        # Escape all fields and ensure description is escaped if available
        escaped_product = {
            "fruit_id": fruit_id,
            "fruit_name": fruit_name or '',
            "freshness_index": freshness_index or '',
            "estimated_shelf_time": estimated_shelf_time or '',
            "fruit_description": html.escape(description or '', quote=True) if description else None
        }

        # Add the escaped product data to the list
        escaped_data.append(escaped_product)

    return escaped_data# Function to capture video frames and send to client

@app.get("/get_fruit_data")
async def get_latest_data(background_tasks: BackgroundTasks):
    global latest_fruit_data
    current_data = get_fruit_data()
    loop = asyncio.get_event_loop()
    # If the data has changed (new products), send the updated data to the client
    if current_data != latest_fruit_data:
        new_data = []

        # Check for new products by comparing with the previously stored latest_data
        if latest_fruit_data is not None:
            old_products = set((prod["fruit_id"]) for prod in latest_fruit_data)  # Set of product_ids in latest_data
        else:
            old_products = set()

        # Identify new products and get their details including the description
        for product in current_data:
            product_id = product["fruit_id"]

            # Only consider new products (not in old_products)
            if product_id not in old_products:
                new_data.append(product)

        # Update the latest_data to include the newly fetched products
        latest_fruit_data = current_data

        # Send new product data (including description if available)
        if new_data:
            return new_data

    # If no new data is found, return a message
    return {"message": "no new data found"}

previous_result = None

def get_count_and_rows():
    global previous_result
    connection = sqlite3.connect('example.db')
    cursor = connection.cursor()

    # Query to get the product counts along with details
    cursor.execute('''
        SELECT brand_name, COUNT(*) AS product_count
        FROM products
        GROUP BY brand_name;
    ''')
    brand_counts = cursor.fetchall()

    # Fetch details for each brand
    brand_data = {}
    for brand_name, product_count in brand_counts:
        cursor.execute('''
            SELECT * FROM products
            WHERE brand_name = ?;
        ''', (brand_name,))
        product_rows = cursor.fetchall()
        
        brand_data[brand_name] = {
            "count": product_count,
            "rows": product_rows  # Store the rows associated with this brand
        }

    # Compare with previous result
    if brand_data == previous_result:
        return {"message": "no new data found"}
    else:
        previous_result = brand_data
        print(f'message": "Updated data found", "data": {brand_data}')
        return {"message": "Updated data found", "data": brand_data}

@app.get("/count_data")
async def get_latest_data():
    return get_count_and_rows()




if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)