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
import cv2
from ultralytics import YOLO
import torch
import time
# from index import cap
from index import device

ip_webcam_url = "http://192.168.187.1:8080/video"  # Replace with your actual IP

    # Initialize the IP webcam
cap = cv2.VideoCapture(0)

reader = easyocr.Reader(['en'])
app = FastAPI()
# ip_webcam_url = "http://192.168.:8080/video"  # Replace with your actual IP

    # Initialize the IP webcam
# cap = cv2.VideoCapture(0)
def insert_data(fruit,condition,estimated_shelf_time,):
    # Connect to the SQLite database
    conn = sqlite3.connect('example.db')
    try:
        with conn:
            conn.execute('''
                INSERT INTO fruit (fruit,freshness_index,estimated_shelf_time)
                VALUES (?, ?, ?)
            ''', (fruit,condition,estimated_shelf_time))
        
        print("Data inserted:", {
            'fruit_name':fruit,
            'freshness_index':condition,
            "estimated_shelf_time":estimated_shelf_time
        })
    except:
        print("error occured")

freshness_model = YOLO("try2.pt").to(device)

# Load the classification model (e.g., YOLO for fruit/vegetable classification)
classification_model = YOLO("classification.pt").to(device)

# Set up the IP webcam URL (replace with your actual IP and port)
ip_webcam_url = 'http://100.66.252.47:8080/video'  # Replace with your IP webcam URL

# Initialize video capture from IP webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream from IP webcam.")
    exit()

# Define known maximum shelf life (in days) for each class of fruit or vegetable
shelf_life_data = {'apple': 30, 'banana': 7, 'cabbage': 14,  'capsicum': 10,  'carrot': 30,'cauli': 10, 'chilli pepper': 14, 'cucumber': 14, 'grapes': 21, 'kiwi': 14, 'lemon': 30, 'mango': 10, 'orange': 30, 'pear': 15, 'pineapple': 7, 'pomegranate': 30, 'potato': 60,  'tomato': 10, 'watermelon': 7}


def estimate_shelf_time(confidence, max_shelf_life):
    remaining_shelf_time = max_shelf_life * confidence  # Adjusting by confidence score
    return remaining_shelf_time


# Track the time to print every 3 seconds


async def capture_fruit_video(websocket: WebSocket):
    await websocket.accept()
    last_print_time = time.time()
    global processed_frame
    # model = YOLO('fruit.pt')
    # Main loop to process frames
    while True:
    # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image from webcam.")
            break
        
        # Resize the frame
        frame = cv2.resize(frame, (640, 480))

        # Step 1: Run the freshness detection model to get the confidence score (if available)
        freshness_results = freshness_model.predict(source=frame, show=False)

        # Check if results are available
        if freshness_results and len(freshness_results) > 0:
            # Access the first result in the list (assuming a single detection for simplicity)
            first_result = freshness_results[0]

            # Check if there are bounding boxes (detections)
            if first_result.boxes is not None and len(first_result.boxes) > 0:
                # Get the first box's confidence score
                freshness_confidence = first_result.boxes[0].conf.item()  # Get the confidence score of detection
                
                # Draw bounding boxes on the frame
                box = first_result.boxes[0]
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Get bounding box coordinates
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw rectangle on the frame

                # Step 2: Run the classification model to get the predicted class and known shelf life
                classification_results = classification_model.predict(source=frame, show=False)  # Classification model inference
                
                if classification_results and len(classification_results) > 0:
                    # Get classification label
                    predicted_class_idx = classification_results[0].probs.top1  # Top-1 predicted index
                    predicted_class = classification_results[0].names[predicted_class_idx]  # Get class name
                    
                    # Fetch shelf life from predefined data
                    max_shelf_life = shelf_life_data.get(predicted_class, None)

                    if max_shelf_life:
                        # Step 3: Estimate remaining shelf time based on confidence and max shelf life
                        remaining_shelf_time = estimate_shelf_time(freshness_confidence, max_shelf_life)

                        # Display the predicted class, freshness confidence, and remaining shelf time on the frame
                        label = f"{predicted_class}: {freshness_confidence:.2f}, Shelf Time: {remaining_shelf_time:.2f} days"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Print information every 3 seconds
                        current_time = time.time()
                        if current_time - last_print_time >= 3:
                            print(f"Fruit/vegetable Name: {predicted_class}")

                            print(f"Freshness Index score: {freshness_confidence:.2f}")
                        
                            print(f"Estimated shelf time: {remaining_shelf_time:.2f} days\n")
                            last_print_time = current_time  # Reset the print timer
                            insert_data(predicted_class,freshness_confidence,remaining_shelf_time)
                    else:
                        cv2.putText(frame, f"Class '{predicted_class}' not found", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    print("No classification results available.")
            else:
                print("No detections in the freshness results.")
        else:
            print("No freshness detection results available.")

        # Display the resulting frame with bounding boxes and labels
        cv2.imshow('Fruit and Vegetable Freshness Detection', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
        _, buffer = cv2.imencode('.jpg', frame)
        
        frame_data = base64.b64encode(buffer).decode('utf-8')

            # Send the frame to the WebSocket client
        await websocket.send_text(frame_data)

        await asyncio.sleep(0.05)    
        # Display the live video feed with predictions
    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

    time.sleep(5)  # Wait before attempting to reconnect



@app.websocket("/fruit_stream")
async def video_stream(websocket: WebSocket):
    try:
        
        await capture_fruit_video(websocket)
        
    except WebSocketDisconnect:
        print("Client disconnected")


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="127.0.0.3", port=8002)