# FLIPKART GRID 6.0 - ROBOTICS CHALLENGE
## SOLUTION BY TEAM SABARI132005



## Table of contents

- [Overview](#overview)
- [Screenshot](#screenshot)
- [Architecture Diagram](#my-process)
- [Built With](#built-with)
- [Installation](#installation)
- [Project Structure](#structure)
- [Result Analysis](#resultanalysis)
- [Database Storage](#database)
- [Author](#author)


## Overview
- This is a solution to the [Smart Vision Technology Quality Control in Flipkart Grid 6.0 - Robotics Challenge](https://unstop.com/hackathons/flipkart-grid-60-robotics-challenge-flipkart-grid-60-flipkart-1024253). 
- Our project is a real-time Product Detection & Freshness Monitoring System, leveraging YOLOv8 Object Detection to identify products, categorize them by brand, and evaluate the Freshness of Fruits and Vegetables and store it in the DB.


## Screenshot
- ### Product Detection, Recognition and Extraction
  <img  src="./assets/image1.jpg">
  <img src="./assets/image2.jpg">

- ### Brand based Counting
  <img src="./assets/count1.jpg">
  <img src="./assets/count2.jpg">

- ### Fruit Freshness Identification
  <img src="./assets/output.jpg">
## Architecture Diagram

![](assets/architecture.jpg)

## Built With

- ### Frontend:
  - HTML, CSS, JS

- ### Backend:
  - FastAPI
  - Python
  - YOLOv8 for object detection
  - YOLOv11 (Instant Segmentation) for Detecting Fruits & Vegetables Freshness
  - K-Means Clustering Algorithm
- ### Database
  - Sqlite3
- ### Libraries
  - `Ultralytics`
  - `Opencv-python`
  - `numpy`, `pandas` for Data Handling
  - `uvicorn` for FastAPI

## Installation

### Prerequrements
### Download the model and upload inside FlipKart_6.0
  - [Download](https://drive.google.com/file/d/1r2YWdnmE_Z7NA7dUX96Cw7Ie8TmvcU-x/view?usp=drive_link)
  

  - `Python3.11`
  - `DB Browser` - To see the database

### Installation steps

  ```
    git clone https://github.com/Sabari2005/FlipKart-Grid_6.0
    cd FlipKart-Grid_6.0
  ```
  ```
  pip install -r requirements.txt
  ```
  - In the `mode1_product.py`  `line 62` replace your Webcam URL

![](assets/url1.png)

  - In the `model2_fruit.py` `line 31` replace your Webcam URL

 ![](assets/url2.png) 


  - Execute each commands in a separate terminal
  ```
  python index.py
  python model1_fastapi.py
  python model2_fastapi.py
  ```
  - Open `http://127.0.0.1:8000` in your Browser

## Project structure

```
├──          
├── static
│   ├── css                    # Frontend styling files
│   └── images                 # Static images (if any)
├── templates
│   └── index.html             # Main HTML page
├── index.py   
├── model1_fastapi.py          
├── model2_fastapi.py                 
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```
## Result Analysis

- ### Product details recognition and extraction model
    ![](assets/detection.png) 
    ![](assets/product_model.png) 
    ![](assets/results.png) 
- ### Fruits & Vegetables Freshness Detection Model

## Sample model Output
- ### Product details Detection and Recognition Model(Real-time Images)
  ![](assets/real-time.png) 

  ![](assets/real-time2.png) 

## Database Storage

- You can see the product details in

## Author

- Sabari Vadivelan S (Team Leader)
- Kamal M (Member 1)
- Uvarajan D (Member 2)
- Kaviarasu K (Member 3)
- Sarathi S (Member 4)


## See our PPT here

- [Download](https://docs.google.com/presentation/d/13D5QP3yGv6fAD_d8x2nDoX0ZLpIVwJJX/edit?usp=drive_link&ouid=108471139347338857018&rtpof=true&sd=true)
