
Jawad konialee 2104740

Fawzy kerret 2018733

Abdulrahman shaar 2018917


# object detection with esp32 cam using custom data with yolov8




we will be collecting data of humans(male and female), dogs and cats, and short hair and long hair in uxga res.(1000 images in total) using the esp32 camera module then train the data to end up with a object identification model.


# ESP32/ESP8266 Telegram Bot LED Controller

This project allows you to control an LED connected to your ESP32/ESP8266 using a Telegram bot. You can turn the LED on or off, and request its current state by sending commands through Telegram.

## Libraries Used

- **WiFi.h / ESP8266WiFi.h**: These libraries handle Wi-Fi connections for ESP32 and ESP8266 devices, respectively.
- **WiFiClientSecure.h**: This library manages secure Wi-Fi client connections, ensuring encrypted communication with the Telegram API.
- **UniversalTelegramBot.h**: A library to interact with the Telegram Bot API, enabling the ESP32/ESP8266 to send and receive messages from your Telegram bot.
- **ArduinoJson.h**: This library is used to parse JSON data received from Telegram, allowing for easy message handling.

## Setup Instructions

1. Install the required libraries in the Arduino IDE:
   - WiFi (for ESP32 or ESP8266)
   - WiFiClientSecure
   - UniversalTelegramBot
   - ArduinoJson

2. Set up a Telegram bot:
   - Create a new bot via [BotFather](https://core.telegram.org/bots#botfather) on Telegram.
   - Obtain your bot's token.
   - Find your chat ID using [@myidbot](https://t.me/myidbot).

3. Edit the following variables in the code:
   - `ssid`: Your Wi-Fi SSID.
   - `password`: Your Wi-Fi password.
   - `BOTtoken`: Your Telegram bot token.
   - `CHAT_ID`: Your chat ID.

4. Upload the code to your ESP32/ESP8266 and open the Serial Monitor. The ESP device will connect to Wi-Fi, and you can start sending commands to control the LED.

## Commands

- **/led_on**: Turns the LED ON.
- **/led_off**: Turns the LED OFF.
- **/state**: Returns the current state of the LED.

## Code Explanation

- The code connects the ESP32/ESP8266 to your Wi-Fi network.
- It then connects to the Telegram Bot API using the `UniversalTelegramBot` library.
- The bot checks for new messages every second, handling commands like `/led_on`, `/led_off`, and `/state`.
- Depending on the command received, the LED connected to GPIO2 is turned ON or OFF.

# step 2 
since we are done collecting the data we can move to annotations:

 1. using roboflow as a platform to annotate our data
 2. manually annotating the data 



# YOLOv8 Colab Training Setup

This README guides you through setting up YOLOv8 training on Google Colab with dataset preparation, model training, and video processing steps. The following steps are detailed, starting from checking GPU availability to processing video files.

---

## Table of Contents
1. [Step 1: Checking GPU Availability](#step-1-checking-gpu-availability)
2. [Step 2: Unzipping Dataset Files](#step-2-unzipping-dataset-files)
3. [Step 3: Installing Ultralytics](#step-3-installing-ultralytics)
4. [Step 4: Mounting Google Drive](#step-4-mounting-google-drive)
5. [Step 5: Define Root Directory and Create Subdirectories](#step-5-define-root-directory-and-create-subdirectories)
6. [Step 6: Organize Dataset for Training and Validation](#step-6-organize-dataset-for-training-and-validation)
7. [Step 7: Generate YAML Configuration File](#step-7-generate-yaml-configuration-file)
8. [Step 8: Train the YOLOv8 Model](#step-8-train-the-yolov8-model)
9. [Step 9: View Training Results](#step-9-view-training-results)
10. [Step 10: Process Video Files](#step-10-process-video-files)
11. [Step 11: Display Processed Video](#step-11-display-processed-video)

---

## Step 1: Checking GPU Availability
Check if a GPU is available in your Google Colab environment. A GPU can significantly accelerate the training process of deep learning models like YOLOv8.

```python
!nvidia-smi
```

## Step 2: Unzipping Dataset Files
Unzip the dataset files containing images and annotations. Ensure the images and their corresponding annotation files have the same names.

```python
!unzip -q '/content/train.zip' -d '/content/images'
!unzip -q '/content/test.zip' -d '/content/annotations'
```

## Step 3: Installing Ultralytics
Install Ultralytics, a library that simplifies working with YOLO object detection models.

```python
!pip install ultralytics
from ultralytics import YOLO
```

## Step 4: Mounting Google Drive
Mount Google Drive to the Colab environment to access files stored in your Google Drive.

```python
from google.colab import drive
drive.mount('/content/Google_Drive')
```

## Step 5: Define Root Directory and Create Subdirectories
Define the root directory and create necessary subdirectories to organize your data.

```python
import os

PATH = input("Enter the desired path: ")

ROOT_DIR = '/content/' + PATH

DATA_DIR = os.path.join(ROOT_DIR, 'data')
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
IMAGES_VAL_DIR = os.path.join(IMAGES_DIR, 'val')
IMAGES_TRAIN_DIR = os.path.join(IMAGES_DIR, 'train')

LABELS_DIR = os.path.join(DATA_DIR, 'labels')
LABELS_VAL_DIR = os.path.join(LABELS_DIR, 'val')
LABELS_TRAIN_DIR = os.path.join(LABELS_DIR, 'train')

TESTING_DIR = os.path.join(ROOT_DIR, 'testing')

if not os.path.exists(ROOT_DIR):
    os.makedirs(ROOT_DIR)
    os.makedirs(DATA_DIR)
    os.makedirs(IMAGES_DIR)
    os.makedirs(IMAGES_VAL_DIR)
    os.makedirs(IMAGES_TRAIN_DIR)
    os.makedirs(LABELS_DIR)
    os.makedirs(LABELS_VAL_DIR)
    os.makedirs(LABELS_TRAIN_DIR)
    os.makedirs(TESTING_DIR)

    print(f"Root directory '{ROOT_DIR}' created successfully.")
else:
    print(f"Root directory '{ROOT_DIR}' already exists.")
```

## Step 6: Organize Dataset for Training and Validation
Organize the dataset by moving images and annotations into separate directories for training and validation.

```python
import shutil

IMAGES_PATH = '/content/images/train/images'
ANNOTATIONS_PATH = '/content/images/train/labels'

image_files = os.listdir(IMAGES_PATH)
annotation_files = os.listdir(ANNOTATIONS_PATH)

image_files.sort()
annotation_files.sort()

train_count = int(len(image_files) * 0.9)

for file in image_files[:train_count]:
    shutil.move(os.path.join(IMAGES_PATH, file), os.path.join(IMAGES_TRAIN_DIR, file))
for file in annotation_files[:train_count]:
    shutil.move(os.path.join(ANNOTATIONS_PATH, file), os.path.join(LABELS_TRAIN_DIR, file))

for file in image_files[train_count:]:
    shutil.move(os.path.join(IMAGES_PATH, file), os.path.join(IMAGES_VAL_DIR, file))
for file in annotation_files[train_count:]:
    shutil.move(os.path.join(ANNOTATIONS_PATH, file), os.path.join(LABELS_VAL_DIR, file))
```

## Step 7: Generate YAML Configuration File
Create a YAML configuration file specifying the paths to the training and validation datasets, as well as the class names used in your dataset.

```python
import yaml

data = {
    'path': f'{DATA_DIR}',
    'train': 'images/train',
    'val': 'images/val',
   'names': {0: 'Andrew tate', 1: 'female', 2: 'long hair', 3: 'male', 4: 'short hair', 5: 'cats', 6:'dogs'}
}

output_file = os.path.join(ROOT_DIR, "config.yaml")

with open(output_file, 'w') as yaml_file:
    yaml.dump(data, yaml_file)
```

## Step 8: Train the YOLOv8 Model
Train the YOLOv8 model using the provided dataset configuration.

```python
model = YOLO("yolov8n.pt")
model_results = model.train(data=os.path.join(ROOT_DIR, "config.yaml"), epochs=50)

shutil.make_archive(base_dir='/content/runs', root_dir='/content/runs', format='zip', base_name=f'{ROOT_DIR}/runs')
```

## Step 9: View Training Results
Display the training results and confusion matrix.

```python
from IPython.display import Image
Image('runs/detect/train2/results.png')

Image('runs/detect/train/confusion_matrix.png')
```

## Step 10: Process Video Files
Process video files using the trained YOLOv8 model and save the processed videos with bounding boxes and class labels.

```python
import cv2

OUTPUT_DIR = '/content/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

model_file = "/content/runs/detect/train/weights/best.pt"
model = YOLO(model_file)

video_files = os.listdir(TESTING_DIR)

for video_file in video_files:
    cap = cv2.VideoCapture(os.path.join(TESTING_DIR, video_file))

    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(f'{OUTPUT_DIR}/{video_file}_object_detection.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    ret, frame = cap.read()
    while ret:
        results = model(frame)[0]

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

            label = f'{results.names[int(class_id)]}: {score:.2f}'
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        out.write(frame)
        ret, frame = cap.read()

    cap.release()
    out.release()
    cv2.destroyAllWindows()
```

## Step 11: Display Processed Video
Use the MoviePy library to display the processed video within the Colab environment.

```python
from moviepy.editor import *
path = 'REPLACE_WITH_PATH_TO_VIDEO_FILE'
clip = VideoFileClip(path)
clip.ipython_display(height=540, width=960)
```

---

This setup allows you to train YOLOv8 for object detection tasks, generate training results, and process video files with bounding boxes and class labels. Make sure to upload the necessary dataset files and videos to Google Colab for proper execution of each step.

# Integrating our results into the esp32 cam

when it comes to esp32 it is a tinyml used for mini projects and its ability to handle programs like yolov8 is very unlikely but through camera webhosting and importing the url to opencv, having real time object detection with the Esp32 is possible!

# ESP32-CAM MJPEG Streamer

This project sets up an **ESP32-CAM** module as an MJPEG stream server, streaming live video over a WiFi network. The camera is configured for VGA resolution and high quality, with MJPEG (Motion JPEG) video encoding.

## Features:
- **MJPEG streaming**: Broadcasts video captured from the ESP32-CAM.
- **WiFi connectivity**: Connects to a specified WiFi network.
- **Easy setup**: Configurable with basic settings like WiFi credentials and camera resolution.

## Requirements

- **ESP32-CAM module**: Ensure your ESP32-CAM is properly wired and functional.
- **FTDI adapter**: If using the ESP32-CAM board that doesn't have a USB connector, you'll need an FTDI adapter for programming.
- **Wiring**: Connect the ESP32-CAM board to your computer for programming via the FTDI adapter.

## Software Setup

1. **Install Arduino IDE**:
   - Download and install the Arduino IDE from [here](https://www.arduino.cc/en/software).
   - Add the ESP32 board support to the Arduino IDE:
     1. Go to `File` -> `Preferences`.
     2. In the "Additional Boards Manager URLs" field, add `https://dl.espressif.com/dl/package_esp32_index.json`.
     3. Go to `Tools` -> `Board` -> `Boards Manager`, search for "esp32" and install the latest version.

2. **Install Libraries**:
   - Install the `eloquent_esp32cam` library from the Arduino Library Manager:
     - Go to `Sketch` -> `Include Library` -> `Manage Libraries`.
     - Search for `eloquent_esp32cam` and install it.

3. **Configure Code**:
   - In the `#define WIFI_SSID` line, replace `---` with the name of your WiFi network.
   - In the `#define WIFI_PASS` line, replace `---` with the password for your WiFi network.

4. **Upload the Code**:
   - Connect your ESP32-CAM to your computer.
   - Select the correct board and port in the Arduino IDE (`Tools` -> `Board` -> `ESP32 Wrover Module` and `Tools` -> `Port`).
   - Upload the code to the ESP32-CAM module.

5. **Monitor the Output**:
   - Open the Serial Monitor (`Tools` -> `Serial Monitor`) in the Arduino IDE.
   - Set the baud rate to `115200`.
   - Once the ESP32-CAM successfully connects to the WiFi, it will print out its MJPEG stream URL (e.g., `http://192.168.x.x:8080`).

6. **Access the MJPEG Stream**:
   - Open a web browser and enter the MJPEG stream URL printed in the Serial Monitor (e.g., `http://192.168.x.x:8080`).
   - You should now see a live MJPEG stream from the ESP32-CAM.

## Code Overview

```cpp
#define WIFI_SSID "---"
#define WIFI_PASS "---"
#define HOSTNAME  "esp32cam"

#include <eloquent_esp32cam.h>
#include <eloquent_esp32cam/viz/mjpeg.h>

using namespace eloq;
using namespace eloq::viz;

/**
 *  Setup function
 */
void setup() {
    delay(3000);
    Serial.begin(115200);
    Serial.println("___MJPEG STREAM SERVER___");

    // Camera settings
    camera.pinout.aithinker(); // Set camera pinout for AI-Thinker board
    camera.brownout.disable(); // Disable brownout detector
    camera.resolution.vga();   // Set resolution to VGA
    camera.quality.high();     // Set quality to high

    // Initialize the camera
    while (!camera.begin().isOk())
        Serial.println(camera.exception.toString());

    // Connect to WiFi
    while (!wifi.connect().isOk())
        Serial.println(wifi.exception.toString());

    // Start MJPEG HTTP server
    while (!mjpeg.begin().isOk())
        Serial.println(mjpeg.exception.toString());

    Serial.println("Camera OK");
    Serial.println("WiFi OK");
    Serial.println("MjpegStream OK");
    Serial.println(mjpeg.address()); // Print the MJPEG stream URL
}

/**
 *  Main loop (nothing needed)
 */
void loop() {
    // The MJPEG server runs in its own task
    // No need to do anything in the main loop
}
```
# opencv yolo model to url real time stream

## Code

```python
from ultralytics import YOLO
import cv2

# Load the YOLOv8 model
model = YOLO('/Users/jawadkon/Downloads/best.pt')  # Replace 'best.pt' with the path to your trained YOLOv8 model

# Set up video capture (e.g., webcam or IP camera)
url = "http://192.168.1.88:81"  # Replace with your IP camera URL
cap = cv2.VideoCapture(url)  # For webcam, use 0

# Process frames from the video feed
while True:
    success, frame = cap.read()
    if not success:
        break

    # Perform inference on the frame
    results = model(frame)

    # Render the detected objects on the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
```


