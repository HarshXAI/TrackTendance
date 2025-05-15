# Smart Attendance System

A facial recognition-based attendance tracking system with an interactive Streamlit interface.

## Features

- Live attendance tracking via webcam
- Process pre-recorded videos for attendance
- View and export attendance records
- Adaptive recognition using LSTM model with clarity scoring

## Setup and Installation

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Usage

### Live Attendance

1. Navigate to the "Live Attendance" page
2. Enter subject name and lecture slot 
3. Click "Start Camera" to begin tracking
4. Students will be recognized and marked present automatically
5. Click "Stop" when the session ends

### Process Video

1. Navigate to the "Process Video" page
2. Upload a video file (.mp4, .avi, .mov)
3. Enter subject and lecture slot details
4. Click "Process Video" 
5. View the results once processing completes

### View Records

1. Navigate to the "View Records" page
2. Set date range, subject, and optional SAP ID filters
3. Click "Search Records" to view matching attendance data
4. Download as CSV if needed

### Settings

Adjust recognition thresholds and view system information.
