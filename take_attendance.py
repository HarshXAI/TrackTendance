import os
from attendance_system import AttendanceSystem
from datetime import datetime

def main():
    # Get lecture details
    print("\nAttendance System")
    print("-----------------")
    subject = input("Enter subject name: ")
    lecture_slot = input("Enter lecture slot (e.g., Monday 10:00 AM): ")
    
    # Get video source
    print("\nVideo Source:")
    print("1. Use webcam")
    print("2. Use video file")
    choice = input("Enter your choice (1/2): ")
    
    if choice == "2":
        video_path = input("\nEnter video file path: ")
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            return
        video_source = video_path
    else:
        video_source = 0  # Use webcam
    
    # Initialize system
    system = AttendanceSystem()
    
    print("\nStarting attendance capture...")
    print("Press 'q' to stop capturing")
    
    # Take attendance
    attendance_log = system.process_video(
        video_source=video_source,
        subject=subject,
        lecture_slot=lecture_slot
    )
    
    # Print results
    print("\nAttendance Results:")
    print(f"Subject: {subject}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nStudents Present:")
    for sap_id, entry in attendance_log.items():
        student = entry['student']
        confidence = entry['confidence']
        print(f"- {student.name} (SAP ID: {sap_id}) - Confidence: {confidence:.2f}")

if __name__ == "__main__":
    main()
