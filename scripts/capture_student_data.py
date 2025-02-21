import cv2
import os
import sys
from datetime import datetime

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_manager import DatabaseManager
from models.feature_extractor import FeatureExtractor

def capture_student_photos():
    db = DatabaseManager()
    feature_extractor = FeatureExtractor()
    
    # Get student details
    sap_id = input("Enter SAP ID: ")
    name = input("Enter student name: ")
    year = int(input("Enter year (1-4): "))
    branch = input("Enter branch: ")
    gmail = input("Enter college gmail: ")
    
    # Create directory for temporary storage
    output_dir = f"../data/temp_captures/{sap_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    # Counter for captured images
    img_counter = 0
    max_images = 10  # Changed to 10 images
    
    print("\nInstructions:")
    print("1. Press 'c' to capture image")
    print("2. Press 'q' to quit")
    print(f"3. Need to capture {max_images} images with different angles")
    print("\nCapturing started...")
    
    while img_counter < max_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Add counter display on frame
        display_frame = frame.copy()
        cv2.putText(
            display_frame,
            f"Captured: {img_counter}/{max_images}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Show frame
        cv2.imshow("Capture", display_frame)
        
        # Handle key press
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            if img_counter < max_images:
                print(f"\nWarning: Only {img_counter} images captured. Need {max_images} images.")
                confirm = input("Do you want to quit anyway? (y/n): ")
                if confirm.lower() == 'y':
                    break
            else:
                break
        elif key == ord('c'):
            # Add delay indicator
            cv2.putText(
                display_frame,
                "Capturing...",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
            cv2.imshow("Capture", display_frame)
            cv2.waitKey(500)  # Small delay to show the capturing message
            
            img_name = f"{sap_id}_{img_counter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            img_path = os.path.join(output_dir, img_name)
            cv2.imwrite(img_path, frame)
            print(f"Captured {img_name}")
            img_counter += 1
            
            if img_counter == max_images:
                print("\nAll required images captured!")
    
    cap.release()
    cv2.destroyAllWindows()
    
    if img_counter < max_images:
        print(f"\nWarning: Only {img_counter}/{max_images} images were captured.")
        proceed = input("Do you want to proceed with these images? (y/n): ")
        if proceed.lower() != 'y':
            print("Aborting registration...")
            # Cleanup temporary files
            for file in os.listdir(output_dir):
                os.remove(os.path.join(output_dir, file))
            os.rmdir(output_dir)
            return
    
    print("\nProcessing images...")
    # Process captured images and extract embeddings
    embeddings = []
    for img_name in os.listdir(output_dir):
        if img_name.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(output_dir, img_name)
            img = cv2.imread(img_path)
            embedding = feature_extractor.extract_features(img)
            embeddings.append(embedding)
    
    # Store in database
    student_data = {
        "sap_id": sap_id,
        "name": name,
        "year": year,
        "branch": branch,
        "gmail": gmail
    }
    
    try:
        db.add_student(student_data, embeddings)
        print(f"\nSuccessfully registered/updated student {name} (SAP ID: {sap_id})")
    except Exception as e:
        print(f"Error registering student: {str(e)}")
        print("Please try again or contact system administrator.")
    finally:
        # Cleanup temporary files
        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file))
        os.rmdir(output_dir)

if __name__ == "__main__":
    capture_student_photos()
