import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os

from models.adaptive_lstm import AdaptiveLSTM
from data.student_database import StudentDatabase
from data.face_dataset import FaceSequenceDataset
from database.db_manager import DatabaseManager
from attendance_system import AttendanceSystem

def prepare_training_data(video_paths, student_database, attendance_system):
    sequences = []
    labels = []
    
    for video_path in tqdm(video_paths):
        # Process video and collect face sequences
        tracks = attendance_system.process_video(video_path)
        
        for track_id, track in tracks.items():
            if len(track['frames']) >= 10:  # Minimum sequence length
                sequence = attendance_system._prepare_sequence(track)
                student_id = attendance_system._identify_student(sequence)
                
                if student_id in student_database.student_ids:
                    sequences.append(sequence)
                    labels.append(student_database.student_ids.index(student_id))
    
    return sequences, labels

def train_model(model, train_loader, val_loader, num_epochs=50, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader):
            embeddings = batch['embedding_sequence'].to(device)
            clarity_scores = batch['clarity_scores'].to(device)
            labels = batch['label'].squeeze().to(device)
            
            optimizer.zero_grad()
            outputs = model(embeddings, clarity_scores)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                embeddings = batch['embedding_sequence'].to(device)
                clarity_scores = batch['clarity_scores'].to(device)
                labels = batch['label'].squeeze().to(device)
                
                outputs = model(embeddings, clarity_scores)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_loader)
        accuracy = 100. * correct / total
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Accuracy: {accuracy:.2f}%')
        
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

def main():
    try:
        # Initialize database manager
        db = DatabaseManager()
        
        # Get all students
        students = db.get_all_students()
        if not students:
            print("No students found in database. Please add students first.")
            return
        
        print(f"Found {len(students)} students in database")
        
        # Define paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        database_dir = os.path.join(current_dir, 'data', 'student_images')
        metadata_file = os.path.join(current_dir, 'data', 'metadata.json')
        
        if not os.path.exists(database_dir):
            os.makedirs(database_dir)
            print(f"Created directory: {database_dir}")
            
        if not os.path.exists(metadata_file):
            print(f"Warning: metadata file not found at {metadata_file}")
            return
        
        # Initialize and create student database
        student_db = StudentDatabase(database_dir)
        students = student_db.create_database(database_dir, metadata_file)
        
        if not students:
            print("No student data found. Please add students first.")
            return
        
        # Initialize attendance system
        attendance_system = AttendanceSystem(
            student_database_path=metadata_file,  # Using metadata file as database path
            model_path=None  # No model path needed during training
        )
        
        # Update this line with your actual video paths
        video_paths = [
            'data/training_videos/video1.mp4',
            'data/training_videos/video2.mp4'
        ]
        
        # Prepare training data
        sequences, labels = prepare_training_data(video_paths, student_db, attendance_system)
        
        # Create dataset and dataloaders
        dataset = FaceSequenceDataset(sequences, labels)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Initialize and train model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = AdaptiveLSTM(
            input_size=128,  # FaceNet embedding size
            hidden_size=256,
            num_classes=len(student_db.student_ids)
        ).to(device)
        
        train_model(model, train_loader, val_loader, num_epochs=50, device=device)
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
