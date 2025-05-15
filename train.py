import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import sys

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

def print_model_summary(model):
    """Print summary of model architecture and parameters"""
    print("\n" + "="*50)
    print("ADAPTIVE LSTM MODEL SUMMARY")
    print("="*50)
    
    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    # Print model structure
    print(f"\nModel Structure:")
    print(f"{str(model)}")
    
    # Print parameter details for each layer
    print("\nLayer Details:")
    for name, module in model.named_children():
        print(f"\n{name.upper()} LAYER:")
        
        # Special handling for LSTM cell to show adaptive gate structure
        if name == 'lstm_cell':
            print("  Adaptive Gate Structure:")
            print(f"  - Input Gate:  Linear({module.input_gate.in_features} -> {module.input_gate.out_features}) with clarity score")
            print(f"  - Forget Gate: Linear({module.forget_gate.in_features} -> {module.forget_gate.out_features}) with clarity score")
            print(f"  - Output Gate: Linear({module.output_gate.in_features} -> {module.output_gate.out_features}) with clarity score")
            print(f"  - Cell Gate:   Linear({module.cell_gate.in_features} -> {module.cell_gate.out_features}) standard")
        else:
            for param_name, param in module.named_parameters():
                print(f"  - {param_name}: {tuple(param.size())}")
    
    # Print general info
    print("\nGeneral Information:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    print("="*50)

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
        
        # Use MongoDB data directly instead of trying to load from files
        print("Using student data directly from MongoDB...")
        
        # Skip the file-based student database and use MongoDB data
        student_ids = [student.sap_id for student in students]
        print(f"Student IDs for training: {student_ids}")
        
        # Initialize attendance system with MongoDB data
        attendance_system = AttendanceSystem()
        
        # Check if we have a video path to process
        video_paths = [
            '/Users/harshkanani/Desktop/ipd_2/data/training_videos/video.mp4'  # Default path
        ]
        
        # Check if video exists
        for path in video_paths:
            if not os.path.exists(path):
                print(f"Warning: Training video not found at {path}")
                print("Please provide a valid training video path.")
                
                # Ask user if they want to provide a video path
                use_custom = input("Do you want to specify a video path? (y/n): ")
                if use_custom.lower() == 'y':
                    custom_path = input("Enter video path: ")
                    if os.path.exists(custom_path):
                        video_paths = [custom_path]
                        break
                    else:
                        print(f"Error: Video not found at {custom_path}")
                        return
                else:
                    print("Using webcam for training data collection.")
                    video_paths = [0]  # Use webcam
        
        # Create a simple dataset from MongoDB
        print("Creating dataset from student embeddings...")
        sequences = []
        labels = []
        
        # Convert MongoDB data to training data
        for idx, student in enumerate(students):
            if hasattr(student, 'face_embeddings') and student.face_embeddings:
                # Convert each student's embeddings into sequences
                embeddings = student.face_embeddings
                
                # If we have enough embeddings, create sequences
                if len(embeddings) >= 5:
                    # Create clarity scores (default to 1.0 for existing embeddings)
                    clarity_scores = [1.0] * len(embeddings)
                    
                    # Create a sequence for each consecutive set of 5 embeddings
                    for i in range(len(embeddings) - 4):
                        seq = embeddings[i:i+5]
                        clar = clarity_scores[i:i+5]
                        sequences.append({'embeddings': seq, 'clarity_scores': clar})
                        labels.append(idx)
        
        if not sequences:
            print("Not enough embedding data for training. Please capture more images.")
            return
            
        print(f"Created {len(sequences)} training sequences")
        
        # Create dataset and dataloaders
        dataset = FaceSequenceDataset(sequences, labels)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=min(32, len(train_dataset)), shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=min(32, len(val_dataset)), shuffle=False)
        
        # Initialize and train model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = AdaptiveLSTM(
            input_size=len(students[0].face_embeddings[0]) if students and students[0].face_embeddings else 512,
            hidden_size=256,
            num_classes=len(students)
        ).to(device)
        
        # Create models/trained directory if it doesn't exist
        os.makedirs('models/trained', exist_ok=True)
        
        # Train model
        train_model(model, train_loader, val_loader, num_epochs=100, device=device)
        
        # Print model summary
        print_model_summary(model)
        
        # Save final model
        torch.save(model.state_dict(), 'models/trained/best_model.pth')
        print("Model training completed and saved to models/trained/best_model.pth")
        
        # Optionally save a visualization of the model
        try:
            from torchviz import make_dot
            
            # Create sample input for visualization
            sample_input = torch.randn(1, 5, model.lstm_cell.input_size)
            sample_clarity = torch.ones(1, 5)
            
            # Generate visualization
            dot = make_dot(model(sample_input, sample_clarity), 
                          params=dict(model.named_parameters()),
                          show_attrs=True, show_saved=True)
            
            # Save visualization
            dot.format = 'png'
            dot.render(filename='models/trained/model_visualization')
            print("Model visualization saved to models/trained/model_visualization.png")
        except ImportError:
            print("Note: Install torchviz package for model visualization: pip install torchviz")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
