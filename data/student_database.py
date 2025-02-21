import os
import json
import torch
import numpy as np
import cv2
from models.feature_extractor import FeatureExtractor
from models.student import Student

class StudentDatabase:
    def __init__(self, database_dir):
        self.feature_extractor = FeatureExtractor()
        self.students = {}  # sap_id -> Student object
        self.embeddings = []  # List of face embeddings
        self.sap_ids = []  # List of SAP IDs in order
        
    def create_database(self, database_dir, metadata_file):
        """
        Create database from directory of student images and metadata
        
        database_dir structure:
        - SAP_ID1/
            - image1.jpg
            - image2.jpg
        - SAP_ID2/
            - image1.jpg
            ...
            
        metadata_file: JSON file containing student details
        {
            "SAP_ID1": {
                "name": "Student Name",
                "year": 2,
                "branch": "Computer Engineering",
                "gmail": "student@email.com"
            },
            ...
        }
        """
        # Load student metadata
        with open(metadata_file, 'r') as f:
            student_metadata = json.load(f)
        
        for sap_id in os.listdir(database_dir):
            if sap_id in student_metadata:
                student_path = os.path.join(database_dir, sap_id)
                if os.path.isdir(student_path):
                    # Collect all face embeddings for this student
                    embeddings = []
                    for img_name in os.listdir(student_path):
                        if img_name.endswith(('.jpg', '.png', '.jpeg')):
                            img_path = os.path.join(student_path, img_name)
                            img = cv2.imread(img_path)
                            embedding = self.feature_extractor.extract_features(img)
                            embeddings.append(embedding)
                    
                    if embeddings:
                        # Create student object with mean embedding
                        mean_embedding = np.mean(embeddings, axis=0)
                        metadata = student_metadata[sap_id]
                        
                        student = Student(
                            sap_id=sap_id,
                            name=metadata['name'],
                            year=metadata['year'],
                            branch=metadata['branch'],
                            gmail=metadata['gmail'],
                            face_embedding=mean_embedding
                        )
                        
                        self.students[sap_id] = student
                        self.sap_ids.append(sap_id)
                        self.embeddings.append(mean_embedding)
        
        self.embeddings = np.array(self.embeddings)
        return self.students
    
    def save_database(self, output_path):
        # Save student data and embeddings
        data = {
            'students': {sap_id: student.to_dict() for sap_id, student in self.students.items()},
            'sap_ids': self.sap_ids,
            'embeddings': self.embeddings
        }
        np.save(output_path, data)
    
    def load_database(self, input_path):
        data = np.load(input_path, allow_pickle=True).item()
        self.sap_ids = data['sap_ids']
        self.embeddings = data['embeddings']
        self.students = {
            sap_id: Student.from_dict(student_data) 
            for sap_id, student_data in data['students'].items()
        }
        return self.students
    
    def get_student_by_index(self, index):
        """Get student by index in the embeddings array"""
        sap_id = self.sap_ids[index]
        return self.students[sap_id]
