import os
from datetime import datetime
from typing import List, Optional
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv
from .models import StudentModel, AttendanceRecord

load_dotenv()

class DatabaseManager:
    def __init__(self):
        mongodb_url = os.getenv('MONGODB_URL', 'mongodb://localhost:27017')
        self.client = MongoClient(mongodb_url)
        self.db = self.client['face_attendance_db']
        
    def add_student(self, student_data: dict, face_embeddings: List[np.ndarray]) -> str:
        """Add a new student to the database"""
        collection = self.db.students
        
        # Check if student already exists
        existing_student = collection.find_one({"sap_id": student_data['sap_id']})
        if existing_student:
            # Update existing student
            embeddings_list = [[float(x) for x in emb.flatten()] for emb in face_embeddings]
            collection.update_one(
                {"sap_id": student_data['sap_id']},
                {
                    "$set": {
                        "name": student_data['name'],
                        "year": student_data['year'],
                        "branch": student_data['branch'],
                        "gmail": student_data['gmail'],
                        "face_embeddings": embeddings_list,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            return str(existing_student['_id'])
        
        # Create new student
        embeddings_list = [[float(x) for x in emb.flatten()] for emb in face_embeddings]
        
        student = StudentModel(
            sap_id=student_data['sap_id'],
            name=student_data['name'],
            year=student_data['year'],
            branch=student_data['branch'],
            gmail=student_data['gmail'],
            face_embeddings=embeddings_list
        )
        
        result = collection.insert_one(student.dict(by_alias=True, exclude={'id'}))  # Exclude id field
        return str(result.inserted_id)
    
    def get_all_students(self) -> List[StudentModel]:
        """Retrieve all students"""
        collection = self.db.students
        students = []
        for doc in collection.find():
            # Convert ObjectId to string and create StudentModel
            if '_id' in doc:
                doc['_id'] = str(doc['_id'])
            try:
                student = StudentModel(**doc)
                students.append(student)
            except Exception as e:
                print(f"Error processing student document: {e}")
                continue
        return students
    
    def get_student_by_sap(self, sap_id: str) -> Optional[StudentModel]:
        """Retrieve a student by SAP ID"""
        collection = self.db.students
        doc = collection.find_one({"sap_id": sap_id})
        if doc:
            if '_id' in doc:
                doc['_id'] = str(doc['_id'])
            try:
                return StudentModel(**doc)
            except Exception as e:
                print(f"Error processing student document: {e}")
                return None
        return None
    
    def update_student_embeddings(self, sap_id: str, new_embedding: np.ndarray):
        """Add a new face embedding to existing student"""
        collection = self.db.students
        student = self.get_student_by_sap(sap_id)
        if student:
            embeddings = student.face_embeddings
            embeddings.append(new_embedding.tolist())
            collection.update_one(
                {"sap_id": sap_id},
                {
                    "$set": {
                        "face_embeddings": embeddings,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
    
    def record_attendance(self, student_id: str, sap_id: str, 
                         confidence: float, subject: str = None,
                         lecture_slot: str = None) -> str:
        """Record a new attendance entry"""
        collection = self.db.attendance
        
        # Create record without _id first
        record_data = {
            'student_id': student_id,
            'sap_id': sap_id,
            'timestamp': datetime.utcnow(),
            'confidence': confidence,
            'subject': subject,
            'lecture_slot': lecture_slot
        }
        
        # Insert into database
        result = collection.insert_one(record_data)
        
        # Update record with the generated _id
        record_data['_id'] = str(result.inserted_id)
        record = AttendanceRecord(**record_data)
        
        return str(result.inserted_id)
    
    def get_student_attendance(self, sap_id: str, 
                             start_date: datetime = None,
                             end_date: datetime = None) -> List[AttendanceRecord]:
        """Get attendance records for a student"""
        collection = self.db.attendance
        query = {"sap_id": sap_id}
        
        if start_date and end_date:
            query["timestamp"] = {
                "$gte": start_date,
                "$lte": end_date
            }
        
        records = collection.find(query).sort("timestamp", -1)
        return [AttendanceRecord(**record) for record in records]
    
    def get_class_attendance(self, branch: str, year: int,
                           date: datetime = None) -> List[AttendanceRecord]:
        """Get attendance records for a specific class"""
        students = self.db.students.find({"branch": branch, "year": year})
        sap_ids = [student["sap_id"] for student in students]
        
        query = {"sap_id": {"$in": sap_ids}}
        if date:
            start = datetime(date.year, date.month, date.day)
            end = datetime(date.year, date.month, date.day, 23, 59, 59)
            query["timestamp"] = {"$gte": start, "$lte": end}
        
        records = self.db.attendance.find(query)
        return [AttendanceRecord(**record) for record in records]
