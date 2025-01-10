import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import sqlite3
from contextlib import contextmanager
import os

class MemoryPersistence:
    """Handles persistence of memory components"""
    
    def __init__(self, db_path: str = "medical_system_memory.db"):
        # Get the current file's directory
        root_dir = Path(__file__).resolve().parent
        
        # Create a memory_db directory if it doesn't exist
        memory_db_dir = root_dir / "memory_db"
        memory_db_dir.mkdir(parents=True, exist_ok=True)
        
        # Construct the full database path
        self.db_path = str(memory_db_dir / db_path)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        try:
            self._initialize_db()
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to initialize database at {self.db_path}: {str(e)}")
    
    def _initialize_db(self):
        """Initialize the database with required tables"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Create patient history table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS patient_history (
                        patient_id TEXT PRIMARY KEY,
                        history_data TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create episodic memory table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS episodic_memory (
                        episode_id TEXT PRIMARY KEY,
                        patient_id TEXT,
                        episode_data TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (patient_id) REFERENCES patient_history (patient_id)
                    )
                ''')
                
                # Create decision history table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS decision_history (
                        decision_id TEXT PRIMARY KEY,
                        episode_id TEXT,
                        decision_data TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (episode_id) REFERENCES episodic_memory (episode_id)
                    )
                ''')
                
                conn.commit()
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to initialize database tables: {str(e)}")
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections"""
        try:
            conn = sqlite3.connect(self.db_path)
            yield conn
        except sqlite3.Error as e:
            raise RuntimeError(f"Database connection error: {str(e)}")
        finally:
            if 'conn' in locals():
                conn.close()
    
    def _serialize_datetime(self, obj):
        """Helper method to serialize datetime objects"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        return obj
    
    def save_patient_history(self, patient_id: str, history_data: Dict[str, Any]):
        """Save or update patient history"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            serialized_data = json.dumps(history_data, default=self._serialize_datetime)
            
            cursor.execute('''
                INSERT OR REPLACE INTO patient_history (patient_id, history_data, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (patient_id, serialized_data))
            
            conn.commit()
    
    def load_patient_history(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Load patient history from database"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT history_data FROM patient_history WHERE patient_id = ?',
                (patient_id,)
            )
            result = cursor.fetchone()
            
            if result:
                return json.loads(result[0])
            return None
    
    def save_episode(self, episode_id: str, patient_id: str, episode_data: Dict[str, Any]):
        """Save an episode to episodic memory"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            serialized_data = json.dumps(episode_data, default=self._serialize_datetime)
            
            cursor.execute('''
                INSERT INTO episodic_memory (episode_id, patient_id, episode_data)
                VALUES (?, ?, ?)
            ''', (episode_id, patient_id, serialized_data))
            
            conn.commit()
    
    def load_episodes(self, patient_id: str) -> list[Dict[str, Any]]:
        """Load all episodes for a patient"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT episode_data FROM episodic_memory WHERE patient_id = ?',
                (patient_id,)
            )
            results = cursor.fetchall()
            
            return [json.loads(row[0]) for row in results]
    
    def save_decision(self, decision_id: str, episode_id: str, decision_data: Dict[str, Any]):
        """Save a decision point"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            serialized_data = json.dumps(decision_data, default=self._serialize_datetime)
            
            cursor.execute('''
                INSERT INTO decision_history (decision_id, episode_id, decision_data)
                VALUES (?, ?, ?)
            ''', (decision_id, episode_id, serialized_data))
            
            conn.commit()
    
    def load_decisions(self, episode_id: str) -> list[Dict[str, Any]]:
        """Load all decisions for an episode"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT decision_data FROM decision_history WHERE episode_id = ?',
                (episode_id,)
            )
            results = cursor.fetchall()
            
            return [json.loads(row[0]) for row in results]
    
    def create_backup(self, backup_path: Optional[str] = None):
        """Create a backup of the memory database"""
        if not backup_path:
            backup_path = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            
        with self._get_connection() as source_conn:
            backup_conn = sqlite3.connect(backup_path)
            source_conn.backup(backup_conn)
            backup_conn.close()
    
    def clear_all_data(self):
        """Clear all data from the database - use with caution!"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM decision_history')
            cursor.execute('DELETE FROM episodic_memory')
            cursor.execute('DELETE FROM patient_history')
            conn.commit()