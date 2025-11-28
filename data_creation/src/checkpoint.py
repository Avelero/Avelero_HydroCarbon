"""
Checkpoint Manager for resumable dataset generation
"""

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class CheckpointManager:
    """Thread-safe checkpoint manager for resumable generation"""

    def __init__(self, session_id: str, output_file: str, generation_config: dict):
        """
        Initialize checkpoint manager.

        Args:
            session_id: Unique session identifier
            output_file: Path to output CSV file
            generation_config: Configuration dict with generation parameters
        """
        self.session_id = session_id
        checkpoint_dir = generation_config.get('checkpoint_dir', 'output/checkpoints')
        self.checkpoint_file = Path(checkpoint_dir) / f"checkpoint_{session_id}.json"
        self.output_file = output_file
        self.generation_config = generation_config
        self.state = self._init_state()
        self.lock = threading.Lock()  # Thread safety
    
    def _init_state(self) -> dict:
        """Initialize checkpoint state"""
        return {
            "session_id": self.session_id,
            "started_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "output_file": self.output_file,
            "config": self.generation_config,
            "progress": {
                "completed_chunks": [],  # List of {"category_index": X, "chunk_index": Y}
                "current_category_index": 0,
                "current_chunk_index": 0,
                "total_products_generated": 0,
                "failed_chunks": []  # List of {"category_index": X, "chunk_index": Y, "error": "..."}
            },
            "status": "running"
        }
    
    def save(self):
        """Save checkpoint to disk with atomic write - thread-safe"""
        with self.lock:
            self.state["last_updated"] = datetime.now().isoformat()

            # Ensure checkpoint dir exists
            Path(self.checkpoint_file).parent.mkdir(parents=True, exist_ok=True)

            # Atomic write (temp file + rename)
            temp_file = str(self.checkpoint_file) + ".tmp"
            with open(temp_file, 'w') as f:
                json.dump(self.state, f, indent=2)
            os.replace(temp_file, self.checkpoint_file)
    
    def mark_chunk_complete(self, category_index: int, chunk_index: int, products_count: int):
        """
        Mark a chunk as completed.

        Args:
            category_index: Index of category
            chunk_index: Index of chunk within category
            products_count: Number of products in chunk
        """
        chunk_id = {"category_index": category_index, "chunk_index": chunk_index}
        if not self.is_chunk_completed(category_index, chunk_index):
            self.state["progress"]["completed_chunks"].append(chunk_id)
        self.state["progress"]["total_products_generated"] += products_count
        self.state["progress"]["current_category_index"] = category_index
        self.state["progress"]["current_chunk_index"] = chunk_index
        self.save()

    def mark_chunk_failed(self, category_index: int, chunk_index: int, error_message: str = ""):
        """
        Mark a chunk as failed.

        Args:
            category_index: Index of category
            chunk_index: Index of chunk within category
            error_message: Optional error description
        """
        failure_info = {
            "category_index": category_index,
            "chunk_index": chunk_index,
            "timestamp": datetime.now().isoformat(),
            "error": error_message
        }
        self.state["progress"]["failed_chunks"].append(failure_info)
        self.save()

    def is_chunk_completed(self, category_index: int, chunk_index: int) -> bool:
        """
        Check if chunk already completed.

        Args:
            category_index: Index of category
            chunk_index: Index of chunk

        Returns:
            True if chunk was completed
        """
        return any(
            c["category_index"] == category_index and c["chunk_index"] == chunk_index
            for c in self.state["progress"]["completed_chunks"]
        )
    
    def mark_complete(self):
        """Mark entire generation as complete"""
        self.state["status"] = "complete"
        self.state["completed_at"] = datetime.now().isoformat()
        self.save()
    
    def cleanup(self):
        """Remove checkpoint file after successful completion"""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            print(f"  Cleaned up checkpoint: {self.checkpoint_file}")
    
    def get_progress_summary(self) -> str:
        """Get human-readable progress summary"""
        completed = len(self.state["progress"]["completed_chunks"])
        total = self.generation_config.get("total_chunks", 0)
        products = self.state["progress"]["total_products_generated"]
        failed = len(self.state["progress"]["failed_chunks"])

        summary = f"Progress: {completed}/{total} chunks ({products} products)"
        if failed > 0:
            summary += f", {failed} failed"
        return summary
    
    @classmethod
    def load_existing(cls, checkpoint_file: str) -> Optional['CheckpointManager']:
        """
        Load existing checkpoint from file.
        
        Args:
            checkpoint_file: Path to checkpoint JSON file
            
        Returns:
            CheckpointManager instance or None if file doesn't exist
        """
        path = Path(checkpoint_file)
        if not path.exists():
            return None
        
        try:
            with open(path, 'r') as f:
                state = json.load(f)
            
            manager = cls.__new__(cls)
            manager.session_id = state["session_id"]
            manager.checkpoint_file = path
            manager.output_file = state["output_file"]
            manager.generation_config = state["config"]
            manager.state = state
            manager.lock = threading.Lock()  # Initialize lock for thread safety

            return manager
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Failed to load checkpoint {checkpoint_file}: {e}")
            return None
    
    @classmethod
    def find_latest_checkpoint(cls, checkpoint_dir: str) -> Optional[str]:
        """
        Find most recent checkpoint file.

        Args:
            checkpoint_dir: Directory to search for checkpoints

        Returns:
            Path to latest checkpoint or None
        """
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            return None

        checkpoints = list(checkpoint_path.glob("checkpoint_*.json"))
        if not checkpoints:
            return None

        # Sort by modification time, return most recent
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return str(latest)

    @classmethod
    def find_checkpoint_by_output(cls, checkpoint_dir: str, output_file: str) -> Optional[str]:
        """
        Find checkpoint matching a specific output file.

        Args:
            checkpoint_dir: Directory to search for checkpoints
            output_file: Output file path to match

        Returns:
            Path to matching checkpoint or None
        """
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            return None

        checkpoints = list(checkpoint_path.glob("checkpoint_*.json"))
        if not checkpoints:
            return None

        # Find checkpoint with matching output_file
        for checkpoint_file in checkpoints:
            try:
                with open(checkpoint_file, 'r') as f:
                    state = json.load(f)
                    if state.get("output_file") == output_file:
                        return str(checkpoint_file)
            except (json.JSONDecodeError, KeyError, IOError):
                continue

        return None
