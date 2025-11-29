"""
Incremental CSV Writer for batch-by-batch CSV writing
"""

import os
import sys
import pandas as pd
from pathlib import Path
import threading

# Add path for config
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'config'))

import config


class IncrementalCSVWriter:
    """Thread-safe CSV writer that writes incrementally, batch by batch"""

    def __init__(self, output_file: str):
        """
        Initialize incremental CSV writer.

        Args:
            output_file: Path to output CSV file
        """
        self.output_file = Path(output_file)
        self.header_written = False
        self.lock = threading.Lock()  # Thread safety
    
    def write_header(self):
        """Write CSV header (call once at start) - thread-safe"""
        with self.lock:
            if not self.header_written:
                self.output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.output_file, 'w') as f:
                    f.write(config.CSV_HEADER + "\n")
                self.header_written = True
                print(f" Created CSV file: {self.output_file}")

    def append_batch(self, batch_df: pd.DataFrame):
        """
        Append a batch to the CSV file - thread-safe.

        Args:
            batch_df: DataFrame with batch data
        """
        with self.lock:
            if not self.header_written:
                self.write_header()

            # Append without header, with proper quoting for fields containing commas
            batch_df.to_csv(
                self.output_file,
                mode='a',
                header=False,
                index=False,
                quoting=1,  # QUOTE_ALL - quote all fields
                escapechar='\\'
            )
    
    def resume_from_existing(self):
        """Resume writing to existing file (sets header as written)"""
        if self.output_file.exists():
            self.header_written = True
            print(f" Resuming with existing CSV: {self.output_file}")
        else:
            self.write_header()
    
    def get_row_count(self) -> int:
        """
        Get number of data rows in CSV (excluding header).
        
        Returns:
            Number of rows
        """
        if not self.output_file.exists():
            return 0
        
        try:
            df = pd.read_csv(self.output_file)
            return len(df)
        except Exception:
            return 0
