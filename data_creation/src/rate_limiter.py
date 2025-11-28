"""
Thread-safe rate limiter for API calls
"""

import threading
import time
from typing import Optional


class TokenBucketRateLimiter:
    """
    Thread-safe token bucket rate limiter.

    Ensures we never exceed the specified requests per minute (RPM).
    """

    def __init__(self, max_rpm: int, bucket_size: Optional[int] = None):
        """
        Initialize rate limiter.

        Args:
            max_rpm: Maximum requests per minute
            bucket_size: Maximum burst size (defaults to max_rpm)
        """
        self.max_rpm = max_rpm
        self.bucket_size = bucket_size or max_rpm
        self.tokens_per_second = max_rpm / 60.0

        self.tokens = float(self.bucket_size)
        self.last_update = time.time()
        self.lock = threading.Lock()

        # Statistics
        self.total_requests = 0
        self.total_wait_time = 0.0

    def acquire(self, tokens: int = 1) -> float:
        """
        Acquire tokens from the bucket (blocks if needed).

        Args:
            tokens: Number of tokens to acquire

        Returns:
            Time waited in seconds
        """
        with self.lock:
            # Refill bucket based on elapsed time
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(
                self.bucket_size,
                self.tokens + elapsed * self.tokens_per_second
            )
            self.last_update = now

            # Calculate wait time if not enough tokens
            wait_time = 0.0
            if self.tokens < tokens:
                deficit = tokens - self.tokens
                wait_time = deficit / self.tokens_per_second

                # Wait outside the lock
                time.sleep(wait_time)

                # Update after waiting
                now = time.time()
                elapsed = now - self.last_update
                self.tokens = min(
                    self.bucket_size,
                    self.tokens + elapsed * self.tokens_per_second
                )
                self.last_update = now

            # Consume tokens
            self.tokens -= tokens

            # Update stats
            self.total_requests += 1
            self.total_wait_time += wait_time

            return wait_time

    def get_stats(self) -> dict:
        """Get rate limiter statistics"""
        with self.lock:
            return {
                "total_requests": self.total_requests,
                "total_wait_time": self.total_wait_time,
                "current_tokens": self.tokens,
                "max_rpm": self.max_rpm,
            }

    def reset_stats(self):
        """Reset statistics"""
        with self.lock:
            self.total_requests = 0
            self.total_wait_time = 0.0


class RateLimitError(Exception):
    """Exception raised when rate limit is hit"""
    pass
