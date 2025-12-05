"""
Progress bar utility for displaying progress in pipeline steps.
"""

import sys
import time
import io

# Set stdout encoding to UTF-8 to handle Unicode characters on Windows
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except AttributeError:
        # If stdout doesn't have a buffer, try to reconfigure it
        pass


class ProgressBar:
    """
    A reusable progress bar class for displaying progress in pipeline steps.
    
    Usage:
        # Simple usage
        progress = ProgressBar(total=100, prefix="Processing", suffix="items")
        for i in range(100):
            # do work
            progress.update(1)
        
        # With elapsed time tracking
        progress = ProgressBar(total=100, prefix="Processing", suffix="items", show_elapsed=True)
        for i in range(100):
            # do work
            progress.update(1)
        
        # Context manager usage (auto-updates)
        with ProgressBar(total=100, prefix="Processing") as progress:
            for i in range(100):
                # do work
                progress.update(1)
    """
    
    def __init__(self, total: int, prefix: str = "", suffix: str = "", 
                 decimals: int = 1, length: int = 30, fill: str = "█", 
                 show_elapsed: bool = True):
        """
        Initialize a progress bar.
        
        Args:
            total: Total number of iterations
            prefix: Text to display before the progress bar
            suffix: Text to display after the progress bar
            decimals: Number of decimal places in percentage
            length: Length of the progress bar in characters
            fill: Character to use for filled portion
            show_elapsed: Whether to show elapsed time
        """
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.length = length
        self.fill = fill
        self.show_elapsed = show_elapsed
        self.current = 0
        self.start_time = time.time() if show_elapsed else None
        
    def update(self, increment: int = 1):
        """Update the progress bar by increment."""
        self.current = min(self.current + increment, self.total)
        self._render()
        
    def set_current(self, value: int, extra_info: str = ""):
        """Set the current progress value directly.
        
        Args:
            value: Current progress value
            extra_info: Optional additional information to display after the progress bar
        """
        self.current = min(max(0, value), self.total)
        self._render(extra_info=extra_info)
        
    def _render(self, extra_info: str = ""):
        """Render the progress bar to stdout.
        
        Args:
            extra_info: Optional additional information to display after the progress bar
        """
        if self.total == 0:
            return
            
        percent = f"{100 * (self.current / float(self.total)):.{self.decimals}f}"
        
        # Format elapsed time as HH:MM:SS
        time_str = ""
        if self.show_elapsed and self.start_time is not None:
            elapsed_time = time.time() - self.start_time
            hours = int(elapsed_time // 3600)
            minutes = int((elapsed_time % 3600) // 60)
            seconds = int(elapsed_time % 60)
            time_str = f"  Time_elapsed : {hours:02d}:{minutes:02d}:{seconds:02d}"
        
        filled_length = int(self.length * self.current // self.total)
        bar = self.fill * filled_length + "░" * (self.length - filled_length)
        
        # Add extra_info if provided (with separator)
        extra_display = f"  | {extra_info}" if extra_info else ""
        
        # Encode safely to handle Unicode characters
        try:
            output = f"\r{self.prefix} |{bar}| {percent}% {self.suffix}{time_str}{extra_display}"
            sys.stdout.write(output)
            sys.stdout.flush()
        except UnicodeEncodeError:
            # Fallback: replace problematic characters
            output = f"\r{self.prefix} |{bar}| {percent}% {self.suffix}{time_str}{extra_display}".encode('ascii', 'replace').decode('ascii')
            sys.stdout.write(output)
            sys.stdout.flush()
        
        if self.current >= self.total:
            sys.stdout.write("\n")
            
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure progress bar is complete."""
        if self.current < self.total:
            self.set_current(self.total)
        return False

