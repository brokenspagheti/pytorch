"""Progress-enabled DataLoader wrapper.

This module provides a DataLoader wrapper that displays progress information
during iteration without requiring external dependencies.
"""

from __future__ import annotations

import sys
from typing import Any

from torch.utils.data import DataLoader

__all__ = ["ProgressDataLoader"]


class ProgressDataLoader(DataLoader):
    """DataLoader with built-in progress bar support.
    
    This class extends the standard DataLoader to provide optional progress
    display during iteration. It shows the current batch number, total batches,
    and completion percentage.
    
    Args:
        dataset: Dataset from which to load the data.
        batch_size: How many samples per batch to load (default: 1).
        show_progress: Whether to display progress bar (default: False).
        progress_desc: Description to show before progress bar (default: "Loading").
        **kwargs: Additional arguments passed to DataLoader.
    
    Example:
        >>> from torch.utils.data import ProgressDataLoader, TensorDataset
        >>> import torch
        >>> 
        >>> dataset = TensorDataset(torch.randn(100, 10))
        >>> loader = ProgressDataLoader(dataset, batch_size=10, show_progress=True)
        >>> 
        >>> for batch in loader:
        >>>     # Your training code here
        >>>     pass
        >>> # Output: Loading: 100.0% [10/10]
    
    Note:
        The progress bar is printed to stderr to avoid interfering with
        stdout output. Progress is only shown when show_progress=True.
    """
    
    def __init__(
        self,
        dataset,
        batch_size: int | None = 1,
        show_progress: bool = False,
        progress_desc: str = "Loading",
        **kwargs: Any
    ) -> None:
        """Initialize ProgressDataLoader.
        
        Args:
            dataset: Dataset to load from
            batch_size: Batch size (default: 1)
            show_progress: Enable progress display (default: False)
            progress_desc: Progress bar description (default: "Loading")
            **kwargs: Additional DataLoader arguments
        """
        super().__init__(dataset, batch_size=batch_size, **kwargs)
        self.show_progress = show_progress
        self.progress_desc = progress_desc
        self._total_batches: int | None = None
    
    def __iter__(self):
        """Iterate over batches with optional progress display."""
        if not self.show_progress:
            # No progress bar - use standard iteration
            yield from super().__iter__()
            return
        
        # Calculate total batches
        try:
            self._total_batches = len(self)
        except TypeError:
            # IterableDataset without __len__
            self._total_batches = None
        
        # Iterate with progress display
        batch_idx = 0
        for batch in super().__iter__():
            batch_idx += 1
            self._display_progress(batch_idx)
            yield batch
        
        # Print newline after completion
        if self._total_batches is not None:
            sys.stderr.write('\n')
            sys.stderr.flush()
    
    def _display_progress(self, current: int) -> None:
        """Display progress bar.
        
        Args:
            current: Current batch number (1-indexed)
        """
        if self._total_batches is None:
            # Unknown total - just show count
            msg = f"\r{self.progress_desc}: [{current} batches]"
        else:
            # Known total - show percentage and count
            percent = (current / self._total_batches) * 100
            bar_length = 30
            filled = int(bar_length * current / self._total_batches)
            bar = '=' * filled + '>' + ' ' * (bar_length - filled - 1)
            
            msg = (
                f"\r{self.progress_desc}: {percent:5.1f}% "
                f"[{bar}] {current}/{self._total_batches}"
            )
        
        sys.stderr.write(msg)
        sys.stderr.flush()
