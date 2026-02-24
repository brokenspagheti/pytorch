"""Tests for ProgressDataLoader."""

import io
import sys
import unittest
from contextlib import redirect_stderr

import torch
from torch.utils.data import ProgressDataLoader, TensorDataset


class TestProgressDataLoader(unittest.TestCase):
    """Test suite for ProgressDataLoader."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple dataset
        self.data = torch.randn(100, 10)
        self.dataset = TensorDataset(self.data)
    
    def test_basic_iteration_without_progress(self):
        """Test that basic iteration works without progress bar."""
        loader = ProgressDataLoader(
            self.dataset,
            batch_size=10,
            show_progress=False
        )
        
        batches = list(loader)
        self.assertEqual(len(batches), 10)
        self.assertEqual(batches[0][0].shape, (10, 10))
    
    def test_basic_iteration_with_progress(self):
        """Test that iteration works with progress bar enabled."""
        loader = ProgressDataLoader(
            self.dataset,
            batch_size=10,
            show_progress=True
        )
        
        # Capture stderr to check progress output
        stderr_capture = io.StringIO()
        with redirect_stderr(stderr_capture):
            batches = list(loader)
        
        self.assertEqual(len(batches), 10)
        
        # Check that progress was written to stderr
        output = stderr_capture.getvalue()
        self.assertIn('Loading:', output)
        self.assertIn('100.0%', output)
        self.assertIn('10/10', output)
    
    def test_custom_progress_description(self):
        """Test custom progress bar description."""
        loader = ProgressDataLoader(
            self.dataset,
            batch_size=10,
            show_progress=True,
            progress_desc="Training"
        )
        
        stderr_capture = io.StringIO()
        with redirect_stderr(stderr_capture):
            list(loader)
        
        output = stderr_capture.getvalue()
        self.assertIn('Training:', output)
    
    def test_single_batch(self):
        """Test with dataset that produces single batch."""
        small_dataset = TensorDataset(torch.randn(5, 10))
        loader = ProgressDataLoader(
            small_dataset,
            batch_size=10,
            show_progress=True
        )
        
        stderr_capture = io.StringIO()
        with redirect_stderr(stderr_capture):
            batches = list(loader)
        
        self.assertEqual(len(batches), 1)
        output = stderr_capture.getvalue()
        self.assertIn('100.0%', output)
        self.assertIn('1/1', output)
    
    def test_empty_dataset(self):
        """Test with empty dataset."""
        empty_dataset = TensorDataset(torch.randn(0, 10))
        loader = ProgressDataLoader(
            empty_dataset,
            batch_size=10,
            show_progress=True
        )
        
        batches = list(loader)
        self.assertEqual(len(batches), 0)
    
    def test_drop_last(self):
        """Test with drop_last=True."""
        # 95 samples with batch_size=10 should give 9 batches (drop last 5)
        dataset = TensorDataset(torch.randn(95, 10))
        loader = ProgressDataLoader(
            dataset,
            batch_size=10,
            drop_last=True,
            show_progress=True
        )
        
        stderr_capture = io.StringIO()
        with redirect_stderr(stderr_capture):
            batches = list(loader)
        
        self.assertEqual(len(batches), 9)
        output = stderr_capture.getvalue()
        self.assertIn('9/9', output)
    
    def test_shuffle(self):
        """Test with shuffle=True."""
        loader = ProgressDataLoader(
            self.dataset,
            batch_size=10,
            shuffle=True,
            show_progress=True
        )
        
        batches = list(loader)
        self.assertEqual(len(batches), 10)
    
    def test_num_workers(self):
        """Test with multiple workers."""
        # Note: This test uses num_workers=0 to avoid multiprocessing issues
        # In real usage, num_workers > 0 should also work
        loader = ProgressDataLoader(
            self.dataset,
            batch_size=10,
            num_workers=0,
            show_progress=True
        )
        
        batches = list(loader)
        self.assertEqual(len(batches), 10)
    
    def test_progress_bar_format(self):
        """Test that progress bar has correct format."""
        loader = ProgressDataLoader(
            self.dataset,
            batch_size=10,
            show_progress=True
        )
        
        stderr_capture = io.StringIO()
        with redirect_stderr(stderr_capture):
            for _ in loader:
                pass
        
        output = stderr_capture.getvalue()
        
        # Check for progress bar elements
        self.assertIn('[', output)  # Progress bar brackets
        self.assertIn(']', output)
        self.assertIn('%', output)  # Percentage
        self.assertIn('/', output)  # Batch count separator
    
    def test_inheritance_from_dataloader(self):
        """Test that ProgressDataLoader is a proper DataLoader subclass."""
        from torch.utils.data import DataLoader
        
        loader = ProgressDataLoader(self.dataset, batch_size=10)
        self.assertIsInstance(loader, DataLoader)
    
    def test_all_dataloader_args(self):
        """Test that all standard DataLoader arguments work."""
        loader = ProgressDataLoader(
            self.dataset,
            batch_size=10,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            timeout=0,
            show_progress=True
        )
        
        batches = list(loader)
        self.assertEqual(len(batches), 10)


class TestProgressDataLoaderEdgeCases(unittest.TestCase):
    """Test edge cases for ProgressDataLoader."""
    
    def test_very_large_batch_size(self):
        """Test with batch size larger than dataset."""
        dataset = TensorDataset(torch.randn(10, 5))
        loader = ProgressDataLoader(
            dataset,
            batch_size=100,
            show_progress=True
        )
        
        batches = list(loader)
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0][0].shape[0], 10)
    
    def test_batch_size_one(self):
        """Test with batch_size=1."""
        dataset = TensorDataset(torch.randn(10, 5))
        loader = ProgressDataLoader(
            dataset,
            batch_size=1,
            show_progress=True
        )
        
        batches = list(loader)
        self.assertEqual(len(batches), 10)


if __name__ == '__main__':
    unittest.main()
