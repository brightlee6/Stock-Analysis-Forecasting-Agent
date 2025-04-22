import unittest
import pandas as pd
import numpy as np
from models.stock_data import StockData
from models.stock_model_holdout import StockModelHoldout
import os

class TestStockModelHoldout(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        self.stock_data = StockData("AAPL", "2023-01-01", "2023-12-31")
        self.stock_data.fetch_closing_prices()
        self.model = StockModelHoldout(self.stock_data)
        
    def test_initialization(self):
        """Test StockModelHoldout initialization."""
        self.assertEqual(self.model.stock_data, self.stock_data)
        self.assertIsNone(self.model.train_data)
        self.assertIsNone(self.model.test_data)
        self.assertIsNone(self.model.model)
        self.assertIsNone(self.model.forecast)
        self.assertIsNone(self.model.metrics)
        
    def test_invalid_input(self):
        """Test handling of invalid input."""
        with self.assertRaises(ValueError):
            StockModelHoldout("invalid_input")
            
    def test_split_data(self):
        """Test data splitting."""
        test_size = 0.2
        self.model.split_data(test_size)
        
        # Check if data is properly split
        self.assertIsNotNone(self.model.train_data)
        self.assertIsNotNone(self.model.test_data)
        self.assertTrue(len(self.model.train_data) > 0)
        self.assertTrue(len(self.model.test_data) > 0)
        
        # Check column names after renaming
        self.assertTrue('ds' in self.model.train_data.columns)
        self.assertTrue('y' in self.model.train_data.columns)
        self.assertTrue('ds' in self.model.test_data.columns)
        self.assertTrue('y' in self.model.test_data.columns)
        
    def test_train_model(self):
        """Test model training."""
        self.model.split_data()
        self.model.train_model()
        self.assertIsNotNone(self.model.model)
        
    def test_make_forecast(self):
        """Test making forecasts."""
        self.model.split_data()
        self.model.train_model()
        self.model.make_forecast()
        self.assertIsNotNone(self.model.forecast)
        
    def test_calculate_metrics(self):
        """Test metric calculation."""
        self.model.split_data()
        self.model.train_model()
        self.model.make_forecast()
        self.model.calculate_metrics()
        
        # Check if metrics are calculated
        self.assertIsNotNone(self.model.metrics)
        self.assertTrue('MAE' in self.model.metrics)
        self.assertTrue('MSE' in self.model.metrics)
        self.assertTrue('RMSE' in self.model.metrics)
        self.assertTrue('R2' in self.model.metrics)
        
        # Check if metrics are valid numbers
        for metric, value in self.model.metrics.items():
            self.assertIsInstance(value, float)
            if metric != 'R2':  # R2 can be negative
                self.assertGreaterEqual(value, 0)
                
    def test_visualize_forecast(self):
        """Test forecast visualization."""
        self.model.split_data()
        self.model.train_model()
        self.model.make_forecast()
        
        # Test visualization with save path
        test_file = "test_forecast.png"
        self.model.visualize_forecast(save_path=test_file)
        self.assertTrue(os.path.exists(test_file))
        os.remove(test_file)  # Clean up
        
    def test_run_analysis(self):
        """Test complete analysis pipeline."""
        metrics = self.model.run_analysis()
        
        # Check if all metrics are present
        self.assertIsNotNone(metrics)
        self.assertTrue('MAE' in metrics)
        self.assertTrue('MSE' in metrics)
        self.assertTrue('RMSE' in metrics)
        self.assertTrue('R2' in metrics)

if __name__ == '__main__':
    unittest.main() 