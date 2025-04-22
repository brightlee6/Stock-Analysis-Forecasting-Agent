import unittest
from datetime import datetime, timedelta
import pandas as pd
import os
from models.stock_data import StockData

class TestStockData(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        self.ticker = "AAPL"
        self.start_date = "2023-01-01"
        self.end_date = "2023-12-31"
        self.stock_data = StockData(self.ticker, self.start_date, self.end_date)
        
    def test_initialization(self):
        """Test StockData initialization."""
        self.assertEqual(self.stock_data.ticker, self.ticker)
        self.assertEqual(self.stock_data.start_date, self.start_date)
        self.assertEqual(self.stock_data.end_date, self.end_date)
        self.assertIsNone(self.stock_data.dataframe)
        
    def test_fetch_closing_prices(self):
        """Test fetching closing prices."""
        self.stock_data.fetch_closing_prices()
        self.assertIsNotNone(self.stock_data.dataframe)
        self.assertIsInstance(self.stock_data.dataframe, pd.DataFrame)
        self.assertTrue('Date' in self.stock_data.dataframe.columns)
        self.assertTrue('Close' in self.stock_data.dataframe.columns)
        
    def test_save_to_csv(self):
        """Test saving data to CSV."""
        self.stock_data.fetch_closing_prices()
        test_file = "test_stock_data.csv"
        self.stock_data.save_to_csv(test_file)
        self.assertTrue(os.path.exists(test_file))
        os.remove(test_file)  # Clean up
        
    def test_visualize_data(self):
        """Test data visualization."""
        self.stock_data.fetch_closing_prices()
        test_file = "test_visualization.png"
        self.stock_data.visualize_data(save_path=test_file)
        self.assertTrue(os.path.exists(test_file))
        os.remove(test_file)  # Clean up
        
    def test_invalid_ticker(self):
        """Test handling of invalid ticker."""
        with self.assertRaises(ValueError) as context:
            StockData("INVALID_TICKER", self.start_date, self.end_date)
        self.assertIn("Invalid ticker symbol", str(context.exception))
            
    def test_invalid_dates(self):
        """Test handling of invalid dates."""
        with self.assertRaises(ValueError) as context:
            StockData(self.ticker, "invalid-date", "invalid-date")
        self.assertIn("Invalid date format", str(context.exception))

if __name__ == '__main__':
    unittest.main() 