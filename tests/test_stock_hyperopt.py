import unittest
import pandas as pd
import numpy as np
import os
from models.stock_data import StockData
from models.stock_hyperopt import StockHyperopt

class TestStockHyperopt(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        self.stock_data = StockData("AAPL", "2023-01-01", "2023-12-31")
        self.stock_data.fetch_closing_prices()
        self.model = StockHyperopt(self.stock_data)
        
    def test_initialization(self):
        """Test StockHyperopt initialization."""
        self.assertEqual(self.model.stock_data, self.stock_data)
        self.assertIsNone(self.model.model)
        self.assertIsNone(self.model.best_params)
        self.assertIsNone(self.model.forecast)
        
    def test_invalid_input(self):
        """Test handling of invalid input."""
        with self.assertRaises(ValueError):
            StockHyperopt("invalid_input")
            
    def test_prepare_data(self):
        """Test data preparation."""
        self.model.prepare_data()
        self.assertIsNotNone(self.model.df)
        self.assertTrue('ds' in self.model.df.columns)
        self.assertTrue('y' in self.model.df.columns)
        
    def test_optimize_hyperparameters(self):
        """Test hyperparameter optimization."""
        self.model.prepare_data()
        self.model.optimize_hyperparameters(max_evals=5)  # Use small number for testing
        
        # Check if best parameters are found
        self.assertIsNotNone(self.model.best_params)
        self.assertTrue('changepoint_prior_scale' in self.model.best_params)
        self.assertTrue('seasonality_prior_scale' in self.model.best_params)
        self.assertTrue('holidays_prior_scale' in self.model.best_params)
        self.assertTrue('seasonality_mode' in self.model.best_params)
        
    def test_train_best_model(self):
        """Test training with best parameters."""
        self.model.prepare_data()
        self.model.optimize_hyperparameters(max_evals=5)
        self.model.train_best_model()
        self.assertIsNotNone(self.model.model)
        
    def test_forecast_next_year(self):
        """Test forecasting."""
        self.model.prepare_data()
        self.model.optimize_hyperparameters(max_evals=5)
        self.model.train_best_model()
        self.model.forecast_next_year()
        
        # Check if forecast is created
        self.assertIsNotNone(self.model.forecast)
        self.assertTrue('yhat' in self.model.forecast.columns)
        self.assertTrue('yhat_lower' in self.model.forecast.columns)
        self.assertTrue('yhat_upper' in self.model.forecast.columns)
        
    def test_visualize_forecast(self):
        """Test forecast visualization."""
        self.model.prepare_data()
        self.model.optimize_hyperparameters(max_evals=5)
        self.model.train_best_model()
        self.model.forecast_next_year()
        
        # Test visualization with save path
        test_file = "test_hyperopt_forecast.png"
        self.model.visualize_forecast(save_path=test_file)
        self.assertTrue(os.path.exists(test_file))
        os.remove(test_file)  # Clean up
        
    def test_run_analysis(self):
        """Test complete analysis pipeline."""
        best_params = self.model.run_analysis(max_evals=5)
        
        # Check if all parameters are present
        self.assertIsNotNone(best_params)
        self.assertTrue('changepoint_prior_scale' in best_params)
        self.assertTrue('seasonality_prior_scale' in best_params)
        self.assertTrue('holidays_prior_scale' in best_params)
        self.assertTrue('seasonality_mode' in best_params)

if __name__ == '__main__':
    unittest.main() 