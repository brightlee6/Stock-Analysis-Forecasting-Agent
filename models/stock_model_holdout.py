import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from .stock_data import StockData

class StockModelHoldout:
    def __init__(self, stock_data):
        """
        Initialize StockModelHoldout with StockData object.
        
        Args:
            stock_data (StockData): StockData object containing the stock data
        """
        if not isinstance(stock_data, StockData):
            raise ValueError("Input must be a StockData object")
            
        self.stock_data = stock_data
        self.train_data = None
        self.test_data = None
        self.model = None
        self.forecast = None
        self.metrics = None
        
    def split_data(self, test_size=0.2):
        """
        Split the data into training and testing sets.
        
        Args:
            test_size (float): Proportion of data to use for testing (default: 0.2)
        """
        if self.stock_data.dataframe is None:
            raise ValueError("No data available. Please fetch data first.")
            
        # Sort data by date
        df = self.stock_data.dataframe.sort_values('Date')
        
        # Calculate split index
        split_idx = int(len(df) * (1 - test_size))
        
        # Split the data
        self.train_data = df.iloc[:split_idx].copy()
        self.test_data = df.iloc[split_idx:].copy()
        
        # Prepare data for Prophet
        self.train_data = self.train_data.rename(columns={'Date': 'ds', 'Close': 'y'})
        self.test_data = self.test_data.rename(columns={'Date': 'ds', 'Close': 'y'})
        
    def train_model(self):
        """
        Train the Prophet model on the training data.
        """
        if self.train_data is None:
            raise ValueError("No training data available. Please split data first.")
            
        # Initialize and fit the model
        self.model = Prophet()
        self.model.fit(self.train_data)
        
    def make_forecast(self):
        """
        Make forecasts on the test data.
        """
        if self.model is None:
            raise ValueError("No trained model available. Please train the model first.")
            
        # Create future dataframe for test dates
        future = self.test_data[['ds']]
        
        # Make predictions
        self.forecast = self.model.predict(future)
        
    def calculate_metrics(self):
        """
        Calculate performance metrics for the forecast.
        """
        if self.forecast is None:
            raise ValueError("No forecast available. Please make forecast first.")
            
        # Extract actual and predicted values
        y_true = self.test_data['y'].values
        y_pred = self.forecast['yhat'].values
        
        # Calculate metrics
        self.metrics = {
            'MAE': mean_absolute_error(y_true, y_pred),
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score(y_true, y_pred)
        }
        
    def visualize_forecast(self, save_path=None):
        """
        Visualize the actual vs predicted values over the test period.
        
        Args:
            save_path (str, optional): Path to save the visualization. If None, the plot will be displayed.
        """
        if self.forecast is None:
            raise ValueError("No forecast available. Please make forecast first.")
            
        try:
            # Set the style
            plt.style.use('seaborn-v0_8')
            
            # Create the plot
            plt.figure(figsize=(12, 6))
            
            # Plot actual values
            plt.plot(self.test_data['ds'], self.test_data['y'], 
                    label='Actual', color='blue', linewidth=2)
            
            # Plot predicted values
            plt.plot(self.test_data['ds'], self.forecast['yhat'], 
                    label='Predicted', color='red', linestyle='--', linewidth=2)
            
            # Add confidence intervals
            plt.fill_between(self.test_data['ds'], 
                           self.forecast['yhat_lower'], 
                           self.forecast['yhat_upper'],
                           color='gray', alpha=0.2, label='Confidence Interval')
            
            # Customize the plot
            plt.title(f'{self.stock_data.ticker} Stock Price: Actual vs Predicted')
            plt.xlabel('Date')
            plt.ylabel('Price ($)')
            plt.legend()
            plt.grid(True)
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)
            
            # Adjust layout
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                print(f"Visualization saved to {save_path}")
            else:
                plt.show()
                
            plt.close()
            
        except Exception as e:
            print(f"Error creating visualization: {str(e)}")
        
    def run_analysis(self, test_size=0.2):
        """
        Run the complete analysis pipeline.
        
        Args:
            test_size (float): Proportion of data to use for testing (default: 0.2)
        """
        self.split_data(test_size)
        self.train_model()
        self.make_forecast()
        self.calculate_metrics()
        
        return self.metrics

# Example usage
if __name__ == "__main__":
    # Create StockData object
    stock_data = StockData(
        ticker="GOOG",
        start_date="2020-01-01",
        end_date="2025-04-19"
    )
    
    # Fetch the data
    stock_data.fetch_closing_prices()
    print(stock_data.dataframe.head())
    
    # Create and run the model
    model = StockModelHoldout(stock_data)
    metrics = model.run_analysis()
    
    # Print the results
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
        
    # Create visualization
    model.visualize_forecast("stock_forecast.png") 