import pandas as pd
import numpy as np
from prophet import Prophet
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from .stock_data import StockData

class StockHyperopt:
    def __init__(self, stock_data):
        """
        Initialize StockHyperopt with StockData object.
        
        Args:
            stock_data (StockData): StockData object containing the stock data
        """
        if not isinstance(stock_data, StockData):
            raise ValueError("Input must be a StockData object")
            
        self.stock_data = stock_data
        self.model = None
        self.best_params = None
        self.forecast = None
        
    def prepare_data(self):
        """
        Prepare data for Prophet model.
        """
        if self.stock_data.dataframe is None:
            raise ValueError("No data available. Please fetch data first.")
            
        # Sort data by date
        self.df = self.stock_data.dataframe.sort_values('Date')
        
        # Prepare data for Prophet
        self.df = self.df.rename(columns={'Date': 'ds', 'Close': 'y'})
        
    def objective(self, params):
        """
        Objective function for hyperparameter optimization.
        
        Args:
            params (dict): Hyperparameters to evaluate
            
        Returns:
            dict: Dictionary containing loss and status
        """
        # Create Prophet model with current parameters
        model = Prophet(
            changepoint_prior_scale=params['changepoint_prior_scale'],
            seasonality_prior_scale=params['seasonality_prior_scale'],
            holidays_prior_scale=params['holidays_prior_scale'],
            seasonality_mode=params['seasonality_mode']
        )
        
        # Fit the model
        model.fit(self.df)
        
        # Make predictions
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        
        # Calculate RMSE on the last 30 days
        y_true = self.df['y'].values[-30:]
        y_pred = forecast['yhat'].values[-30:]
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        return {'loss': rmse, 'status': STATUS_OK}
        
    def optimize_hyperparameters(self, max_evals=50):
        """
        Optimize hyperparameters using hyperopt.
        
        Args:
            max_evals (int): Maximum number of evaluations (default: 50)
        """
        if self.df is None:
            raise ValueError("No data available. Please prepare data first.")
            
        # Define the search space
        space = {
            'changepoint_prior_scale': hp.loguniform('changepoint_prior_scale', -5, 0),
            'seasonality_prior_scale': hp.loguniform('seasonality_prior_scale', -5, 0),
            'holidays_prior_scale': hp.loguniform('holidays_prior_scale', -5, 0),
            'seasonality_mode': hp.choice('seasonality_mode', ['additive', 'multiplicative'])
        }
        
        # Run optimization
        trials = Trials()
        best = fmin(
            fn=self.objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            show_progressbar=False
        )
        
        # Get the best parameters
        self.best_params = {
            'changepoint_prior_scale': best['changepoint_prior_scale'],
            'seasonality_prior_scale': best['seasonality_prior_scale'],
            'holidays_prior_scale': best['holidays_prior_scale'],
            'seasonality_mode': ['additive', 'multiplicative'][best['seasonality_mode']]
        }
        
    def train_best_model(self):
        """
        Train the Prophet model with the best hyperparameters.
        """
        if self.best_params is None:
            raise ValueError("No optimized parameters available. Please run optimize_hyperparameters first.")
            
        # Create Prophet model with best parameters
        self.model = Prophet(
            changepoint_prior_scale=self.best_params['changepoint_prior_scale'],
            seasonality_prior_scale=self.best_params['seasonality_prior_scale'],
            holidays_prior_scale=self.best_params['holidays_prior_scale'],
            seasonality_mode=self.best_params['seasonality_mode']
        )
        
        # Fit the model
        self.model.fit(self.df)
        
    def forecast_next_year(self):
        """
        Forecast stock prices for the next year.
        """
        if self.model is None:
            raise ValueError("No trained model available. Please train the model first.")
            
        # Create future dataframe for next year
        future = self.model.make_future_dataframe(periods=365)
        
        # Make predictions
        self.forecast = self.model.predict(future)
        
    def visualize_forecast(self, save_path=None):
        """
        Visualize the forecast for the next year.
        
        Args:
            save_path (str, optional): Path to save the visualization. If None, the plot will be displayed.
        """
        if self.forecast is None:
            raise ValueError("No forecast available. Please run forecast_next_year first.")
            
        try:
            # Set the style
            plt.style.use('seaborn-v0_8')
            
            # Create the plot
            plt.figure(figsize=(12, 6))
            
            # Plot historical data
            plt.plot(self.df['ds'], self.df['y'], 
                    label='Historical', color='blue', linewidth=2)
            
            # Plot forecast
            plt.plot(self.forecast['ds'], self.forecast['yhat'], 
                    label='Forecast', color='red', linestyle='--', linewidth=2)
            
            # Add confidence intervals
            plt.fill_between(self.forecast['ds'], 
                           self.forecast['yhat_lower'], 
                           self.forecast['yhat_upper'],
                           color='gray', alpha=0.2, label='Confidence Interval')
            
            # Customize the plot
            plt.title(f'{self.stock_data.ticker} Stock Price Forecast for Next Year')
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
            
    def run_analysis(self, max_evals=50):
        """
        Run the complete analysis pipeline.
        
        Args:
            max_evals (int): Maximum number of evaluations for hyperparameter optimization
        """
        self.prepare_data()
        self.optimize_hyperparameters(max_evals)
        self.train_best_model()
        self.forecast_next_year()
        
        return self.best_params

# Example usage
if __name__ == "__main__":
    # Create StockData object
    stock_data = StockData(
        ticker="GOOG",
        start_date="2020-01-01",
        end_date="2023-12-31"
    )
    
    # Fetch the data
    stock_data.fetch_closing_prices()
    
    # Create and run the model
    model = StockHyperopt(stock_data)
    best_params = model.run_analysis()
    
    # Print the best parameters
    print("\nBest Hyperparameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
        
    # Create visualization
    model.visualize_forecast()
    model.visualize_forecast("stock_forecast_next_year.png") 