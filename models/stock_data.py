import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re

class StockData:
    def __init__(self, ticker, start_date, end_date):
        """
        Initialize a StockData object with ticker and date range.
        
        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL' for Apple)
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            
        Raises:
            ValueError: If ticker is invalid or dates are in wrong format
        """
        # Validate ticker format (1-5 uppercase letters)
        if not re.match(r'^[A-Z]{1,5}$', ticker):
            raise ValueError(f"Invalid ticker symbol: {ticker}. Must be 1-5 uppercase letters.")
            
        # Validate date formats
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            if end < start:
                raise ValueError("End date must be after start date")
        except ValueError as e:
            raise ValueError(f"Invalid date format. Dates must be in YYYY-MM-DD format. Error: {str(e)}")
            
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.dataframe = None
        
    def fetch_closing_prices(self):
        """
        Fetch closing prices for the stock and store in dataframe.
        
        Raises:
            ValueError: If stock data cannot be fetched
        """
        try:
            # Convert string dates to datetime objects
            start = datetime.strptime(self.start_date, '%Y-%m-%d')
            end = datetime.strptime(self.end_date, '%Y-%m-%d')
            
            # Fetch stock data using yf.download
            data = yf.download(self.ticker, start=start, end=end)
            
            if data.empty:
                raise ValueError(f"No data found for ticker {self.ticker}")
            
            # Extract closing prices and reset index to make Date a column
            self.dataframe = data[['Close']].reset_index()
            
            # Rename columns for clarity
            self.dataframe.columns = ['Date', 'Close']
            
            print(f"Successfully fetched closing prices for {self.ticker}")
            
        except Exception as e:
            raise ValueError(f"Error fetching stock data: {str(e)}")
            
    def save_to_csv(self, output_file):
        """
        Save the closing prices to a CSV file.
        
        Args:
            output_file (str): Path to save the CSV file
        """
        if self.dataframe is not None:
            self.dataframe.to_csv(output_file, index=False)
            print(f"Successfully saved closing prices to {output_file}")
        else:
            print("No data available. Please call fetch_closing_prices() first.")
            
    def visualize_data(self, save_path=None):
        """
        Create visualizations for the stock data.
        
        Args:
            save_path (str, optional): Path to save the visualization. If None, the plot will be displayed.
        """
        if self.dataframe is None:
            print("No data available. Please call fetch_closing_prices() first.")
            return
            
        try:
            # Set the style
            plt.style.use('seaborn-v0_8')  # Use a valid style name
            
            # Create a figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot 1: Closing Price Over Time
            sns.lineplot(data=self.dataframe, x='Date', y='Close', ax=ax1)
            ax1.set_title(f'{self.ticker} Closing Price Over Time')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Price ($)')
            ax1.grid(True)
            
            # Calculate daily returns
            self.dataframe['Daily_Return'] = self.dataframe['Close'].pct_change()
            
            # Plot 2: Daily Returns Distribution
            sns.histplot(data=self.dataframe, x='Daily_Return', bins=50, ax=ax2)
            ax2.set_title(f'{self.ticker} Daily Returns Distribution')
            ax2.set_xlabel('Daily Return')
            ax2.set_ylabel('Frequency')
            ax2.grid(True)
            
            # Add some statistics
            mean_return = self.dataframe['Daily_Return'].mean()
            std_return = self.dataframe['Daily_Return'].std()
            ax2.axvline(mean_return, color='r', linestyle='--', label=f'Mean: {mean_return:.4f}')
            ax2.axvline(mean_return + std_return, color='g', linestyle='--', label=f'Std Dev: {std_return:.4f}')
            ax2.axvline(mean_return - std_return, color='g', linestyle='--')
            ax2.legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                print(f"Visualization saved to {save_path}")
            else:
                plt.show()
                
            plt.close()
            
        except Exception as e:
            print(f"Error creating visualization: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Create a StockData object
    apple_stock = StockData(
        #ticker="AAPL",
        ticker="GOOG",
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    
    # Fetch the closing prices
    apple_stock.fetch_closing_prices()
    
    # Save to CSV
    apple_stock.save_to_csv("apple_stock_prices.csv")
    
    # Create visualizations
    apple_stock.visualize_data() 
    apple_stock.visualize_data("apple_stock_analysis.png") 