"""
Script to run the Stock Analysis and Forecasting Agent.
"""
from agent import StockAgent

def main():
    agent = StockAgent()
    
    # Example interactions
    print(agent.process_user_input("Show me the historical stock price of GOOG"))
    print(agent.process_user_input("Forecast the stock price of GOOG"))
    print(agent.process_user_input("Tune hyperparameters and forecast the stock price of GOOG"))

if __name__ == "__main__":
    main() 