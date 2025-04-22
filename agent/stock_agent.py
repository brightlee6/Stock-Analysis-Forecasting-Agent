import os
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import os
from typing import Dict, List, TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from models.stock_data import StockData
from models.stock_model_holdout import StockModelHoldout
from models.stock_hyperopt import StockHyperopt
import matplotlib
matplotlib.use('Agg')  # Use Agg backend to avoid GUI issues
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import re
import pandas as pd

# Load environment variables
load_dotenv()

# Define the state type
class AgentState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], "The conversation history"]
    stock_data: StockData | None
    holdout_model: StockModelHoldout | None
    hyperopt_model: StockHyperopt | None
    last_action: str | None
    error: str | None

# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    # model="gemini-pro",
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7
)

# Define the system prompt
system_prompt = """You are a helpful stock market analysis assistant. Your role is to:
1. Understand user requests about stock data
2. Extract stock tickers and date ranges from user input
3. Perform appropriate stock analysis (historical data, forecasting, or hyperparameter tuning)
4. Provide clear and informative responses

When analyzing stocks, you can:
- Show historical price data
- Create forecasts using Prophet
- Tune hyperparameters for better predictions
- Visualize results

Always be clear about what you're doing and explain the results in a way that's easy to understand."""

# Create the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="messages"),
])

def parse_relative_date(date_str: str) -> str:
    """
    Convert relative date expressions to YYYY-MM-DD format.
    
    Args:
        date_str (str): Date string that might contain relative expressions
        
    Returns:
        str: Date in YYYY-MM-DD format
    """
    today = datetime.now()
    
    # Handle "today"
    if date_str.lower() == "today":
        return today.strftime('%Y-%m-%d')
    
    # Handle "X years ago"
    match = re.match(r'(\d+)\s+years?\s+ago', date_str.lower())
    if match:
        years = int(match.group(1))
        return (today - timedelta(days=years*365)).strftime('%Y-%m-%d')
    
    # Handle "X months ago"
    match = re.match(r'(\d+)\s+months?\s+ago', date_str.lower())
    if match:
        months = int(match.group(1))
        return (today - timedelta(days=months*30)).strftime('%Y-%m-%d')
    
    # Handle "X days ago"
    match = re.match(r'(\d+)\s+days?\s+ago', date_str.lower())
    if match:
        days = int(match.group(1))
        return (today - timedelta(days=days)).strftime('%Y-%m-%d')
    
    # If it's already in YYYY-MM-DD format, return as is
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return date_str
    except ValueError:
        # If we can't parse it, return today's date
        return today.strftime('%Y-%m-%d')

# Define the nodes in the graph
def extract_stock_info(state: AgentState) -> AgentState:
    """Extract stock information from user input."""
    try:
        # Get the last user message
        last_message = state["messages"][-1].content
        
        # Use LLM to extract stock info
        response = llm.invoke([
            HumanMessage(content=f"""Extract the stock ticker symbol and date range from this text: {last_message}
            Return the information in this format:
            TICKER: [ticker]
            START_DATE: [start_date]
            END_DATE: [end_date]
            If any information is missing, use defaults:
            - Default start_date: 3 years ago
            - Default end_date: today""")
        ])
        
        # Parse the response
        info = {}
        for line in response.content.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                info[key.strip()] = value.strip()
        
        # Parse dates
        ticker = info.get('TICKER', '')
        start_date = parse_relative_date(info.get('START_DATE', '3 years ago'))
        end_date = parse_relative_date(info.get('END_DATE', 'today'))
        
        if ticker:
            state["stock_data"] = StockData(ticker, start_date, end_date)
            state["stock_data"].fetch_closing_prices()
            
        return state
    except Exception as e:
        state["last_action"] = f"Error extracting stock info: {str(e)}"
        return state

def analyze_historical_data(state: AgentState) -> AgentState:
    """Analyze and visualize historical stock data."""
    try:
        if state["stock_data"] and state["stock_data"].dataframe is not None:
            # Create visualization using StockData's visualize_data method
            state["stock_data"].visualize_data()
            save_path = f"{state['stock_data'].ticker}_historical.png"
            state["stock_data"].visualize_data(save_path=save_path)
            
            state["last_action"] = f"Historical analysis completed. Plot saved as {save_path}"
        else:
            state["last_action"] = "No stock data available for analysis"
            
        return state
    except Exception as e:
        state["last_action"] = f"Error in historical analysis: {str(e)}"
        return state

def run_holdout_analysis(state: AgentState) -> AgentState:
    """Run holdout analysis on stock data."""
    try:
        # Extract stock ticker and date range from state
        ticker = state.get("ticker")
        if not ticker:
            return AgentState(
                last_action="Error: No stock ticker provided",
                error="No stock ticker provided"
            )
                
        # Create StockData instance and fetch data
        stock_data = StockData(ticker, "2020-01-01", "2023-12-31")
        stock_data.fetch_closing_prices()
        
        # Split data into train and test sets
        data = stock_data.dataframe
        train_size = int(len(data) * 0.8)
        train_data = data[:train_size]
        test_data = data[train_size:]
        
        # Calculate metrics
        train_mean = train_data['Close'].mean()
        test_mean = test_data['Close'].mean()
        train_std = train_data['Close'].std()
        test_std = test_data['Close'].std()
        
        # Create analysis message
        analysis_msg = f"""
        Holdout Analysis Results for {ticker}:
        - Training Period: {train_data['Date'].iloc[0].strftime('%Y-%m-%d')} to {train_data['Date'].iloc[-1].strftime('%Y-%m-%d')}
        - Test Period: {test_data['Date'].iloc[0].strftime('%Y-%m-%d')} to {test_data['Date'].iloc[-1].strftime('%Y-%m-%d')}
        - Training Mean Price: ${train_mean:.2f}
        - Test Mean Price: ${test_mean:.2f}
        - Training Standard Deviation: ${train_std:.2f}
        - Test Standard Deviation: ${test_std:.2f}
        """
        
        return AgentState(
            last_action=analysis_msg,
            error=None
        )
        
    except Exception as e:
        return AgentState(
            last_action=f"Error running holdout analysis: {str(e)}",
            error=str(e)
        )

def run_hyperopt_analysis(state: AgentState) -> AgentState:
    """Run hyperparameter optimization analysis."""
    try:
        # Extract stock ticker from state
        ticker = state.get("ticker")
        if not ticker:
            return AgentState(
                last_action="Error: No stock ticker provided",
                error="No stock ticker provided"
            )
                
        # Create StockData instance and fetch data
        stock_data = StockData(ticker, "2020-01-01", "2023-12-31")
        stock_data.fetch_closing_prices()
        
        # Calculate optimal parameters
        data = stock_data.dataframe
        optimal_window = _calculate_optimal_window(data['Close'])
        optimal_threshold = _calculate_optimal_threshold(data['Close'])
        
        # Create analysis message
        analysis_msg = f"""
        Hyperparameter Optimization Results for {ticker}:
        - Optimal Moving Average Window: {optimal_window} days
        - Optimal Trading Threshold: {optimal_threshold:.2f}%
        - Analysis Period: {data['Date'].iloc[0].strftime('%Y-%m-%d')} to {data['Date'].iloc[-1].strftime('%Y-%m-%d')}
        """
        
        return AgentState(
            last_action=analysis_msg,
            error=None
        )
        
    except Exception as e:
        return AgentState(
            last_action=f"Error running hyperparameter optimization: {str(e)}",
            error=str(e)
        )

def _calculate_optimal_window(prices: pd.Series) -> int:
    """Calculate optimal moving average window size."""
    # Simple implementation - can be enhanced with more sophisticated methods
    return min(50, len(prices) // 10)

def _calculate_optimal_threshold(prices: pd.Series) -> float:
    """Calculate optimal trading threshold."""
    # Simple implementation - can be enhanced with more sophisticated methods
    return 2.0  # 2% threshold

def generate_response(state: AgentState) -> AgentState:
    """Generate a response based on the last action."""
    try:
        # Get the conversation history
        messages = state["messages"]
        
        # Add the last action to the context
        context = f"Last action: {state['last_action']}\n\nUser's last message: {messages[-1].content}"
        
        # Generate response
        response = llm.invoke([
            HumanMessage(content=context)
        ])
        
        # Add the response to the conversation history
        state["messages"].append(AIMessage(content=response.content))
        
        return state
    except Exception as e:
        state["messages"].append(AIMessage(content=f"Error generating response: {str(e)}"))
        return state

# Create the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("extract_stock_info", extract_stock_info)
workflow.add_node("analyze_historical", analyze_historical_data)
workflow.add_node("run_holdout", run_holdout_analysis)
workflow.add_node("run_hyperopt", run_hyperopt_analysis)
workflow.add_node("generate_response", generate_response)

# Define edges
def route_based_on_intent(state: AgentState) -> str:
    """Route to the appropriate node based on user intent."""
    last_message = state["messages"][-1].content.lower()
    
    # Check for specific keywords in the message
    if "forecast" in last_message and ("tune" in last_message or "optimize" in last_message or "hyperparameter" in last_message):
        return "run_hyperopt"
    elif "forecast" in last_message or "predict" in last_message:
        return "run_holdout"
    elif "historical" in last_message or "price" in last_message or "show" in last_message:
        return "analyze_historical"
    else:
        # If no specific intent is detected, ask the LLM to determine the intent
        try:
            response = llm.invoke([
                HumanMessage(content=f"""Based on this user message, determine which analysis to perform:
                {last_message}
                
                Return ONLY one of these exact options:
                - analyze_historical
                - run_holdout
                - run_hyperopt
                - generate_response
                
                Choose based on the user's intent to:
                - analyze_historical: for showing historical data or prices
                - run_holdout: for forecasting or predicting future prices
                - run_hyperopt: for optimizing or tuning forecast models
                - generate_response: for general questions or clarifications""")
            ])
            return response.content.strip()
        except Exception as e:
            print(f"Error determining intent: {str(e)}")
            return "generate_response"

# Add edges
workflow.add_conditional_edges(
    "extract_stock_info",
    route_based_on_intent,
    {
        "analyze_historical": "analyze_historical",
        "run_holdout": "run_holdout",
        "run_hyperopt": "run_hyperopt",
        "generate_response": "generate_response"
    }
)

workflow.add_edge("analyze_historical", "generate_response")
workflow.add_edge("run_holdout", "generate_response")
workflow.add_edge("run_hyperopt", "generate_response")
workflow.add_edge("generate_response", END)

# Set the entry point
workflow.set_entry_point("extract_stock_info")

# Compile the graph
app = workflow.compile()

class StockAgent:
    def __init__(self):
        """Initialize the StockAgent."""
        self.state = {
            "messages": [],
            "stock_data": None,
            "holdout_model": None,
            "hyperopt_model": None,
            "last_action": None,
            "error": None
        }
        
    def process_user_input(self, user_input: str) -> str:
        """
        Process user input and return a response.
        
        Args:
            user_input (str): The user's input message
            
        Returns:
            str: The agent's response
        """
        try:
            # Add user message to state
            self.state["messages"].append(HumanMessage(content=user_input))
            
            # Run the workflow
            self.state = app.invoke(self.state)
            
            # Return the last AI message
            return self.state["messages"][-1].content
            
        except Exception as e:
            return f"Error processing request: {str(e)}"

# Example usage
if __name__ == "__main__":
    agent = StockAgent()
    
    # Example interactions
    print(agent.process_user_input("Show me the historical stock price of GOOG"))
    print(agent.process_user_input("Forecast the stock price of GOOG"))
    print(agent.process_user_input("Tune hyperparameters and forecast the stock price of GOOG")) 
    