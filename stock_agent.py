import os
from typing import Dict, List, TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from stock_data import StockData
from stock_model_holdout import StockModelHoldout
from stock_hyperopt import StockHyperopt
import matplotlib
matplotlib.use('Agg')  # Use Agg backend to avoid GUI issues
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import re

# Load environment variables
load_dotenv()

# Define the state type
class AgentState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], "The conversation history"]
    stock_data: StockData | None
    holdout_model: StockModelHoldout | None
    hyperopt_model: StockHyperopt | None
    last_action: str | None

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

def create_visualization(data, title, xlabel, ylabel, save_path=None):
    """
    Create and save a visualization without displaying it.
    
    Args:
        data: DataFrame containing the data to plot
        title: Title of the plot
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        save_path: Path to save the plot (optional)
    """
    try:
        plt.style.use('seaborn-v0_8')
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot the data
        ax.plot(data.index, data.values, label=ylabel, color='blue', linewidth=2)
        
        # Customize the plot
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot if path is provided
        if save_path:
            plt.savefig(save_path)
        
        # Close the figure to free memory
        plt.close(fig)
        
        return True
    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
        return False

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
            # Create visualization using the helper function
            success = create_visualization(
                data=state["stock_data"].dataframe['Close'],
                title=f'{state["stock_data"].ticker} Historical Stock Price',
                xlabel='Date',
                ylabel='Price ($)',
                save_path=f"{state['stock_data'].ticker}_historical.png"
            )
            
            if success:
                state["last_action"] = f"Historical analysis completed. Plot saved as {state['stock_data'].ticker}_historical.png"
            else:
                state["last_action"] = "Error creating historical visualization"
        else:
            state["last_action"] = "No stock data available for analysis"
            
        return state
    except Exception as e:
        state["last_action"] = f"Error in historical analysis: {str(e)}"
        return state

def run_holdout_analysis(state: AgentState) -> AgentState:
    """Run holdout analysis and create forecast."""
    try:
        if state["stock_data"]:
            state["holdout_model"] = StockModelHoldout(state["stock_data"])
            metrics = state["holdout_model"].run_analysis()
            
            # Create visualization using the helper function
            success = create_visualization(
                data=state["holdout_model"].forecast['yhat'],
                title=f'{state["stock_data"].ticker} Forecast',
                xlabel='Date',
                ylabel='Predicted Price ($)',
                save_path=f"{state['stock_data'].ticker}_holdout_forecast.png"
            )
            
            if success:
                metrics_msg = "\n".join([f"{metric}: {value:.4f}" for metric, value in metrics.items()])
                state["last_action"] = f"Holdout analysis completed. Metrics:\n{metrics_msg}\nForecast saved as {state['stock_data'].ticker}_holdout_forecast.png"
            else:
                state["last_action"] = "Error creating forecast visualization"
        else:
            state["last_action"] = "No stock data available for holdout analysis"
            
        return state
    except Exception as e:
        state["last_action"] = f"Error in holdout analysis: {str(e)}"
        return state

def run_hyperopt_analysis(state: AgentState) -> AgentState:
    """Run hyperopt analysis and create optimized forecast."""
    try:
        if state["stock_data"]:
            state["hyperopt_model"] = StockHyperopt(state["stock_data"])
            best_params = state["hyperopt_model"].run_analysis()
            
            # Create visualization using the helper function
            success = create_visualization(
                data=state["hyperopt_model"].forecast['yhat'],
                title=f'{state["stock_data"].ticker} Optimized Forecast',
                xlabel='Date',
                ylabel='Predicted Price ($)',
                save_path=f"{state['stock_data'].ticker}_hyperopt_forecast.png"
            )
            
            if success:
                params_msg = "\n".join([f"{param}: {value}" for param, value in best_params.items()])
                state["last_action"] = f"Hyperopt analysis completed. Best parameters:\n{params_msg}\nForecast saved as {state['stock_data'].ticker}_hyperopt_forecast.png"
            else:
                state["last_action"] = "Error creating optimized forecast visualization"
        else:
            state["last_action"] = "No stock data available for hyperopt analysis"
            
        return state
    except Exception as e:
        state["last_action"] = f"Error in hyperopt analysis: {str(e)}"
        return state

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
            "last_action": None
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
    