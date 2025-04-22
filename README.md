# Stock Analysis and Forecasting AI Agent

⚠️ **IMPORTANT DISCLAIMER** ⚠️

This project is a **DEMONSTRATION** of how to apply AI and traditional machine learning techniques to stock analysis. It is **NOT** intended for actual investment decisions. Please note:

- This is an educational project demonstrating AI/ML applications
- It does NOT provide investment advice or recommendations
- The forecasts and analyses are for demonstration purposes only
- Do NOT use this for actual investment decisions
- The developers take NO responsibility for any investment decisions made using this tool
- This project is designed to help beginners understand how AI can be applied to traditional ML problems

---

An AI-powered stock analysis system that combines traditional financial analysis with modern generative AI capabilities. This project demonstrates the integration of various AI techniques for stock market analysis and forecasting.

## Project Structure

```
stock-analysis-ai/
├── agent/                  # AI agent implementation
│   ├── __init__.py        # Agent package initialization
│   └── stock_agent.py     # Core agent implementation
├── models/                 # Core analysis modules
│   ├── __init__.py        # Package initialization
│   ├── stock_data.py      # Data fetching and preprocessing
│   ├── stock_model_holdout.py  # Holdout validation model
│   └── stock_hyperopt.py  # Hyperparameter optimization
├── tests/                  # Test suite
│   ├── __init__.py        # Test package initialization
│   ├── test_stock_data.py # Tests for stock_data.py
│   ├── test_stock_model_holdout.py # Tests for holdout model
│   └── test_stock_hyperopt.py # Tests for hyperopt model
├── setup.py               # Package installation configuration
├── install.py            # Installation script
├── run_agent.py          # Script to run the agent
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Project Ideals

This project aims to:
- Demonstrate practical applications of Generative AI in financial analysis
- Combine traditional financial models with modern AI capabilities
- Provide an interactive and intelligent stock analysis experience
- Showcase various AI capabilities including structured output, agents, and MLOps
- Make stock analysis more accessible through natural language interaction

## Potential Usage

This system can be used for:
- Individual investors seeking AI-assisted stock analysis
- Financial analysts looking to augment their analysis with AI insights
- Educational purposes to understand AI applications in finance
- Research in AI-powered financial analysis
- Development of more sophisticated financial AI agents

## Main Components

### 1. StockData Class (`models/stock_data.py`)
- Handles data fetching from Yahoo Finance
- Manages data preprocessing and storage
- Provides visualization capabilities
- Core data management component

### 2. StockModelHoldout Class (`models/stock_model_holdout.py`)
- Implements holdout validation for stock forecasting
- Uses Prophet for time series forecasting
- Calculates performance metrics
- Provides visualization of forecast results

### 3. StockHyperopt Class (`models/stock_hyperopt.py`)
- Implements hyperparameter optimization for forecasting models
- Uses hyperopt for parameter tuning
- Creates optimized forecasting models
- Generates future price predictions

### 4. StockAgent Class (`agent/stock_agent.py`)
- Core AI agent implementation
- Uses Gemini API for natural language understanding
- Manages user interactions and requests
- Coordinates between different analysis components
- Implements various AI capabilities:
  - Structured output generation
  - Few-shot prompting
  - Document understanding
  - Response grounding
  - Context caching

## Testing

The project includes a comprehensive test suite in the `tests/` directory:

### Test Coverage

1. **StockData Tests** (`tests/test_stock_data.py`)
   - Data fetching and validation
   - CSV export functionality
   - Visualization generation
   - Error handling for invalid inputs

2. **StockModelHoldout Tests** (`tests/test_stock_model_holdout.py`)
   - Model initialization and validation
   - Data splitting
   - Model training
   - Forecast generation
   - Performance metrics calculation
   - Visualization testing

3. **StockHyperopt Tests** (`tests/test_stock_hyperopt.py`)
   - Hyperparameter optimization
   - Model training with best parameters
   - Forecast generation
   - Visualization testing
   - Error handling

### Running Tests

To run the tests, use one of the following commands:

```bash
# Run all tests
python -m unittest discover tests

# Run a specific test file
python -m unittest tests/test_stock_data.py

# Run with verbosity
python -m unittest discover tests -v
```

## Detailed Implementation

### AI Capabilities Demonstrated

1. **Structured Output/JSON Mode**
   - Converts natural language queries into structured JSON
   - Enables precise parameter extraction
   - Facilitates consistent data handling

2. **Agents**
   - Implements an intelligent agent architecture
   - Manages complex workflows
   - Coordinates multiple analysis components

3. **Context Caching**
   - Maintains conversation history
   - Enables context-aware responses
   - Supports continuous interaction

4. **Function Calling**
   - Porcess user input
   - Determines appropriate actions
   - Calls corresponding methods (fetch_stock_data, visualize_data, run_holdout_analysis, etc.)

5. **MLOps(with GenAI)**
   - Hyperparameter optimization using hyperopt
   - Model evaluation and metrics calculation
   - Data preprocessing and visualization
   - Model training and forecasting pipeline

6. **Few-shot Prompting** (In progress)
   - Uses example-based learning for better understanding
   - Improves response consistency
   - Reduces ambiguity in user requests

7. **Document Understanding** (In progress)
   - Analyzes company financial reports
   - Extracts key information from documents
   - Provides structured insights

8. **Grounding** (In Progress)
   - Validates responses against historical data
   - Ensures consistency with known facts
   - Provides confidence scores

### Data Flow

1. User input → StockAgent
2. StockAgent processes request using Gemini API
3. Appropriate analysis component is selected
4. Analysis is performed and results are generated
5. Results are validated and returned to user

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd stock-analysis-ai
```

2. Install the package in development mode:
```bash
# Method 1: Using the installation script
python install.py

# Method 2: Using pip directly
pip install -e .
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your Gemini API key
```

## Usage

You can run the agent in several ways:

1. Using the provided script:
```bash
python run_agent.py
```

2. Running the agent module directly:
```bash
python -m agent.stock_agent
```

3. Importing in your own code:
```python
from agent import StockAgent

# Initialize the agent
agent = StockAgent()

# Example interactions
print(agent.process_user_input("Show me Apple's stock data from last year"))
print(agent.process_user_input("Analyze Apple's annual report"))
print(agent.process_user_input("Create a forecast for the next year"))
```

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`:
  - yfinance
  - pandas
  - prophet
  - hyperopt
  - scikit-learn
  - matplotlib
  - seaborn
  - google-generativeai
  - networkx

## Future Enhancements

1. **Additional AI Capabilities**
   - Image understanding for chart analysis
   - Long context window for comprehensive analysis
   - Retrieval augmented generation for better insights

2. **Extended Functionality**
   - Portfolio management
   - Risk assessment
   - Market sentiment analysis
   - Real-time data integration

3. **User Interface**
   - Web interface
   - Mobile application
   - Interactive visualizations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Yahoo Finance for stock data
- Google for Gemini API
- Meta for Prophet library
- Hyperopt developers
- All other open-source contributors 