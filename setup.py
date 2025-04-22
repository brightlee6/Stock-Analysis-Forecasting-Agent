from setuptools import setup, find_packages

setup(
    name="stock-analysis-agent",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain-google-genai",
        "langchain-core",
        "langgraph",
        "matplotlib",
        "seaborn",
        "pandas",
        "python-dotenv",
    ],
) 