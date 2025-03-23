from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import base64
from typing import Annotated, Literal, List, Dict, Any, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from IPython.display import HTML
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command

# Flask
app = Flask(__name__)
CORS(app)


# Create a random dataset that we'll use for analysis
def create_random_dataset(rows=100):
    """Create a random sales dataset for demonstration purposes."""
    np.random.seed(42)
    
    # Create date range for the past 2 years
    dates = pd.date_range(end=pd.Timestamp.now(), periods=rows, freq='W')
    
    # Create product categories and regions
    categories = ['Electronics', 'Clothing', 'Home Goods', 'Sports', 'Books']
    regions = ['North', 'South', 'East', 'West', 'Central']
    
    # Generate random data
    data = {
        'Date': dates,
        'Product': np.random.choice(categories, size=rows),
        'Region': np.random.choice(regions, size=rows),
        'Sales': np.random.randint(1000, 10000, size=rows),
        'Units': np.random.randint(10, 100, size=rows),
        'Customer_Age': np.random.randint(18, 65, size=rows),
        'Rating': np.round(np.random.uniform(1, 5, size=rows), 1),
        'Discount': np.round(np.random.uniform(0, 0.3, size=rows), 2)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some calculated columns
    df['Revenue'] = df['Sales'] - (df['Sales'] * df['Discount'])
    df['Month'] = df['Date'].dt.month_name()
    df['Quarter'] = 'Q' + df['Date'].dt.quarter.astype(str)
    df['Year'] = df['Date'].dt.year
    
    return df

# Create the dataset
sales_df = create_random_dataset(150)


# Function to display dataframe info
def get_dataframe_info():
    """Get information about the dataframe structure"""
    buffer = io.StringIO()
    sales_df.info(buf=buffer)
    info_str = buffer.getvalue()
    
    # Get basic statistics
    stats = sales_df.describe().to_string()
    
    # Get sample rows
    sample = sales_df.head(5).to_string()
    
    return f"""
DataFrame Information:
{info_str}

Basic Statistics:
{stats}

Sample Rows:
{sample}
"""


# Create global variables to store intermediate results
filtered_data = None
analysis_data = None
# Define tools for each agent
@tool
def filter_data_function(
    query: Annotated[str, "The filtering criteria to apply to the dataset"]
) -> str:
    """
    Filter the sales dataset based on the provided query.
    The query should be a valid Python expression that can be evaluated in the context of a pandas DataFrame.
    Examples: "df[df['Region'] == 'North']", "df[(df['Sales'] > 5000) & (df['Product'] == 'Electronics')]"
    """
    global filtered_data
    
    try:
        # Create a local copy of the dataframe to evaluate the expression
        df = sales_df.copy()
        
        # Evaluate the query
        filtered_df = eval(query)
        
        if isinstance(filtered_df, pd.DataFrame):
            if filtered_df.empty:
                return "The filter returned no results. Please try a different filter."
            
            result = f"Filtered data successfully. {len(filtered_df)} rows returned.\n"
            result += f"Sample of filtered data (first 5 rows):\n{filtered_df.head(5).to_string()}"
            
            # Store the filtered dataframe for later use
            filtered_data = filtered_df
            
            return result
        else:
            return "Error: The filter did not return a DataFrame. Make sure your filter returns a DataFrame."
    except Exception as e:
        return f"Error in filtering data: {str(e)}"

@tool
def analyze_data_function(
    analysis_code: Annotated[str, "The Python code to analyze the filtered data"]
) -> str:
    """
    Run analysis on the filtered dataset. The analysis_code should be valid Python code.
    The filtered data is available as the variable 'df'.
    Example: "df.groupby('Product')['Sales'].sum()"
    """
    global analysis_data
    
    try:
        if filtered_data is None:
            return "No filtered data available. Please run filter_data first."
        
        # Create a local copy with a simple name
        df = filtered_data.copy()
        
        # Execute the analysis code
        result = eval(analysis_code)
        
        # Convert various result types to string representation
        if isinstance(result, pd.DataFrame):
            analysis_result = f"Analysis results:\n{result.to_string()}"
            
            # Store for visualization
            analysis_data = result
            
            return analysis_result
        elif isinstance(result, pd.Series):
            analysis_result = f"Analysis results:\n{result.to_string()}"
            
            # Store for visualization
            analysis_data = result
            
            return analysis_result
        else:
            return f"Analysis results:\n{result}"
    except Exception as e:
        return f"Error in analyzing data: {str(e)}"

@tool
def visualize_data_function(
    viz_code: Annotated[str, "The Python code using matplotlib to visualize the analyzed data"]
) -> str:
    """
    Create visualizations based on the analyzed data using matplotlib.
    The analyzed data is available as 'df' for filtered data or 'analysis_data' for analysis results.
    The code should create a matplotlib figure and save it.
    """
    print("executing visualize_data_function")
    try:
        plt.close('all')
        
        # Check if we have data to visualize
        if filtered_data is None:
            return "No filtered data available. Please run filter_data first."
        
        # Make data available as df
        df = filtered_data.copy()
        
        # Make analysis data available if it exists
        if analysis_data is not None:
            analysis_df = analysis_data
        
        # Create a figure
        plt.figure(figsize=(10, 6))
        
        # Execute the visualization code
        exec(viz_code)
        
        # Save the figure to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        
        # Encode to base64 for display
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        # Close the figure to free memory
        plt.close()
        print("visualize_data_function finished")
        # For Jupyter notebook, we can display the image
        # if 'get_ipython' in globals():
        #     display(HTML(f'<img src="data:image/png;base64,{img_str}"/>'))
        # print("visualize_data_function finished")
        return f"FINAL ANSWER. Visualization created successfully.\n<img src='data:image/png;base64,{img_str}'/>"
    except Exception as e:
        return f"Error in creating visualization: {str(e)}"

# Initialize the OpenAI LLM
llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

# Create system prompts for each agent
def make_system_prompt(role_description):
    dataset_info = get_dataframe_info()
    return f"""You are an AI assistant that is part of a multi-agent system for data analysis and visualization.

{role_description}

You have access to a sales dataset with the following structure:

{dataset_info}

Follow these guidelines:
1. Think step by step about the user's question
2. Determine what actions you need to take based on your specific role
3. Use your available tools to complete your part of the task
4. When you've completed your responsibility, hand off to the appropriate next agent
5. Only respond with FINAL ANSWER when the entire workflow is complete and a visualization has been created

Collaborate with the other agents to complete the full task pipeline: 
data_filtering → data_analysis → data_visualization
"""

# Create the agents
data_filtering_agent = create_react_agent(
    llm,
    tools=[filter_data_function],
    prompt=make_system_prompt("Your role is DATA FILTERING. You can only filter the dataset based on user queries. You work with data analysis and visualization colleagues.")
)

data_analysis_agent = create_react_agent(
    llm,
    tools=[analyze_data_function],
    prompt=make_system_prompt("Your role is DATA ANALYSIS. You can only analyze the filtered dataset. You work with data filtering and visualization colleagues.")
)

data_vis_agent = create_react_agent(
    llm,
    tools=[visualize_data_function],
    prompt=make_system_prompt("Your role is DATA VISUALIZATION. You can only create visualizations using matplotlib based on the filtered and analyzed data. You work with data filtering and analysis colleagues.")
)

# Define the graph nodes
def get_next_node(last_message: BaseMessage, goto: str):
    if "FINAL ANSWER" in last_message.content:
        return END
    return goto

def data_filtering_node(state: MessagesState) -> Command[Literal["data_analysis", END]]:
    result = data_filtering_agent.invoke(state)
    print("data_filtering_node result")
    print(result)
    goto = get_next_node(result["messages"][-1], "data_analysis")
    
    # Wrap in a human message with agent name
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name="data_filtering_agent"
    )
    
    return Command(
        update={"messages": result["messages"]},
        goto=goto,
    )

def data_analysis_node(state: MessagesState) -> Command[Literal["data_vis", END]]:
    result = data_analysis_agent.invoke(state)
    print("data_analysis_node result")
    print(result)
    goto = get_next_node(result["messages"][-1], "data_vis")
    
    # Wrap in a human message with agent name
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name="data_analysis_agent"
    )
    
    return Command(
        update={"messages": result["messages"]},
        goto=goto,
    )

def data_vis_node(state: MessagesState) -> Command[Literal[END]]:
    result = data_vis_agent.invoke(state)
    print("data_vis_node result")
    print(result)
    goto = END  # Visualization is the last step
    
    # Wrap in a human message with agent name
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name="data_vis_agent"
    )
    
    return Command(
        update={"messages": result["messages"]},
        goto=goto,
    )

# Create the graph
workflow = StateGraph(MessagesState)
workflow.add_node("data_filtering", data_filtering_node)
workflow.add_node("data_analysis", data_analysis_node)
workflow.add_node("data_vis", data_vis_node)

# Add edges
workflow.add_edge(START, "data_filtering")
workflow.add_edge("data_filtering", "data_analysis")
workflow.add_edge("data_analysis", "data_vis")

# Compile the graph
graph = workflow.compile()


def process_query(user_query: str):
    """Process a user query through the multi-agent workflow"""
    messages = [HumanMessage(content=user_query)]

    print(messages)
    result = graph.invoke({"messages": messages})
    
    # Extract the final messages
    final_messages = result["messages"]

    # **DEBUG: 检查 function.name 是否符合规范**
    for i, message in enumerate(final_messages):
        if isinstance(message, AIMessage) and hasattr(message, "tool_calls"):
            for j, tool_call in enumerate(message.tool_calls):
                # print(j,"toolcall:", tool_call)
                if "name" in tool_call:  # 确保 `name` 字段存在
                    function_name = tool_call["name"]
                    if not re.match(r"^[a-zA-Z0-9_-]+$", function_name):
                        print(f"\n⚠️ Invalid function name at messages[{i}].tool_calls[{j}]: {function_name}")


    # **DEBUG: 打印整个对话过程**
    for message in final_messages:
        if hasattr(message, 'name') and message.name:
            print(f"\n[{message.name}]: {message.content}\n")
        elif isinstance(message, HumanMessage):
            print(f"\n[User]: {message.content}\n")
        elif isinstance(message, AIMessage):
            print(f"\n[AI]: {message.content}\n")
        else:
            print(f"\n[Unknown]: {message.content}\n")

    # Find and extract the last message with visualization
    for message in reversed(final_messages):
        if isinstance(message, HumanMessage) and "data_vis_agent" == message.name and "FINAL ANSWER" in message.content:
            return message.content
    
    return "No visualization was generated."



# Flask API 路由
@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.json
        user_query = data.get("query")
        if not user_query:
            return jsonify({"error": "Missing query parameter"}), 400
        
        result = process_query(user_query)
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 运行 Flask 应用
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
