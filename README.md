# Multi-Agent Data Analysis Workflow with LangChain and Flask

This project implements a multi-agent data processing pipeline using Python, Flask, and LangChain (LangGraph). The system consists of three specialized agents—`data_filtering_agent`, `data_analysis_agent`, and `data_vis_agent`—which collaboratively process user queries to filter data, analyze it, and generate visualizations using `matplotlib`.

## Project Structure
```
.
├── app.py               # Flask application entry point
├── my_agents.py            # LangChain agents for filtering, analyzing, and visualizing data
├── multi-agent-collaboration.ipynb    # sourced from the official LangGraph repository
└── README.md            # Project documentation
```
The file my_agents.ipynb contains the implementation of this project, while multi-agent-collaboration.ipynb serves as a reference, sourced from the official LangGraph repository.

## Features
- **Multi-Agent Architecture:** Implements a workflow with three specialized agents for data processing.
- **Dynamic Data Filtering:** Users can specify filtering criteria to extract relevant data.
- **Automated Data Analysis:** The analysis agent performs computations on the filtered data.
- **Data Visualization:** Generates static visualizations of analysis results using `matplotlib`.
- **Flask API Integration:** Exposes the multi-agent workflow as an API endpoint.


## Usage
### Running the API
Start the Flask server:
```sh
python app.py
```
The API will be available at `http://127.0.0.1:5000`.

## Here are some possible data visualization questions based on the given dataset:  

1. **Sales Trend Over Time**:  
   - How have total sales changed over time? Can we visualize the sales trend using a time series plot?  

2. **Revenue Distribution by Product Category**:  
   - What is the distribution of revenue across different product categories? Can we use a bar chart or boxplot to compare them?  

3. **Regional Sales Performance**:  
   - How do sales vary by region? Can we visualize this with a grouped bar chart or a heatmap?  

4. **Customer Age vs. Sales Impact**:  
   - Is there a relationship between customer age and total sales? Can we visualize this using a scatter plot or a box plot?  

5. **Effect of Discounts on Revenue**:  
   - How does the discount percentage affect revenue? Can we visualize this with a scatter plot or a regression line?  

6. **Product Sales Comparison Across Quarters**:  
   - How do sales of different product categories fluctuate across different quarters? Can we use a stacked bar chart or a line plot to observe the trend?  

7. **Sales Performance by Customer Rating**:  
   - How does the customer rating affect total sales? Can we use a scatter plot to analyze the correlation?  

8. **Monthly Sales Comparison**:  
   - Which month has the highest and lowest sales? Can we visualize this with a line plot or a bar chart?  

9. **Units Sold vs. Revenue Relationship**:  
   - How do the number of units sold relate to revenue? Can we use a scatter plot to visualize the correlation?  

10. **Top Performing Products**:  
   - Which products generate the highest revenue? Can we use a ranked bar chart to highlight the top-performing products?  
   ```

### Workflow
1. **Data Filtering:** Extracts relevant records based on user-defined conditions.
2. **Data Analysis:** Computes summary statistics or grouped aggregations.
3. **Data Visualization:** Generates `matplotlib` visualizations based on the processed data.



