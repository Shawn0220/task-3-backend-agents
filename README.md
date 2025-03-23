# Multi-Agent Data Analysis Workflow with LangChain and Flask

This project implements a multi-agent data processing pipeline using Python, Flask, and LangChain (LangGraph). The system consists of three specialized agents—`data_filtering_agent`, `data_analysis_agent`, and `data_vis_agent`—which collaboratively process user queries to filter data, analyze it, and generate visualizations using `matplotlib`.

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

### Example Query
1. **Filter Data:**
   ```json
   {
      "query": "df[df['Region'] == 'North']"
   }
   ```
2. **Analyze Data:**
   ```json
   {
      "query": "df.groupby('Product')['Sales'].sum()"
   }
   ```
3. **Visualize Data:**
   ```json
   {
      "query": "df.plot(kind='bar'); plt.xlabel('Product'); plt.ylabel('Total Sales'); plt.title('Sales by Product')"
   }
   ```

### Workflow
1. **Data Filtering:** Extracts relevant records based on user-defined conditions.
2. **Data Analysis:** Computes summary statistics or grouped aggregations.
3. **Data Visualization:** Generates `matplotlib` visualizations based on the processed data.



