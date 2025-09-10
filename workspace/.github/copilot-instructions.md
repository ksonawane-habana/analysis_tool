<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Django Model Analysis Tool

This is a Django web application for analyzing model performance data. The tool provides:

## Features
- Interactive bar charts showing model performance distribution
- Drill-down capability from chart bars to model lists
- Detailed subgraph analysis for individual models
- Responsive design with Bootstrap
- CSV data processing with pandas

## Key Components
- **Views**: Handle CSV data loading and processing
- **Templates**: Bootstrap-styled HTML with Chart.js integration
- **Data**: CSV files for model and subgraph performance data
- **Navigation**: Breadcrumb navigation and clickable charts

## Data Structure
- `model_results.csv`: Contains model_name and run_time_gain columns
- `subgraph_results.csv`: Contains model_name, subgraph_name, and run_time_gain columns

## URL Patterns
- `/`: Home page with performance chart
- `/models/<category>/`: Model list for specific performance category
- `/model/<model_name>/`: Subgraph details for specific model
- `/api/category/<category>/`: JSON API for category data

When working with this codebase:
- Use Django template syntax for dynamic content
- Maintain Bootstrap styling consistency
- Ensure pandas is used for CSV processing
- Keep Chart.js charts interactive and responsive
