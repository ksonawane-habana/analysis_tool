from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
import os
from django.conf import settings
import json
from collections import defaultdict
from functools import lru_cache
import glob

# Cache the CSV data loading to avoid reloading on every request
@lru_cache(maxsize=2)
def load_csv_data():
    """Load and return CSV data - cached for performance"""
    feature_dir_path = os.environ["FEATURE_DIR"]
    
    # Load your actual result files
    model_file = os.path.join(feature_dir_path, 'asic2/g3_promo_models_full', 'results.csv')
    graph_file = os.path.join(feature_dir_path, 'asic1/g3_promo_models','results.csv')
    
    model_data = None
    graph_data = None

    print(f"Loading CSV files from {feature_dir_path}")

    if os.path.exists(model_file):
        print(f"Loading model data from {model_file}")
        model_data = pd.read_csv(model_file)
        # Filter for 'feature' results to get the optimized performance
        model_data = model_data[model_data['Name'] == 'feature']
        # Clean up model names for better display
        model_data['display_name'] = model_data['Model'].str.replace('pt_', '').str.replace('_', ' ')
        print(f"Loaded {len(model_data)} model records")
    
    if os.path.exists(graph_file):
        print(f"Loading graph data from {graph_file}")
        # Load all necessary columns for comprehensive display
        cols_needed = ['Model', 'Name', 'Device', 'Status', 'Run time [ms]', 'Run time gain [%]', 
                      'Run time diff [ms]', 'Compile time [ms]', 'Compile time gain [%]', 
                      'Workspace [bytes]', 'Workspace gain [%]']
        graph_data = pd.read_csv(graph_file, usecols=cols_needed)
        print(f"Loaded {len(graph_data)} graph records")
    
    return model_data, graph_data


def home(request):
    """Main page with bar chart showing models performance"""
    model_data, _ = load_csv_data()
    
    if model_data is not None and len(model_data) > 0:
        # Dynamic categorization based on actual data distribution
        gains = model_data['Run time gain [%]']
        min_gain = gains.min()
        max_gain = gains.max()
        
        # Calculate how many bins we need on each side of 0 to accommodate the data
        # Use 1% intervals and ensure 0 is in the middle
        bins_needed_negative = max(1, int(abs(min_gain)) + 1) if min_gain < 0 else 0
        bins_needed_positive = max(1, int(abs(max_gain)) + 1) if max_gain > 0 else 0
        
        # Make sure we have the same number of bins on each side for symmetry
        max_bins_per_side = max(bins_needed_negative, bins_needed_positive)
        max_bins_per_side = min(max_bins_per_side, 20)  # Cap at 20 to avoid too many bins
        
        # Create symmetric range around 0 with 1% intervals
        range_min = -max_bins_per_side
        range_max = max_bins_per_side
        
        # Create bins with 1% intervals, ensuring 0 is at the center
        bins = list(range(range_min, range_max + 1, 1))
        
        # Create labels for the bins with special handling for 0%
        labels = []
        for i in range(len(bins) - 1):
            if bins[i] == -1 and bins[i+1] == 0:
                labels.append(f'-1% to <0%')
            elif bins[i] == 0 and bins[i+1] == 1:
                labels.append(f'>0% to 1%')
            elif bins[i+1] == bins[-1]:  # Last bin
                labels.append(f'≥ {bins[i]}%')
            else:
                labels.append(f'{bins[i]}% to {bins[i+1]}%')
        
        # Add the standalone 0% column - insert it in the middle
        zero_bin_index = None
        for i, (lower, upper) in enumerate(zip(bins[:-1], bins[1:])):
            if lower < 0 and upper > 0:
                zero_bin_index = i + 1  # Insert after the negative bin
                break
        
        if zero_bin_index is not None:
            labels.insert(zero_bin_index, '0%')
        else:
            # Fallback: add 0% in the middle if we can't find the right position
            labels.insert(len(labels) // 2, '0%')
        
        # Handle exact 0% values first by creating a temporary series
        gains_copy = gains.copy()
        zero_mask = (gains == 0.0)
        
        # Temporarily replace 0% values with a small negative number for categorization
        # This ensures they get categorized in the negative range, then we'll fix them
        if zero_mask.any():
            gains_copy.loc[zero_mask] = -0.001  # Small negative value that will go in the <0% bin
        
        # Use pandas cut to categorize the data
        # Use right=False to make bins left-inclusive, right-exclusive [a, b)
        regular_labels = [l for l in labels if l != '0%']
        model_data['gain_category'] = pd.cut(gains_copy, bins=bins, labels=regular_labels, include_lowest=True, right=False)
        
        # Now fix the 0% values by converting to object dtype first (allows arbitrary string assignment)
        model_data['gain_category'] = model_data['gain_category'].astype(str)
        if zero_mask.any():
            model_data.loc[zero_mask, 'gain_category'] = '0%'
        
        category_counts = model_data['gain_category'].value_counts().reindex(labels, fill_value=0)
        
        # Generate colors dynamically based on actual data distribution
        colors = []
        total_bins = len(labels)
        
        # Find the index of the 0% column
        zero_index = None
        try:
            zero_index = labels.index('0%')
        except ValueError:
            # Fallback if 0% column not found
            zero_index = len(labels) // 2
        
        for i, label in enumerate(labels):
            if label == '0%':
                # Special color for exactly 0% - neutral gray
                colors.append('rgb(128, 128, 128)')
            elif i < zero_index:
                # Degradation bins (left of zero)
                distance_from_zero = zero_index - i
                max_distance_left = zero_index
                intensity = distance_from_zero / max_distance_left if max_distance_left > 0 else 0
                # Scale from light red to dark red
                red_value = int(139 + (255 - 139) * intensity)
                colors.append(f'rgb({red_value}, {max(0, 139 - int(139 * intensity))}, {max(0, 139 - int(139 * intensity))})')
            else:
                # Improvement bins (right of zero)
                distance_from_zero = i - zero_index
                max_distance_right = total_bins - zero_index - 1
                intensity = min(distance_from_zero / max_distance_right if max_distance_right > 0 else 0, 1)
                # Scale from light green to dark green
                green_value = int(144 + (255 - 144) * (1 - intensity))
                colors.append(f'rgb({max(0, 50 - int(50 * intensity))}, {green_value}, {max(0, 50 - int(50 * intensity))})')
        
        # Store category mapping for URL generation
        category_mapping = {}
        bin_index = 0  # Track position in bins array
        
        for i, label in enumerate(labels):
            if label == '0%':
                # Special handling for exact 0%
                category_mapping[label] = 'range_0_0'
            else:
                # Handle regular bins
                if bin_index < len(bins) - 1:
                    # Handle negative numbers in URL
                    min_part = f"neg{abs(bins[bin_index])}" if bins[bin_index] < 0 else str(bins[bin_index])
                    max_part = f"neg{abs(bins[bin_index + 1])}" if bins[bin_index + 1] < 0 else str(bins[bin_index + 1])
                    key = f'range_{min_part}_{max_part}'
                else:
                    # Last bin (≥ X%)
                    min_part = f"neg{abs(bins[bin_index])}" if bins[bin_index] < 0 else str(bins[bin_index])
                    key = f'range_{min_part}_inf'
                category_mapping[label] = key
                bin_index += 1
        
        chart_data = {
            'labels': labels,
            'data': category_counts.tolist(),
            'colors': colors,
            'categoryMapping': category_mapping
        }
    else:
        # Fallback data if no CSV is loaded - use 1% intervals with 0 in the middle
        labels = ['-10% to -9%', '-9% to -8%', '-8% to -7%', '-7% to -6%', '-6% to -5%',
                 '-5% to -4%', '-4% to -3%', '-3% to -2%', '-2% to -1%', '-1% to <0%',
                 '0%', '>0% to 1%', '1% to 2%', '2% to 3%', '3% to 4%', '4% to 5%',
                 '5% to 6%', '6% to 7%', '7% to 8%', '8% to 9%', '9% to 10%']
        
        # Generate dynamic colors for fallback data
        fallback_colors = []
        total_fallback_bins = len(labels)
        zero_index = labels.index('0%')
        
        for i, label in enumerate(labels):
            if label == '0%':
                # Special color for exactly 0% - neutral gray
                fallback_colors.append('rgb(128, 128, 128)')
            elif i < zero_index:  # Degradation categories (left of zero)
                # Distance from zero
                distance_from_zero = zero_index - i
                max_distance_left = zero_index
                intensity = distance_from_zero / max_distance_left if max_distance_left > 0 else 0
                red_value = int(139 + (255 - 139) * intensity)
                fallback_colors.append(f'rgb({red_value}, {max(0, 139 - int(139 * intensity))}, {max(0, 139 - int(139 * intensity))})')
            else:  # Improvement categories (right of zero)
                # Distance from zero
                distance_from_zero = i - zero_index
                max_distance_right = total_fallback_bins - zero_index - 1
                intensity = min(distance_from_zero / max_distance_right if max_distance_right > 0 else 0, 1)
                green_value = int(144 + (255 - 144) * (1 - intensity))
                fallback_colors.append(f'rgb({max(0, 50 - int(50 * intensity))}, {green_value}, {max(0, 50 - int(50 * intensity))})')
        
        category_mapping = {
            '-10% to -9%': 'range_neg10_neg9',
            '-9% to -8%': 'range_neg9_neg8',
            '-8% to -7%': 'range_neg8_neg7',
            '-7% to -6%': 'range_neg7_neg6',
            '-6% to -5%': 'range_neg6_neg5',
            '-5% to -4%': 'range_neg5_neg4',
            '-4% to -3%': 'range_neg4_neg3',
            '-3% to -2%': 'range_neg3_neg2',
            '-2% to -1%': 'range_neg2_neg1',
            '-1% to <0%': 'range_neg1_0',
            '0%': 'range_0_0',
            '>0% to 1%': 'range_0_1',
            '1% to 2%': 'range_1_2',
            '2% to 3%': 'range_2_3',
            '3% to 4%': 'range_3_4',
            '4% to 5%': 'range_4_5',
            '5% to 6%': 'range_5_6',
            '6% to 7%': 'range_6_7',
            '7% to 8%': 'range_7_8',
            '8% to 9%': 'range_8_9',
            '9% to 10%': 'range_9_10'
        }
        chart_data = {
            'labels': labels,
            'data': [1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 5, 12, 10, 8, 6, 5, 4, 3, 2, 2, 1],
            'colors': fallback_colors,
            'categoryMapping': category_mapping
        }
    
    return render(request, 'home.html', {'chart_data': json.dumps(chart_data)})


def model_list(request, category):
    """List of models in a specific runtime gain category"""
    model_data, _ = load_csv_data()
    
    # Handle both old static categories and new dynamic categories for backward compatibility
    if category in ['degrading_low', 'degrading_medium', 'degrading_high', 'improving_low', 'improving_medium', 'improving_high']:
        # Handle old static categories as fallback
        static_ranges = {
            'degrading_high': (-float('inf'), -5),
            'degrading_medium': (-5, -2),
            'degrading_low': (-2, 0),
            'improving_low': (0, 2),
            'improving_medium': (2, 5),
            'improving_high': (5, float('inf'))
        }
        min_val, max_val = static_ranges.get(category, (0, 2))
        category_label = f'{min_val}% to {max_val}%' if max_val != float('inf') else f'≥ {min_val}%'
    else:
        # Parse dynamic category from URL
        # Format: range_X_Y or range_X_inf where X and Y are numbers (neg for negative)
        try:
            parts = category.replace('range_', '').split('_')
            
            # Special handling for exact 0% category
            if parts == ['0', '0']:
                min_val, max_val = 0.0, 0.0
                category_label = '0%'
            else:
                # Parse minimum value
                if parts[0].startswith('neg'):
                    min_val = -int(parts[0][3:])  # Remove 'neg' prefix and convert to negative int
                    remaining_parts = parts[1:]
                else:
                    min_val = int(parts[0]) if parts[0] != 'inf' else -float('inf')
                    remaining_parts = parts[1:]
                
                # Parse maximum value
                if len(remaining_parts) == 0:
                    max_val = float('inf')
                elif remaining_parts[0] == 'inf':
                    max_val = float('inf')
                elif remaining_parts[0].startswith('neg'):
                    max_val = -int(remaining_parts[0][3:])  # Remove 'neg' prefix and convert to negative int
                else:
                    max_val = int(remaining_parts[0])
                    
                # Create category label with special handling for modified ranges
                if max_val == float('inf'):
                    category_label = f'≥ {min_val}%'
                elif min_val == -1 and max_val == 0:
                    category_label = f'-1% to <0%'
                elif min_val == 0 and max_val == 1:
                    category_label = f'>0% to 1%'
                else:
                    category_label = f'{min_val}% to {max_val}%'
                    
        except Exception as e:
            # Fallback to default if parsing fails
            min_val, max_val = 0, 2
            category_label = '0% to 2%'
    
    models = []
    if model_data is not None and len(model_data) > 0:
        # Filter models in the specified range using 'Run time gain [%]'
        if min_val == 0.0 and max_val == 0.0:
            # Special case for exact 0%
            filtered_models = model_data[model_data['Run time gain [%]'] == 0.0]
        elif min_val == -float('inf'):
            filtered_models = model_data[model_data['Run time gain [%]'] < max_val]
        elif max_val == float('inf'):
            filtered_models = model_data[model_data['Run time gain [%]'] >= min_val]
        elif min_val == -1 and max_val == 0:
            # Special case for "-1% to <0%"
            filtered_models = model_data[
                (model_data['Run time gain [%]'] >= min_val) & 
                (model_data['Run time gain [%]'] < 0)
            ]
        elif min_val == 0 and max_val == 1:
            # Special case for ">0% to 1%"
            filtered_models = model_data[
                (model_data['Run time gain [%]'] > 0) & 
                (model_data['Run time gain [%]'] < max_val)
            ]
        else:
            filtered_models = model_data[
                (model_data['Run time gain [%]'] >= min_val) & 
                (model_data['Run time gain [%]'] < max_val)
            ]
        
        # Convert to dict format for template
        models = []
        for _, row in filtered_models.iterrows():
            models.append({
                'model_name': row['Model'],
                'display_name': row['display_name'],
                'name': row.get('Name', 'feature'),
                'device': row.get('Device', 'Unknown'),
                'status': row.get('Status', 'Unknown'),
                'run_time_ms': row.get('Run time [ms]', 0),
                'run_time_gain': row['Run time gain [%]'],
                'run_time_diff_ms': row.get('Run time diff [ms]', 0),
                'compile_time_ms': row.get('Compile time [ms]', 0),
                'compile_time_gain': row.get('Compile time gain [%]', 0),
                'workspace_bytes': row.get('Workspace [bytes]', 0),
                'workspace_gain': row.get('Workspace gain [%]', 0)
            })
        
        # Sort models by runtime gain (ascending - degradations first, then improvements)
        models.sort(key=lambda x: x['run_time_gain'], reverse=False)
    else:
        # Fallback sample data
        sample_models = [
            {'model_name': 'ResNet50', 'display_name': 'ResNet50', 'name': 'feature', 'device': 'gaudi 3', 
             'status': 'PASS', 'run_time_ms': 10.5, 'run_time_gain': 2.3, 'run_time_diff_ms': 0.2, 
             'compile_time_ms': 1200, 'compile_time_gain': 5.2, 'workspace_bytes': 1024000, 'workspace_gain': 3.1},
            {'model_name': 'VGG16', 'display_name': 'VGG16', 'name': 'feature', 'device': 'gaudi 3',
             'status': 'PASS', 'run_time_ms': 15.2, 'run_time_gain': 3.7, 'run_time_diff_ms': 0.5,
             'compile_time_ms': 800, 'compile_time_gain': -2.1, 'workspace_bytes': 2048000, 'workspace_gain': -1.5},
        ]
        models = [m for m in sample_models if min_val <= m['run_time_gain'] < max_val or 
                 (max_val == float('inf') and m['run_time_gain'] >= min_val)]
    
    context = {
        'category_label': category_label,
        'models': models,
        'category': category
    }
    
    return render(request, 'model_list.html', context)


def model_detail(request, model_name):
    """Detail page showing subgraphs for a specific model"""
    _, graph_data = load_csv_data()
    
    subgraphs = []
    if graph_data is not None and len(graph_data) > 0:
        print(f"Looking for model: {model_name}")
        
        # Filter for 'feature' rows only to get actual performance gains
        # Use more efficient filtering
        mask = (graph_data['Model'].str.startswith(model_name, na=False)) & (graph_data['Name'] == 'feature')
        model_graphs = graph_data[mask]
        
        print(f"Found {len(model_graphs)} matching subgraphs")
        
        # Process subgraphs more efficiently
        if len(model_graphs) > 0:
            # Extract subgraph names vectorized
            model_graphs = model_graphs.copy()
            model_graphs['subgraph_name'] = model_graphs['Model'].str.split('.graph_dumps_').str[-1]
            model_graphs['subgraph_name'] = model_graphs['subgraph_name'].fillna('Unknown')
            
            # Convert to list of dicts with all requested columns
            subgraph_columns = ['subgraph_name', 'Name', 'Device', 'Status', 'Run time [ms]', 
                              'Run time gain [%]', 'Run time diff [ms]', 'Compile time [ms]', 
                              'Compile time gain [%]', 'Workspace [bytes]', 'Workspace gain [%]']
            subgraphs = model_graphs[subgraph_columns].to_dict('records')
            
            # Clean up column names for template
            for subgraph in subgraphs:
                subgraph['model_name'] = model_name
                subgraph['name'] = subgraph.pop('Name', 'feature')
                subgraph['device'] = subgraph.pop('Device', 'Unknown')
                subgraph['status'] = subgraph.pop('Status', 'Unknown')
                subgraph['run_time_ms'] = subgraph.pop('Run time [ms]', 0)
                subgraph['run_time_gain'] = subgraph.pop('Run time gain [%]', 0)
                subgraph['run_time_diff_ms'] = subgraph.pop('Run time diff [ms]', 0)
                subgraph['compile_time_ms'] = subgraph.pop('Compile time [ms]', 0)
                subgraph['compile_time_gain'] = subgraph.pop('Compile time gain [%]', 0)
                subgraph['workspace_bytes'] = subgraph.pop('Workspace [bytes]', 0)
                subgraph['workspace_gain'] = subgraph.pop('Workspace gain [%]', 0)
            
            # Sort by runtime gain (ascending - degradations first, then improvements)
            subgraphs.sort(key=lambda x: x['run_time_gain'], reverse=False)
        
        print(f"Processed {len(subgraphs)} subgraphs")
    else:
        print("No graph data available")
        # Fallback sample data
        subgraphs = [
            {'subgraph_name': 'Sample_Graph_1', 'name': 'feature', 'device': 'gaudi 3', 'status': 'PASS',
             'run_time_ms': 0.043, 'run_time_gain': 5.2, 'run_time_diff_ms': 0.002, 'compile_time_ms': 45,
             'compile_time_gain': 12.5, 'workspace_bytes': 512000, 'workspace_gain': 8.3, 'model_name': model_name},
            {'subgraph_name': 'Sample_Graph_2', 'name': 'feature', 'device': 'gaudi 3', 'status': 'PASS',
             'run_time_ms': 0.007, 'run_time_gain': 3.8, 'run_time_diff_ms': 0.0003, 'compile_time_ms': 23,
             'compile_time_gain': -5.2, 'workspace_bytes': 256000, 'workspace_gain': -2.1, 'model_name': model_name},
        ]
    
    context = {
        'model_name': model_name,
        'display_name': model_name.replace('pt_', '').replace('_', ' '),
        'subgraphs': subgraphs
    }
    
    return render(request, 'model_detail.html', context)


def get_category_data(request, category):
    """API endpoint to get models data for a specific category"""
    model_data, _ = load_csv_data()
    
    # Parse dynamic category from URL
    try:
        parts = category.replace('range_', '').split('_')
        
        # Special handling for exact 0% category
        if parts == ['0', '0']:
            min_val, max_val = 0.0, 0.0
        else:
            # Parse minimum value
            if parts[0].startswith('neg'):
                min_val = -int(parts[0][3:])  # Remove 'neg' prefix and convert to negative int
                remaining_parts = parts[1:]
            else:
                min_val = int(parts[0]) if parts[0] != 'inf' else -float('inf')
                remaining_parts = parts[1:]
            
            # Parse maximum value
            if len(remaining_parts) == 0:
                max_val = float('inf')
            elif remaining_parts[0] == 'inf':
                max_val = float('inf')
            elif remaining_parts[0].startswith('neg'):
                max_val = -int(remaining_parts[0][3:])  # Remove 'neg' prefix and convert to negative int
            else:
                max_val = int(remaining_parts[0])
    except Exception as e:
        print(f"Error parsing category {category}: {e}")
        return JsonResponse({'error': 'Invalid category format'}, status=400)
    
    models = []
    if model_data is not None and len(model_data) > 0:
        if min_val == 0.0 and max_val == 0.0:
            # Special case for exact 0%
            filtered_models = model_data[model_data['Run time gain [%]'] == 0.0]
        elif min_val == -float('inf'):
            filtered_models = model_data[model_data['Run time gain [%]'] < max_val]
        elif max_val == float('inf'):
            filtered_models = model_data[model_data['Run time gain [%]'] >= min_val]
        elif min_val == -1 and max_val == 0:
            # Special case for "-1% to <0%"
            filtered_models = model_data[
                (model_data['Run time gain [%]'] >= min_val) & 
                (model_data['Run time gain [%]'] < 0)
            ]
        elif min_val == 0 and max_val == 1:
            # Special case for ">0% to 1%"
            filtered_models = model_data[
                (model_data['Run time gain [%]'] > 0) & 
                (model_data['Run time gain [%]'] < max_val)
            ]
        else:
            filtered_models = model_data[
                (model_data['Run time gain [%]'] >= min_val) & 
                (model_data['Run time gain [%]'] < max_val)
            ]
        
        models = filtered_models[['Model', 'display_name', 'Run time gain [%]']].to_dict('records')
    
    return JsonResponse({'models': models})


def subgraph_index(request, model_name, subgraph_name):
    """Display the index_in_file for a specific subgraph and search for guid"""
    
    index_in_file = -1
    is_supported_model = False
    feature_guid_results = []
    withoutfeature_guid_results = []
    feature_post_json_path = ""
    withoutfeature_post_json_path = ""
    
    # Construct the path to the JSON file using the specified pattern
    feature_dir_path = os.environ["FEATURE_DIR"]
    base_path = os.path.join(feature_dir_path, "asic1/g3_promo_models")
    feature_json_file_path = os.path.join(base_path, "feature", model_name, f"{model_name}.com.json")
    
    print(f"Looking for JSON file at: {feature_json_file_path}")
    
    def extract_guids_from_path(logs_base_path, path_type):
        """Helper function to extract GUIDs from a specific path type"""
        logs_path = os.path.join(logs_base_path, model_name, "logs-compile", f"{model_name}_post-graphs")
        pattern = f"{model_name}.g_{index_in_file}.*.*.post.json"
        full_pattern = os.path.join(logs_path, pattern)
        
        print(f"Searching for {path_type} post.json files with pattern: {full_pattern}")
        
        # Find matching post.json files
        matching_files = glob.glob(full_pattern)
        print(f"Found {len(matching_files)} {path_type} matching files: {matching_files}")
        
        if matching_files:
            # Use the first matching file
            post_json_path = matching_files[0]
            print(f"Using {path_type} post.json file: {post_json_path}")
            
            try:
                with open(post_json_path, 'r') as f:
                    post_data = json.load(f)
                
                # Search for "guid" in the JSON data
                def find_guids(obj, path="root"):
                    """Recursively search for 'guid' keys that are directly under 'nodes'"""
                    results = []
                    if isinstance(obj, dict):
                        for key, value in obj.items():
                            current_path = f"{path}.{key}" if path != "root" else key
                            
                            # Only look for GUIDs if we're directly under a "nodes" key
                            if path.endswith(".nodes") and key.lower() == "guid":
                                results.append(value)
                            elif key.lower() == "nodes" and isinstance(value, (dict, list)):
                                # Found a "nodes" key, search within it for direct GUID children
                                if isinstance(value, dict):
                                    for node_key, node_value in value.items():
                                        if isinstance(node_value, dict):
                                            for attr_key, attr_value in node_value.items():
                                                if attr_key.lower() == "guid":
                                                    results.append(attr_value)
                                elif isinstance(value, list):
                                    for node_item in value:
                                        if isinstance(node_item, dict):
                                            for attr_key, attr_value in node_item.items():
                                                if attr_key.lower() == "guid":
                                                    results.append(attr_value)
                            elif isinstance(value, (dict, list)) and key.lower() != "nodes":
                                # Continue searching for "nodes" keys, but don't look for GUIDs yet
                                results.extend(find_guids(value, current_path))
                    elif isinstance(obj, list):
                        for i, item in enumerate(obj):
                            current_path = f"{path}[{i}]"
                            if isinstance(item, (dict, list)):
                                results.extend(find_guids(item, current_path))
                    return results
                
                all_guids = find_guids(post_data)
                # Get unique GUID values only and sort them alphabetically
                unique_guids = sorted(list(set(all_guids))) if all_guids else []
                print(f"Found {len(unique_guids)} unique {path_type} GUID values (sorted): {unique_guids}")
                
                return unique_guids, post_json_path
                
            except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
                print(f"Error reading {path_type} post.json file {post_json_path}: {e}")
                return [f'error:Error reading file: {str(e)}'], post_json_path
        else:
            print(f"No matching {path_type} post.json files found with pattern: {full_pattern}")
            return [f'error:No post.json file found with specified pattern'], ""
    
    if os.path.exists(feature_json_file_path):
        is_supported_model = True
        try:
            with open(feature_json_file_path, 'r') as f:
                data = json.load(f)
            
            print(f"Successfully loaded JSON data for model: {model_name}")
            
            # Look for the subgraph in the compile section
            if 'compile' in data and 'graphs' in data['compile']:
                graphs = data['compile']['graphs']
                print(f"Found {len(graphs)} graphs in compile section")
                
                # Search for the subgraph by name
                for graph_id, graph_info in graphs.items():
                    # Extract the subgraph name from the full path
                    graph_name = graph_info.get('name', '').split('/')[-1]  # Get the last part after '/'
                    
                    # Check if this matches our subgraph
                    if graph_name == subgraph_name or subgraph_name in graph_name:
                        index_in_file = graph_info.get('index_in_file', -1)
                        print(f"Found matching subgraph {subgraph_name} with index_in_file: {index_in_file}")
                        break
                    
                    # Also check if the subgraph_name matches the recipe file pattern
                    recipe_file = graph_info.get('recipe_file', '')
                    if subgraph_name in recipe_file:
                        index_in_file = graph_info.get('index_in_file', -1)
                        print(f"Found matching subgraph via recipe file {subgraph_name} with index_in_file: {index_in_file}")
                        break
                
                if index_in_file == -1:
                    print(f"Could not find subgraph {subgraph_name} in the graphs data")
                else:
                    # Extract GUIDs from both feature and withoutfeature paths
                    feature_base_path = os.path.join(base_path, "feature")
                    withoutfeature_base_path = os.path.join(base_path, "withoutfeature")
                    
                    # Get feature GUIDs
                    feature_guid_results, feature_post_json_path = extract_guids_from_path(feature_base_path, "feature")
                    
                    # Get withoutfeature GUIDs
                    withoutfeature_guid_results, withoutfeature_post_json_path = extract_guids_from_path(withoutfeature_base_path, "withoutfeature")
                    
            else:
                print("No compile section or graphs found in JSON data")
                        
        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            print(f"Error reading JSON file {feature_json_file_path}: {e}")
            index_in_file = -1
            is_supported_model = False
    else:
        print(f"JSON file not found: {feature_json_file_path}")
        # For models without JSON files, return -1 as specified
        index_in_file = -1
        is_supported_model = False
    
    context = {
        'model_name': model_name,
        'subgraph_name': subgraph_name,
        'index_in_file': index_in_file,
        'is_supported_model': is_supported_model,
        'json_file_path': feature_json_file_path,
        'feature_post_json_path': feature_post_json_path,
        'withoutfeature_post_json_path': withoutfeature_post_json_path,
        'feature_guid_results': feature_guid_results,
        'withoutfeature_guid_results': withoutfeature_guid_results
    }
    
    return render(request, 'subgraph_index.html', context)


def mlir_content(request, model_name, fused_kernel_name, source_type):
    """Display the content of MLIR file for a specific fused kernel"""
    
    # Base paths for both feature and withoutfeature directories
    base_path = "/software/users/ksonawane/tpcfuser/feature/fusion_before_norm/asic1/g3_promo_models"
    
    # Determine which path to search based on source_type
    if source_type == "withoutfeature":
        logs_compile_path = os.path.join(base_path, "withoutfeature", model_name, "logs-compile")
        search_label = "WITHOUT Feature"
    else:  # Default to feature for any other value (including "feature")
        logs_compile_path = os.path.join(base_path, "feature", model_name, "logs-compile")
        search_label = "WITH Feature"
        source_type = "feature"  # Normalize the source_type
    
    # Pattern for MLIR file: <MODEL_NAME>_fusergraph-*-<fused_kernel_name>.*.mlir
    pattern = f"{model_name}_fusergraph-*-{fused_kernel_name}.*.mlir"
    full_pattern = os.path.join(logs_compile_path, pattern)
    
    print(f"Searching for MLIR file in {search_label} path: {full_pattern}")
    
    # Find matching MLIR files in the specified location only
    matching_files = glob.glob(full_pattern)
    print(f"Found {len(matching_files)} MLIR files: {matching_files}")
    
    mlir_content_text = ""
    mlir_file_path = ""
    file_found = False
    error_message = ""
    
    if matching_files:
        # Use the first matching file
        mlir_file_path = matching_files[0]
        print(f"Using MLIR file from {search_label}: {mlir_file_path}")
        
        try:
            with open(mlir_file_path, 'r', encoding='utf-8') as f:
                mlir_content_text = f.read()
            file_found = True
            print(f"Successfully read MLIR file, content length: {len(mlir_content_text)} characters")
            
        except (FileNotFoundError, PermissionError, UnicodeDecodeError) as e:
            print(f"Error reading MLIR file {mlir_file_path}: {e}")
            error_message = f"Error reading file: {str(e)}"
            file_found = False
    else:
        print(f"No matching MLIR files found in {search_label} directory")
        error_message = f"No MLIR file found for fused kernel: {fused_kernel_name} in {search_label} path"
        file_found = False
    
    # Get file stats if file exists
    file_stats = {}
    if file_found and mlir_file_path:
        try:
            stat = os.stat(mlir_file_path)
            file_stats = {
                'size': stat.st_size,
                'modified': stat.st_mtime,
                'lines': len(mlir_content_text.split('\n')) if mlir_content_text else 0
            }
        except Exception as e:
            print(f"Error getting file stats: {e}")
    
    context = {
        'model_name': model_name,
        'fused_kernel_name': fused_kernel_name,
        'mlir_content': mlir_content_text,
        'mlir_file_path': mlir_file_path,
        'file_found': file_found,
        'error_message': error_message,
        'file_stats': file_stats,
        'source_type': source_type,
        'search_label': search_label,
        'pattern_searched': pattern,
        'search_path': logs_compile_path
    }
    
    return render(request, 'mlir_content.html', context)
