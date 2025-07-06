import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State # Updated import
import plotly.express as px
import pandas as pd
from data_generator import generate_sensor_data

# Generate or load data
# In a real application, you might load this from a file or database
print("Generating initial sensor data, please wait...")
sensor_df = generate_sensor_data()
print("Sensor data generation complete. Application is ready.")

import plotly.graph_objects as go # Import go for empty figure

# Initialize the Dash app
app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    html.H1("Sensor Waveform Viewer"),

    html.Div([
        html.Label("Select Location ID:"),
        dcc.Dropdown(
            id='location-dropdown',
            options=[{'label': 'All Locations', 'value': 'all'}] + [{'label': loc, 'value': loc} for loc in sensor_df['location_id'].unique()],
            value='all',  # Default to "All Locations"
            clearable=False
        ),
    ], style={'width': '48%', 'display': 'inline-block'}),

    html.Div([
        html.Label("Select Sensor ID(s):"),
        dcc.Dropdown(
            id='sensor-dropdown',
            # Options will be populated by callback, will include 'all' option
            options=[],
            value=[],  # Default value (empty list)
            multi=True
        ),
    ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),

    # Moved Waveform Graph to be above Feature Extraction settings
    dcc.Graph(id='waveform-graph'),
    html.Hr(),

    html.Div([
        html.H3("Feature Extraction Settings"),
        html.Div([
            html.Label("Select Features:"),
            dcc.Checklist(
                id='feature-checklist',
                options=[
                    {'label': 'Mean', 'value': 'mean'},
                    {'label': 'Standard Deviation', 'value': 'std'},
                    {'label': 'Maximum', 'value': 'max'},
                    {'label': 'Minimum', 'value': 'min'},
                    {'label': 'RMS', 'value': 'rms'},
                    {'label': 'Segmented Features', 'value': 'segmented_features'}
                ],
                value=[], # Default no features selected
                labelStyle={'display': 'block'} # Display options vertically
            ),
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}),

        html.Div([
            html.Label("Number of Segments (N):"),
            dcc.Input(
                id='n-segments-input',
                type='number',
                placeholder='Enter N for segments',
                value=5, # Default N value
                min=1, # Minimum N value
                step=1, # Increment step
                style={'display': 'block'} # Initially hidden, shown by callback
            ),
        ], id='n-segments-input-div', style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '20px'}), # Hidden by default

        html.Div([
            html.Button('Extract Features', id='extract-features-button', n_clicks=0, style={'marginTop': '20px'}),
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'bottom', 'marginLeft': '20px'}),

    ], style={'marginTop': '20px'}),

    # Placeholder for feature table output
    html.Div(id='feature-table-output'),

    html.Hr(),
    html.H3("Feature Visualization"),
    html.Div([
        html.Div([
            html.Label("X-axis Feature:"),
            dcc.Dropdown(id='x-axis-feature-dropdown', options=[], placeholder="Select X-axis feature")
        ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '10px'}),
        html.Div([
            html.Label("Y-axis Feature (for Scatter Plot):"),
            dcc.Dropdown(id='y-axis-feature-dropdown', options=[], placeholder="Select Y-axis feature")
        ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '10px'}),
        html.Div([
            html.Label("Graph Type:"),
            dcc.Dropdown(
                id='feature-graph-type-dropdown',
                options=[
                    {'label': 'Histogram', 'value': 'histogram'},
                    {'label': 'Scatter Plot', 'value': 'scatter'}
                ],
                value='histogram' # Default graph type
            )
        ], style={'width': '30%', 'display': 'inline-block'}),
    ]),
    dcc.Graph(id='feature-visualization-graph'),
    html.Div(id='feature-stats-output', style={'marginTop': '20px'}), # Div for stats output
    html.Button("Download Extracted Features as CSV", id="btn-download-csv", style={'marginTop': '20px'}),
    dcc.Download(id="download-dataframe-csv"),

    html.Hr(style={'marginTop': '30px', 'marginBottom': '30px'}),
    html.H3("Feature Correlation Heatmap"),
    dcc.Graph(id='feature-correlation-heatmap'),

    # dcc.Graph(id='waveform-graph'), # This was moved up
    dcc.Store(id='extracted-features-store') # To store extracted features data

])

# --- Import feature extractor ---
from feature_extractor import extract_features as calculate_waveform_features, numerize_result # Renamed for clarity
from dash import dash_table # Import DataTable

# Callback to show/hide N Segments input based on "Segmented Features" selection
@app.callback(
    Output('n-segments-input-div', 'style'),
    [Input('feature-checklist', 'value')]
)
def toggle_n_segments_input(selected_features):
    base_style = {'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '20px'}
    if selected_features and 'segmented_features' in selected_features: # Check if selected_features is not None
        base_style['display'] = 'inline-block' # Show
    else:
        base_style['display'] = 'none' # Hide
    return base_style

# Callback to extract features and store them
@app.callback(
    Output('extracted-features-store', 'data'),
    Output('feature-table-output', 'children'), # For now, just show a message or raw data
    [Input('extract-features-button', 'n_clicks')],
    [State('location-dropdown', 'value'),
     State('sensor-dropdown', 'value'),
     State('feature-checklist', 'value'),
     State('n-segments-input', 'value'),
     State('extracted-features-store', 'data')] # Keep existing data if no sensors selected
)
def handle_feature_extraction(n_clicks, selected_location, selected_sensor_ids,
                              selected_features_names, n_segments, existing_store_data):
    if n_clicks == 0 or not selected_location or not selected_sensor_ids or not selected_features_names:
        # No button click or essential inputs missing, return existing or empty data
        # Display a message if button was clicked but inputs were missing.
        if n_clicks > 0: # Button was clicked
            if not selected_location:
                return existing_store_data, html.P("Please select a location.", style={'color': 'red'})
            if not selected_sensor_ids:
                return existing_store_data, html.P("Please select at least one sensor.", style={'color': 'red'})
            if not selected_features_names:
                return existing_store_data, html.P("Please select at least one feature to extract.", style={'color': 'red'})
        return existing_store_data, "Feature data will appear here after extraction."


    # Get the actual waveform data for selected sensors

    # Determine the list of sensor IDs to process based on 'all' selection
    sensors_to_process = []
    if 'all' in selected_sensor_ids:
        if selected_location == 'all':
            sensors_to_process = list(sensor_df['sensor_id'].unique())
        else: # 'all' sensors for a specific location
            sensors_to_process = list(sensor_df[sensor_df['location_id'] == selected_location]['sensor_id'].unique())
    else: # Specific sensors selected
        sensors_to_process = [sid for sid in selected_sensor_ids if sid != 'all']

    # Filter the main sensor_df based on selected_location (if not 'all')
    if selected_location == 'all':
        wave_df_for_extraction = sensor_df[sensor_df['sensor_id'].isin(sensors_to_process)]
    else:
        wave_df_for_extraction = sensor_df[
            (sensor_df['location_id'] == selected_location) &
            (sensor_df['sensor_id'].isin(sensors_to_process))
        ]

    if wave_df_for_extraction.empty:
        return [], html.P("No waveform data found for selected sensors and location to extract features.", style={'color': 'orange'})

    all_extracted_features = []
    # Iterate over the unique sensor_ids present in the filtered wave_df_for_extraction
    for s_id in wave_df_for_extraction['sensor_id'].unique():
        sensor_waveform_data = wave_df_for_extraction[wave_df_for_extraction['sensor_id'] == s_id]['value']
        # Need to get the correct location_id if selected_location was 'all'
        actual_loc_id = wave_df_for_extraction[wave_df_for_extraction['sensor_id'] == s_id]['location_id'].iloc[0]

        if sensor_waveform_data.empty:
            continue

        # Call the feature extraction function from feature_extractor.py
        current_n_segments = n_segments if selected_features_names and 'segmented_features' in selected_features_names else None

        if selected_features_names and 'segmented_features' in selected_features_names:
            if current_n_segments is None or not isinstance(current_n_segments, int) or current_n_segments <= 0:
                # This message might be better placed near the input or as a general validation message.
                return existing_store_data, html.P(f"Invalid number of segments (N={current_n_segments}). Please provide a positive integer for N.", style={'color': 'red'})

        features = calculate_waveform_features(
            series=sensor_waveform_data,
            selected_feature_names=selected_features_names,
            n_segments=current_n_segments
        )

        # Add identifiers
        features['sensor_id'] = s_id
        features['location_id'] = actual_loc_id # Use the actual location ID for this sensor
        features['result'] = wave_df_for_extraction[wave_df_for_extraction['sensor_id'] == s_id]['result'].iloc[0]
        all_extracted_features.append(features)

    if not all_extracted_features:
        return [], html.P("Could not extract features for the selected sensors/settings.", style={'color': 'orange'})

    # Convert to DataFrame for easier handling later, then to dict for store
    features_df = pd.DataFrame(all_extracted_features)

    if not features_df.empty:
        # Add numeric result column for potential correlation analysis
        features_df['result_numeric'] = numerize_result(features_df['result'])

    # Prepare data for DataTable
    if not features_df.empty:
        # Reorder columns for better display in table: identifiers, result, result_numeric, then sorted feature columns
        id_cols = ['location_id', 'sensor_id', 'result', 'result_numeric']
        # Ensure these ID columns actually exist in the df before trying to use them
        present_id_cols = [col for col in id_cols if col in features_df.columns]

        feature_col_names = sorted([col for col in features_df.columns if col not in present_id_cols])
        ordered_columns = present_id_cols + feature_col_names
        features_df_ordered = features_df[ordered_columns]

        datatable_output = dash_table.DataTable(
            id='interactive-feature-table',
            columns=[{"name": i, "id": i, "deletable": False, "selectable": True} for i in features_df_ordered.columns],
            data=features_df_ordered.to_dict('records'),
            editable=False,
            filter_action="native",
            sort_action="native",
            sort_mode="multi",
            row_selectable="multi", # or "single"
            row_deletable=False,
            selected_rows=[],
            page_action="native",
            page_current=0,
            page_size=10, # Show 10 rows per page
            style_table={'overflowX': 'auto', 'marginTop': '20px'}, # Horizontal scroll for many features
            style_cell={
                'height': 'auto',
                'minWidth': '90px', 'width': '120px', 'maxWidth': '180px',
                'whiteSpace': 'normal',
                'textAlign': 'left'
            },
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            }
        )
    else:
        datatable_output = html.P("No features extracted or data is empty.", style={'marginTop': '20px'})

    return features_df.to_dict('records'), datatable_output

# Callback to update feature dropdowns based on stored data
@app.callback(
    Output('x-axis-feature-dropdown', 'options'),
    Output('y-axis-feature-dropdown', 'options'),
    Output('x-axis-feature-dropdown', 'value'), # Reset value when options change
    Output('y-axis-feature-dropdown', 'value'), # Reset value when options change
    [Input('extracted-features-store', 'data')]
)
def update_feature_plot_dropdowns(stored_feature_data):
    if not stored_feature_data:
        return [], [], None, None

    df = pd.DataFrame(stored_feature_data)
    # Exclude non-numeric or identifier columns from feature selection
    # 'result' is categorical, 'result_numeric' can be used as a feature if desired
    potential_features = [col for col in df.columns if col not in ['sensor_id', 'location_id', 'result']]
    options = [{'label': col, 'value': col} for col in potential_features]

    # Set default values if options are available, otherwise None
    default_x = options[0]['value'] if options else None
    default_y = options[1]['value'] if len(options) > 1 else None

    return options, options, default_x, default_y

# Callback to generate feature visualization graph
@app.callback(
    Output('feature-visualization-graph', 'figure'),
    Output('feature-stats-output', 'children'), # Add output for stats
    [Input('extracted-features-store', 'data'),
     Input('x-axis-feature-dropdown', 'value'),
     Input('y-axis-feature-dropdown', 'value'),
     Input('feature-graph-type-dropdown', 'value')]
)
def update_feature_graph_and_stats(stored_feature_data, x_feature, y_feature, graph_type):
    empty_figure = go.Figure()
    empty_figure.update_layout(title_text="Extract features and select X-axis to generate a graph.",
                               xaxis=dict(showgrid=False, zeroline=False, visible=True, range=[0,1]),
                               yaxis=dict(showgrid=False, zeroline=False, visible=True, range=[0,1]))
    empty_stats = html.P("")

    if not stored_feature_data or not x_feature:
        return empty_figure, empty_stats

    df = pd.DataFrame(stored_feature_data)

    if x_feature not in df.columns:
        fig_error = go.Figure()
        fig_error.update_layout(title_text=f"X-axis feature '{x_feature}' not found in extracted data.",
                                xaxis=dict(showgrid=False, zeroline=False, visible=True, range=[0,1]),
                                yaxis=dict(showgrid=False, zeroline=False, visible=True, range=[0,1]))
        return fig_error, empty_stats
    if graph_type == 'scatter' and (not y_feature or y_feature not in df.columns):
        fig_error = go.Figure()
        fig_error.update_layout(title_text=f"Y-axis feature '{y_feature}' not found or not selected for scatter plot.",
                                xaxis=dict(showgrid=False, zeroline=False, visible=True, range=[0,1]),
                                yaxis=dict(showgrid=False, zeroline=False, visible=True, range=[0,1]))
        return fig_error, empty_stats

    fig_title = ""
    fig = empty_figure # Initialize fig with a valid empty figure
    stats_output = []

    try:
        if graph_type == 'histogram':
            fig_title = f"Histogram of {x_feature}"
            fig = px.histogram(df, x=x_feature, color='result',
                               marginal="box", barmode='overlay', title=fig_title,
                               color_discrete_map={"OK": "green", "NG": "red"})

            # Calculate and display stats for histogram's x_feature
            stats_output.append(html.H5(f"Descriptive Statistics for '{x_feature}':"))
            ok_data = df[(df['result'] == 'OK') & pd.notna(df[x_feature])][x_feature]
            ng_data = df[(df['result'] == 'NG') & pd.notna(df[x_feature])][x_feature]

            if not ok_data.empty:
                stats_output.append(html.H6("OK Group:"))
                stats_output.append(html.Pre(ok_data.describe().to_string()))
            if not ng_data.empty:
                stats_output.append(html.H6("NG Group:"))
                stats_output.append(html.Pre(ng_data.describe().to_string()))
            if ok_data.empty and ng_data.empty:
                stats_output.append(html.P("No data available for statistics."))

        elif graph_type == 'scatter':
            if not y_feature:
                 return px.line(title="Please select a Y-axis feature for the scatter plot."), empty_stats
            fig_title = f"Scatter Plot: {x_feature} vs {y_feature}"
            fig = px.scatter(df, x=x_feature, y=y_feature, color='result', title=fig_title,
                             color_discrete_map={"OK": "green", "NG": "red"},
                             hover_data=['sensor_id', 'location_id'])
            # For scatter, could show stats for X and Y separately if desired, or correlation.
            # For now, let's keep it simple and only show stats if histogram is chosen.
            stats_output.append(html.P("Descriptive statistics are shown for Histogram view."))
        else:
            fig = go.Figure()
            fig.update_layout(title_text="Invalid graph type selected.",
                              xaxis=dict(showgrid=False, zeroline=False, visible=True, range=[0,1]),
                              yaxis=dict(showgrid=False, zeroline=False, visible=True, range=[0,1]))
            stats_output = html.P("Select a valid graph type.")

    except Exception as e:
        error_message = f"Error generating {graph_type} for {x_feature}" + \
                        (f" vs {y_feature}" if y_feature and graph_type == 'scatter' else "") + \
                        f". Check data. Error: {str(e)}"
        fig_error = go.Figure()
        fig_error.update_layout(title_text=error_message,
                                xaxis=dict(showgrid=False, zeroline=False, visible=True, range=[0,1]),
                                yaxis=dict(showgrid=False, zeroline=False, visible=True, range=[0,1]))
        return fig_error, html.P(error_message, style={'color': 'red'})

    fig.update_layout(legend_title_text='Result')
    return fig, html.Div(stats_output)

# Callback to generate feature correlation heatmap
@app.callback(
    Output('feature-correlation-heatmap', 'figure'),
    [Input('extracted-features-store', 'data')]
)
def update_correlation_heatmap(stored_feature_data):
    if not stored_feature_data:
        fig_empty = go.Figure()
        fig_empty.update_layout(title_text="Extract features to view correlation heatmap.",
                                xaxis=dict(showgrid=False, zeroline=False, visible=False),
                                yaxis=dict(showgrid=False, zeroline=False, visible=False))
        return fig_empty


    df = pd.DataFrame(stored_feature_data)

    # Select only numeric columns for correlation, including 'result_numeric'
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols or len(numeric_cols) < 2: # Need at least two numeric columns for correlation
        fig_empty_numeric = go.Figure()
        fig_empty_numeric.update_layout(title_text="Not enough numeric features for correlation heatmap.",
                                        xaxis=dict(showgrid=False, zeroline=False, visible=False),
                                        yaxis=dict(showgrid=False, zeroline=False, visible=False))
        return fig_empty_numeric

    correlation_matrix = df[numeric_cols].corr()

    fig = px.imshow(
        correlation_matrix,
        text_auto=True, # Show correlation values on the heatmap
        aspect="auto",  # Adjust aspect ratio
        color_continuous_scale='RdBu_r', # Red-Blue diverging scale, good for correlations
        zmin=-1, zmax=1, # Fix the color scale range
        title="Feature Correlation Heatmap (including Result_Numeric)"
    )
    fig.update_layout(height=600) # Adjust height as needed
    return fig


# Callback for CSV download
@app.callback(
    Output("download-dataframe-csv", "data"),
    [Input("btn-download-csv", "n_clicks")],
    [State('extracted-features-store', 'data')],
    prevent_initial_call=True, # Important to prevent download on app load
)
def download_csv(n_clicks, stored_feature_data):
    if not stored_feature_data:
        # Or raise PreventUpdate if you don't want to send anything if no data
        return None

    df = pd.DataFrame(stored_feature_data)
    if df.empty:
        return None

    # Ensure columns are in a somewhat logical order for CSV
    # Put identifiers first, then result, then feature columns
    id_cols = ['location_id', 'sensor_id', 'result', 'result_numeric'] # Added result_numeric
    feature_cols = [col for col in df.columns if col not in id_cols]

    # Handle cases where some id_cols might be missing (though unlikely with current setup)
    final_cols = [col for col in id_cols if col in df.columns]
    final_cols.extend(sorted(feature_cols)) # Sort feature columns alphabetically for consistency

    df_to_download = df[final_cols]

    return dcc.send_data_frame(df_to_download.to_csv, "extracted_features.csv", index=False)

# Callback to update sensor dropdown based on location
@app.callback(
    Output('sensor-dropdown', 'options'),
    Output('sensor-dropdown', 'value'), # Also reset selected sensors when location changes
    [Input('location-dropdown', 'value')]
)
def update_sensor_options(selected_location):
    if not selected_location:
        return [], []

    options = []
    if selected_location == 'all':
        # If 'All Locations' is selected, get all unique sensor IDs from the entire dataframe
        all_sensor_ids = sorted(sensor_df['sensor_id'].unique())
        options = [{'label': 'All Sensors', 'value': 'all'}] + [{'label': sid, 'value': sid} for sid in all_sensor_ids]
    else:
        # Filter sensors based on the specific selected location
        filtered_sensors = sorted(sensor_df[sensor_df['location_id'] == selected_location]['sensor_id'].unique())
        options = [{'label': 'All Sensors', 'value': 'all'}] + [{'label': sid, 'value': sid} for sid in filtered_sensors]

    # When location changes, reset the selected sensors to none (or to 'all' by default if preferred)
    return options, []


# Callback to update graph
@app.callback(
    Output('waveform-graph', 'figure'),
    [Input('location-dropdown', 'value'),
     Input('sensor-dropdown', 'value')]
)
def update_graph(selected_location, selected_sensor_ids):
    if not selected_location or not selected_sensor_ids:
        # Return an empty figure using graph_objects for better stability
        fig = go.Figure()
        fig.update_layout(
            title_text="Please select a location and one or more sensors.",
            xaxis_title="Timestamp",
            yaxis_title="Value",
            # Ensure axis are visible even if empty by setting range or dummy invisible trace
            xaxis=dict(showgrid=False, zeroline=False, visible=True, range=[0,1]),
            yaxis=dict(showgrid=False, zeroline=False, visible=True, range=[0,1])
        )
        return fig

    query_parts = []

    # Handle location filtering
    if selected_location != 'all':
        query_parts.append(f"location_id == '{selected_location}'")

    # Handle sensor filtering
    actual_sensor_ids_to_filter = []
    if 'all' in selected_sensor_ids:
        if selected_location == 'all':
            # 'All Sensors' in 'All Locations' means no sensor ID filtering needed at this stage,
            # or rather, get all sensor IDs present in the current (potentially location-filtered) df.
            # This case is complex if we want to show ALL sensors from ALL locations.
            # For simplicity, if 'all' sensors is chosen, we rely on location filter only,
            # or if location is also 'all', then no filter.
            # Let's refine: if 'all' sensors, then get all sensors for the given location context.
            if selected_location == 'all':
                 pass # No specific sensor ID filtering, all sensors from all locations
            else: # 'all' sensors for a specific location
                actual_sensor_ids_to_filter = list(sensor_df[sensor_df['location_id'] == selected_location]['sensor_id'].unique())
        else: # 'all' sensors for a specific location (already handled by selected_location != 'all')
             actual_sensor_ids_to_filter = list(sensor_df[sensor_df['location_id'] == selected_location]['sensor_id'].unique())

    else: # Specific sensor IDs are selected (and 'all' is not one of them)
        actual_sensor_ids_to_filter = [sid for sid in selected_sensor_ids if sid != 'all']

    if actual_sensor_ids_to_filter: # If there are specific sensors to filter by
         query_parts.append(f"sensor_id in {actual_sensor_ids_to_filter}")


    # Build the query string
    if query_parts:
        current_df = sensor_df.query(" and ".join(query_parts))
    else: # No location and no specific sensors selected (e.g. "All Locations" and "All Sensors")
        current_df = sensor_df.copy()

    if current_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title_text="No data found for selected criteria.",
            xaxis_title="Timestamp",
            yaxis_title="Value",
            xaxis=dict(showgrid=False, zeroline=False, visible=True, range=[0,1]),
            yaxis=dict(showgrid=False, zeroline=False, visible=True, range=[0,1])
        )
        return fig

    # Limit data points if too many sensors are selected to avoid performance issues
    # For example, if more than 10 sensors, maybe take a sample or warn user.
    # This is a placeholder for a more sophisticated handling if needed.
    MAX_SENSORS_TO_PLOT_INDIVIDUALLY = 30 # Adjust as needed
    if current_df['sensor_id'].nunique() > MAX_SENSORS_TO_PLOT_INDIVIDUALLY:
        # Potentially, instead of erroring, one could plot an aggregation or a subset.
        # For now, let's allow it but be mindful of browser performance.
        # Or, we could restrict selection in the UI to prevent this.
        pass

    # Create a title that lists the selected location and sensors with their results
    title_parts = []
    # Use unique sensor_ids from the filtered dataframe for the title
    # Limit the number of sensors in title to avoid excessively long titles
    sensors_in_title = current_df['sensor_id'].unique() # Corrected from selected_sensor_ids
    if len(sensors_in_title) > 5:
        # Get results for the first 3 sensors shown in title
        for sid in sensors_in_title[:3]:
            result = current_df[current_df['sensor_id'] == sid]['result'].iloc[0]
            title_parts.append(f"{sid} ({result})")
        title_parts.append(f"...and {len(sensors_in_title) - 3} more sensors")
    else:
        for sid in sensors_in_title: # Iterate over actual sensors in current_df
            result = current_df[current_df['sensor_id'] == sid]['result'].iloc[0]
            title_parts.append(f"{sid} ({result})")

    location_str = selected_location if selected_location != 'all' else "All Locations"
    # Use actual_sensor_ids_to_filter or sensors_in_title to determine sensor part of title
    if 'all' in selected_sensor_ids and not actual_sensor_ids_to_filter : # true if 'all' sensors for 'all' locations, or 'all' for specific location
        sensors_str = "All Sensors"
    elif not actual_sensor_ids_to_filter and selected_location == 'all': # Should be covered by above
        sensors_str = "All Sensors"
    elif not sensors_in_title.any(): # No sensors in current_df
        sensors_str = "None (or no data)"
    else:
        sensors_str = ", ".join(title_parts)

    title = f"Waveforms for Location {location_str} - Sensors: {sensors_str}"


    # Define the color map for sensor_id based on their 'result'
    color_discrete_map = {}
    for sensor_id_val in current_df['sensor_id'].unique(): # Use unique sensors from the current_df
        result_status = current_df[current_df['sensor_id'] == sensor_id_val]['result'].iloc[0]
        if result_status == "OK":
            color_discrete_map[sensor_id_val] = "green"
        elif result_status == "NG":
            color_discrete_map[sensor_id_val] = "red"
        else:
            color_discrete_map[sensor_id_val] = "blue" # Fallback for unexpected results

    # Create the line plot
    fig = px.line(current_df,  # Corrected from filtered_df
                  x='timestamp',
                  y='value',
                  color='sensor_id', # Color lines by sensor_id
                  title=title,
                  labels={'sensor_id': 'Sensor ID', 'value': 'Value', 'timestamp': 'Timestamp'},
                  color_discrete_map=color_discrete_map) # Apply the custom color map

    fig.update_layout(
        xaxis_title="Timestamp",
        yaxis_title="Value",
        legend_title="Sensor ID"
    )
    return fig

if __name__ == '__main__':
    # Note: Setting debug=False for production or if issues arise with reloader
    # For development, debug=True is fine.
    app.run(debug=True, host='127.0.0.1', port=8050) # Changed from app.run_server and host

#TODO: Next steps from the plan:
# 4. 特徴量可視化機能の実装
#    - `app.py` のレイアウトに、特徴量可視化のためのグラフエリアと、表示する特徴量を選択するドロップダウン（X軸、Y軸、グラフタイプ選択など）を追加します。
#    - `dcc.Store` に格納された特徴量データに基づいて、選択された特徴量の散布図やヒストグラムを生成するコールバックを実装します。プロットは `result` ('OK'/'NG') で色分けします。
# 5. CSVエクスポート機能の実装
