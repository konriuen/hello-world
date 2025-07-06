import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
from data_generator import generate_sensor_data

# Generate or load data
# In a real application, you might load this from a file or database
sensor_df = generate_sensor_data()

# Initialize the Dash app
app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    html.H1("Sensor Waveform Viewer"),

    html.Div([
        html.Label("Select Location ID:"),
        dcc.Dropdown(
            id='location-dropdown',
            options=[{'label': loc, 'value': loc} for loc in sensor_df['location_id'].unique()],
            value=sensor_df['location_id'].unique()[0],  # Default to the first location
            clearable=False
        ),
    ], style={'width': '48%', 'display': 'inline-block'}),

    html.Div([
        html.Label("Select Sensor ID(s):"),
        dcc.Dropdown(
            id='sensor-dropdown',
            # Options will be populated by callback
            options=[],
            value=[],  # Default value (empty list)
            multi=True  # Allow multiple selections
        ),
    ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),

    html.Hr(), # Horizontal line to separate sections

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

    # Placeholder for feature visualization and export (to be added in later steps)
    html.Div(id='feature-table-output'), # For displaying extracted features (e.g. in a table)

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
    html.Button("Download Extracted Features as CSV", id="btn-download-csv", style={'marginTop': '20px'}),
    dcc.Download(id="download-dataframe-csv"),

    dcc.Graph(id='waveform-graph'), # Original waveform graph
    dcc.Store(id='extracted-features-store') # To store extracted features data

])

# --- Import feature extractor ---
from feature_extractor import extract_features as calculate_waveform_features # Renamed for clarity

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
    wave_df = sensor_df[
        (sensor_df['location_id'] == selected_location) &
        (sensor_df['sensor_id'].isin(selected_sensor_ids))
    ]

    if wave_df.empty:
        return [], html.P("No waveform data found for selected sensors.", style={'color': 'orange'})

    all_extracted_features = []
    for sensor_id in selected_sensor_ids:
        sensor_waveform_data = wave_df[wave_df['sensor_id'] == sensor_id]['value']
        if sensor_waveform_data.empty:
            continue

        # Call the feature extraction function from feature_extractor.py
        # Ensure n_segments is passed correctly if 'segmented_features' is chosen
        current_n_segments = n_segments if selected_features_names and 'segmented_features' in selected_features_names else None

        # Validate n_segments if 'segmented_features' is selected
        if selected_features_names and 'segmented_features' in selected_features_names:
            if current_n_segments is None or not isinstance(current_n_segments, int) or current_n_segments <= 0:
                return existing_store_data, html.P(f"Invalid number of segments (N={current_n_segments}). Please provide a positive integer for N.", style={'color': 'red'})

        features = calculate_waveform_features(
            series=sensor_waveform_data,
            selected_feature_names=selected_features_names,
            n_segments=current_n_segments
        )

        # Add identifiers
        features['sensor_id'] = sensor_id
        features['location_id'] = selected_location
        features['result'] = wave_df[wave_df['sensor_id'] == sensor_id]['result'].iloc[0]
        all_extracted_features.append(features)

    if not all_extracted_features:
        return [], html.P("Could not extract features for the selected sensors/settings.", style={'color': 'orange'})

    # For now, display a simple message or the raw JSON in the table-output div
    # In a later step, this could be a Dash DataTable
    # Convert to DataFrame for easier handling later, then to dict for store
    features_df = pd.DataFrame(all_extracted_features)

    # Temporary display for verification
    table_preview = html.Div([
        html.H4("Extracted Features (Preview)"),
        html.Pre(features_df.to_string()) # Simple string representation
    ])

    return features_df.to_dict('records'), table_preview

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
    potential_features = [col for col in df.columns if col not in ['sensor_id', 'location_id', 'result']]
    options = [{'label': col, 'value': col} for col in potential_features]

    # Set default values if options are available, otherwise None
    default_x = options[0]['value'] if options else None
    default_y = options[1]['value'] if len(options) > 1 else None

    return options, options, default_x, default_y

# Callback to generate feature visualization graph
@app.callback(
    Output('feature-visualization-graph', 'figure'),
    [Input('extracted-features-store', 'data'),
     Input('x-axis-feature-dropdown', 'value'),
     Input('y-axis-feature-dropdown', 'value'),
     Input('feature-graph-type-dropdown', 'value')]
)
def update_feature_graph(stored_feature_data, x_feature, y_feature, graph_type):
    if not stored_feature_data or not x_feature:
        return px.line(title="Extract features and select X-axis to generate a graph.")

    df = pd.DataFrame(stored_feature_data)

    # Ensure the selected features exist in the dataframe
    if x_feature not in df.columns:
        return px.line(title=f"X-axis feature '{x_feature}' not found in extracted data.")
    if graph_type == 'scatter' and (not y_feature or y_feature not in df.columns):
        return px.line(title=f"Y-axis feature '{y_feature}' not found or not selected for scatter plot.")

    fig_title = ""
    try:
        if graph_type == 'histogram':
            fig_title = f"Histogram of {x_feature}"
            fig = px.histogram(df, x=x_feature, color='result',
                               marginal="box", # or violin, rug
                               barmode='overlay', # To see distributions overlapped
                               title=fig_title,
                               color_discrete_map={"OK": "green", "NG": "red"})
        elif graph_type == 'scatter':
            if not y_feature: # Should be caught above, but as a safeguard
                 return px.line(title="Please select a Y-axis feature for the scatter plot.")
            fig_title = f"Scatter Plot: {x_feature} vs {y_feature}"
            fig = px.scatter(df, x=x_feature, y=y_feature, color='result',
                             title=fig_title,
                             color_discrete_map={"OK": "green", "NG": "red"},
                             hover_data=['sensor_id', 'location_id'])
        else:
            fig = px.line(title="Invalid graph type selected.")
    except Exception as e:
        # print(f"Error generating feature plot: {e}")
        return px.line(title=f"Error generating {graph_type} for {x_feature}" + (f" vs {y_feature}" if y_feature and graph_type == 'scatter' else "") + f". Check data. Error: {str(e)}")

    fig.update_layout(legend_title_text='Result')
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
    id_cols = ['location_id', 'sensor_id', 'result']
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
        # Return empty options and value if no location is selected (e.g. if clearable=True for location)
        return [], []

    # Filter sensors based on the selected location
    filtered_sensors = sensor_df[sensor_df['location_id'] == selected_location]['sensor_id'].unique()
    sensor_options = [{'label': sid, 'value': sid} for sid in filtered_sensors]

    # When location changes, reset the selected sensors to none.
    # This prevents displaying data from a previous location's sensor if its ID happens to exist in the new location.
    return sensor_options, []


# Callback to update graph
@app.callback(
    Output('waveform-graph', 'figure'),
    [Input('location-dropdown', 'value'),
     Input('sensor-dropdown', 'value')]
)
def update_graph(selected_location, selected_sensor_ids):
    # Ensure both a location and at least one sensor are selected
    if not selected_location or not selected_sensor_ids:
        fig = px.line(title="Please select a location and one or more sensors.")
        fig.update_layout(xaxis_title="Timestamp", yaxis_title="Value")
        return fig

    # Filter the DataFrame based on the selected location and sensor IDs
    filtered_df = sensor_df[
        (sensor_df['location_id'] == selected_location) &
        (sensor_df['sensor_id'].isin(selected_sensor_ids))
    ]

    if filtered_df.empty:
        fig = px.line(title=f"No data found for selected sensors in location {selected_location}.")
        fig.update_layout(xaxis_title="Timestamp", yaxis_title="Value")
        return fig

    # Create a title that lists the selected location and sensors with their results
    title_parts = []
    for sid in selected_sensor_ids:
        # Ensure we get the result for the specific sensor within the selected location
        # This is important if sensor IDs could somehow be non-unique across locations (though not in current data gen)
        result = filtered_df[(filtered_df['sensor_id'] == sid)]['result'].iloc[0]
        title_parts.append(f"{sid} ({result})")
    title = f"Waveforms for Location {selected_location}: {', '.join(title_parts)}"

    # Define the color map for sensor_id based on their 'result'
    color_discrete_map = {}
    for sensor_id in filtered_df['sensor_id'].unique(): # Use unique sensors from the filtered_df
        result_status = filtered_df[filtered_df['sensor_id'] == sensor_id]['result'].iloc[0]
        if result_status == "OK":
            color_discrete_map[sensor_id] = "green"
        elif result_status == "NG":
            color_discrete_map[sensor_id] = "red"
        else:
            color_discrete_map[sensor_id] = "blue" # Fallback for unexpected results

    # Create the line plot
    fig = px.line(filtered_df,
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
    app.run_server(debug=True, host='0.0.0.0', port=8050)

#TODO: Next steps from the plan:
# 4. 特徴量可視化機能の実装
#    - `app.py` のレイアウトに、特徴量可視化のためのグラフエリアと、表示する特徴量を選択するドロップダウン（X軸、Y軸、グラフタイプ選択など）を追加します。
#    - `dcc.Store` に格納された特徴量データに基づいて、選択された特徴量の散布図やヒストグラムを生成するコールバックを実装します。プロットは `result` ('OK'/'NG') で色分けします。
# 5. CSVエクスポート機能の実装
