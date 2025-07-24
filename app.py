import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import numpy as np
from data_generator import generate_sensor_data
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from feature_extractor import extract_features as calculate_waveform_features, numerize_result
import ml_processor
import webbrowser
import json
import os

# --- Configuration Handling ---
CONFIG_FILE = "config.json"

def load_config():
    """Loads configuration from config.json or returns default."""
    default_config = {
      "waveform": { "min_count": 10000, "max_count": 40000, "interval": 1 },
      "columns": { "time": "DATE_AND_TIME", "assy_number_col_name": "AssyNo", "ct": "CoreAssy1[10]", "plc_signal": "CoreAssy1[42]" },
      "ngdata_columns": { "ng_data_qr_col_name": "qrcode", "ng_type_col_name": "furyo", "ng_type_name": "77.チューブ根付け漏れ　カシメ側" },
      "sensor_name": [ "SNS1[0]", "SNS1[1]", "SNS1[2]", "SNS2[0]", "SNS2[1]", "SNS2[2]", "SNS3[0]", "SNS3[1]", "SNS3[2]", "SNS3[3]", "SNS4[0]", "SNS4[1]", "SNS4[2]", "SNS4[3]", "SNS4[4]", "SNS4[5]", "SNS4[6]", "SNS4[7]", "Hiwin1", "Hiwin2", "Hiwin3", "Hiwin4" ],
      "paths": { "data_folder": "./data/", "ng_data_folder": "./ng_data/" }
    }
    if os.path.exists(CONFIG_FILE):
        print(f"Loading configuration from {CONFIG_FILE}...")
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode {CONFIG_FILE}. Using default config.")
                return default_config
    else:
        print("No config.json found. Using default configuration.")
        return default_config

# Load initial config
config_data = load_config()

# Generate or load data
print("Generating initial sensor data, please wait...")
sensor_df = generate_sensor_data() # This might use config_data in a future implementation
print("Sensor data generation complete. Application is ready.")

# Initialize the Dash app with a DBC theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# App layout using dbc.Container for better spacing and responsiveness
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Sensor Waveform Analysis Dashboard", className="text-center my-4"), width=11),
        dbc.Col(dbc.Button("Settings", id="open-settings-modal-button", className="mt-4"), width=1, style={'textAlign': 'right'})
    ]),
    
    # Settings Modal
    dbc.Modal([
        dbc.ModalHeader("Application Settings"),
        dbc.ModalBody(
            dbc.Form([
                # Waveform Settings
                html.H5("Waveform"),
                dbc.Row([
                    dbc.Col(dbc.Label("Min Count"), width=3),
                    dbc.Col(dbc.Input(type="number", id="config-waveform-min_count"), width=9)
                ], className="mb-2"),
                dbc.Row([
                    dbc.Col(dbc.Label("Max Count"), width=3),
                    dbc.Col(dbc.Input(type="number", id="config-waveform-max_count"), width=9)
                ], className="mb-2"),
                dbc.Row([
                    dbc.Col(dbc.Label("Interval"), width=3),
                    dbc.Col(dbc.Input(type="number", id="config-waveform-interval"), width=9)
                ], className="mb-3"),

                # Column Names
                html.H5("Column Names"),
                dbc.Row([
                    dbc.Col(dbc.Label("Time Column"), width=4),
                    dbc.Col(dbc.Input(type="text", id="config-columns-time"), width=8)
                ], className="mb-2"),
                dbc.Row([
                    dbc.Col(dbc.Label("Assy Number Column"), width=4),
                    dbc.Col(dbc.Input(type="text", id="config-columns-assy_number"), width=8)
                ], className="mb-2"),
                dbc.Row([
                    dbc.Col(dbc.Label("CT Column"), width=4),
                    dbc.Col(dbc.Input(type="text", id="config-columns-ct"), width=8)
                ], className="mb-2"),
                dbc.Row([
                    dbc.Col(dbc.Label("PLC Signal Column"), width=4),
                    dbc.Col(dbc.Input(type="text", id="config-columns-plc_signal"), width=8)
                ], className="mb-3"),

                # NG Data Columns
                html.H5("NG Data Columns"),
                dbc.Row([
                    dbc.Col(dbc.Label("QR Code Column"), width=4),
                    dbc.Col(dbc.Input(type="text", id="config-ng-qr"), width=8)
                ], className="mb-2"),
                dbc.Row([
                    dbc.Col(dbc.Label("NG Type Column"), width=4),
                    dbc.Col(dbc.Input(type="text", id="config-ng-type_col"), width=8)
                ], className="mb-2"),
                dbc.Row([
                    dbc.Col(dbc.Label("NG Type Name"), width=4),
                    dbc.Col(dbc.Input(type="text", id="config-ng-type_name"), width=8)
                ], className="mb-3"),

                # Sensor Names
                html.H5("Sensor Names"),
                dbc.Textarea(id="config-sensor_names", placeholder="Enter sensor names, one per line", style={"height": "150px"}, className="mb-3"),

                # Paths
                html.H5("Paths"),
                dbc.Row([
                    dbc.Col(dbc.Label("Data Folder"), width=3),
                    dbc.Col(dbc.Input(type="text", id="config-paths-data"), width=9)
                ], className="mb-2"),
                dbc.Row([
                    dbc.Col(dbc.Label("NG Data Folder"), width=3),
                    dbc.Col(dbc.Input(type="text", id="config-paths-ng_data"), width=9)
                ], className="mb-2"),
            ])
        ),
        dbc.ModalFooter([
            dbc.Button("Save", id="save-settings-button", color="primary"),
            dbc.Button("Cancel", id="cancel-settings-button", color="secondary")
        ])
    ], id="settings-modal", is_open=False, size="lg"), # Large modal

    dbc.Tabs([
        dbc.Tab(label="Waveform Explorer & Feature Extraction", children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Select ng_type:"),
                    dcc.Dropdown(
                        id='ng-type-dropdown',
                        options=[{'label': 'All', 'value': 'all'}] + [{'label': ng_type, 'value': ng_type} for ng_type in sensor_df['ng_type'].unique()],
                        value='all',
                        clearable=False
                    ),
                ], md=2),
                dbc.Col([
                    html.Label("Select AssyNo:"),
                    dcc.Dropdown(
                        id='assy-no-dropdown',
                        options=[{'label': 'All', 'value': 'all'}],
                        value=['all'],
                        multi=True
                    ),
                ], md=3),
                dbc.Col([
                    html.Label("Select hinban:"),
                    dcc.Dropdown(
                        id='hinban-dropdown',
                        options=[{'label': 'All', 'value': 'all'}],
                        value='all',
                        clearable=False
                    ),
                ], md=2),
                dbc.Col([
                    html.Label("Select Sensor_Type:"),
                    dcc.Dropdown(
                        id='sensor-type-dropdown',
                        options=[{'label': 'All', 'value': 'all'}],
                        value='all',
                        clearable=False
                    ),
                ], md=3),
                dbc.Col([
                    html.Label("Select result:"),
                    dcc.Dropdown(
                        id='result-dropdown',
                        options=[{'label': 'All', 'value': 'all'}] + [{'label': res, 'value': res} for res in sensor_df['result'].unique()],
                        value='all',
                        clearable=False
                    ),
                ], md=2),
            ], className="mt-3"),

            dbc.Row([
                dbc.Col([
                    html.Label("Line Width:"),
                    dcc.Slider(id='line-width-slider', min=0.5, max=5, step=0.5, value=1, marks={i: str(i) for i in range(1, 6)}),
                ], width=6),
                dbc.Col([
                    html.Label("Opacity:"),
                    dcc.Slider(id='opacity-slider', min=0.1, max=1, step=0.1, value=1.0, marks={i/10: str(i/10) for i in range(1, 11, 2)}),
                ], width=6),
            ], className="mt-4"),

            dbc.Row(dbc.Col(dcc.Graph(id='waveform-graph'), className="mt-3")),
            dbc.Row(dbc.Col(html.Hr(), className="my-3")),

            dbc.Row([
                dbc.Col(html.H3("Feature Extraction Settings"), width=12, className="mb-3"),
                dbc.Col([
                    html.Label("Select Features:"),
                    dcc.Checklist(
                        id='feature-checklist',
                        options=[
                            # Basic
                            {'label': 'Mean', 'value': 'mean'},
                            {'label': 'Std Dev', 'value': 'std'},
                            {'label': 'Max', 'value': 'max'},
                            {'label': 'Min', 'value': 'min'},
                            {'label': 'RMS', 'value': 'rms'},
                            # General
                            {'label': 'Skewness', 'value': 'skewness'},
                            {'label': 'Kurtosis', 'value': 'kurtosis'},
                            {'label': 'Peak Count', 'value': 'num_peaks'},
                            {'label': 'Zero Crossings', 'value': 'num_zero_crossings'},
                            {'label': 'Mean Abs. Change', 'value': 'mean_abs_change'},
                            # Segmented
                            {'label': 'Segmented Features', 'value': 'segmented_features'},
                            # TSFRESH based
                            {'label': 'TSF: Abs. Energy', 'value': 'tsf_abs_energy'},
                            {'label': 'TSF: CID CE (norm)', 'value': 'tsf_cid_ce'},
                            {'label': 'TSF: Quantile 0.25', 'value': 'tsf_quantile_0.25'},
                            {'label': 'TSF: Quantile 0.75', 'value': 'tsf_quantile_0.75'},
                            {'label': 'TSF: Autocorr lag 1', 'value': 'tsf_autocorrelation_lag1'},
                            {'label': 'TSF: Autocorr lag 2', 'value': 'tsf_autocorrelation_lag2'},
                            {'label': 'TSF: Mean Slope (chunks)', 'value': 'tsf_agg_linear_trend_slope_mean'},
                        ],
                        value=['mean', 'std'], # Default to a few common ones
                        labelStyle={'display': 'block'},
                        inputClassName="mb-1" # Margin bottom for each checklist item
                    ),
                ], md=4),

                dbc.Col([
                    html.Label("Number of Segments (N):"),
                    dcc.Input(
                        id='n-segments-input', type='number', placeholder='Enter N', value=5,
                        min=1, step=1, className="form-control"
                    ),
                ], id='n-segments-input-div', md=4, style={'display': 'none'}), # Hide by default

                dbc.Col([
                    dbc.Button('Extract Features', id='extract-features-button', n_clicks=0, className="mt-4", color="primary")
                ], md=4, className="d-flex align-items-end"),
            ], className="mb-2"),
            dbc.Row(dbc.Col(html.Div(id='feature-extraction-notification'), width=12, className="mb-4")), # Notification area moved here
        ]),

        dbc.Tab(label="Feature Analysis", children=[
            dbc.Row(dbc.Col(
                dcc.Loading(
                    id="loading-feature-table",
                    type="default",
                    children=[html.Div(id='feature-table-output')]
                ),
                width=12
            )),
            dbc.Row(dbc.Col(html.Hr(), className="my-3")),
            dbc.Row([
                dbc.Col(html.H3("Feature Visualization"), width=12, className="mb-3"),
                dbc.Col([
                    html.Label("Graph Type:"),
                    dcc.Dropdown(
                        id='feature-graph-type-dropdown',
                        options=[
                            {'label': 'Parallel Coordinates', 'value': 'parallel_coordinates'},
                            {'label': 'Scatter Plot', 'value': 'scatter'}
                        ],
                        value='parallel_coordinates',
                        clearable=False
                    )
                ], md=4),
                dbc.Col([
                    html.Label("X-axis Feature (Scatter):"),
                    dcc.Dropdown(id='x-axis-feature-dropdown', options=[], placeholder="Select X-axis")
                ], id='x-axis-div', md=4, style={'display': 'none'}),
                dbc.Col([
                    html.Label("Y-axis Feature (Scatter):"),
                    dcc.Dropdown(id='y-axis-feature-dropdown', options=[], placeholder="Select Y-axis")
                ], id='y-axis-div', md=4, style={'display': 'none'}),
            ]),
            dbc.Row(dbc.Col(dcc.Graph(id='feature-visualization-graph'), className="mt-3")),
            dbc.Row(dbc.Col(html.Div(id='feature-stats-output', style={'marginTop': '20px', 'display': 'none'}), width=12)),
            dbc.Row(dbc.Col(html.Hr(), className="my-3")),
            dbc.Row([
                dbc.Col(html.H3("Feature Correlation Heatmap"), width=12, className="mb-3"),
                dbc.Col(dcc.Graph(id='feature-correlation-heatmap'), width=12)
            ]),
            dbc.Row(dbc.Col(dbc.Button("Download Extracted Features as CSV", id="btn-download-csv", className="mt-3 mb-3", color="success"), width=12)),
        ]),

        dbc.Tab(label="Machine Learning Analysis", children=[
            dbc.Row([
                dbc.Col(html.H3("Machine Learning Model Training & Evaluation"), width=12, className="my-3"),
                dbc.Col([
                    html.Label("Select Models to Train:"),
                    dcc.Checklist(
                        id='ml-model-checklist',
                        options=[
                            {'label': 'Logistic Regression', 'value': 'Logistic Regression'},
                            {'label': 'Random Forest', 'value': 'Random Forest'},
                            {'label': 'LightGBM', 'value': 'LightGBM'}
                        ],
                        value=['Logistic Regression'],
                        labelStyle={'display': 'block'}
                    ),
                ], md=4),

                dbc.Col([
                    dbc.Button('Run ML Analysis', id='run-ml-analysis-button', n_clicks=0, className="mt-4", color="primary")
                ], md=4, className="d-flex align-items-end"),
            ], className="mb-4"),
            dbc.Row(dbc.Col(
                dcc.Loading(
                    id="loading-ml-results", type="default",
                    children=[html.Div(id='ml-results-output-area')]
                ),
                width=12, className="mt-3"
            )),
        ]),
    ]),

    dcc.Download(id="download-dataframe-csv"),
    dcc.Store(id='extracted-features-store'),
    dcc.Store(id='config-store', data=config_data) # Add config store to layout
], fluid=True)


# --- Settings Modal Callbacks ---

@app.callback(
    Output('config-store', 'data'),
    Output("settings-modal", "is_open", allow_duplicate=True), # Use allow_duplicate
    Output("feature-extraction-notification", "children", allow_duplicate=True), # For save notification
    [Input("save-settings-button", "n_clicks")],
    [State("config-waveform-min_count", "value"),
     State("config-waveform-max_count", "value"),
     State("config-waveform-interval", "value"),
     State("config-columns-time", "value"),
     State("config-columns-assy_number", "value"),
     State("config-columns-ct", "value"),
     State("config-columns-plc_signal", "value"),
     State("config-ng-qr", "value"),
     State("config-ng-type_col", "value"),
     State("config-ng-type_name", "value"),
     State("config-sensor_names", "value"),
     State("config-paths-data", "value"),
     State("config-paths-ng_data", "value"),
     State("config-store", "data")],
    prevent_initial_call=True
)
def save_settings(n_clicks, min_count, max_count, interval, time_col, assy_col, ct_col, plc_col,
                  ng_qr, ng_type_col, ng_type_name, sensor_names_str, data_path, ng_data_path,
                  current_config):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate

    # Reconstruct the config dictionary from form values
    new_config = {
        "waveform": {
            "min_count": min_count,
            "max_count": max_count,
            "interval": interval
        },
        "columns": {
            "time": time_col,
            "assy_number_col_name": assy_col,
            "ct": ct_col,
            "plc_signal": plc_col
        },
        "ngdata_columns": {
            "ng_data_qr_col_name": ng_qr,
            "ng_type_col_name": ng_type_col,
            "ng_type_name": ng_type_name
        },
        # Convert sensor names from string (one per line) to list
        "sensor_name": [name.strip() for name in sensor_names_str.split('\n') if name.strip()],
        "paths": {
            "data_folder": data_path,
            "ng_data_folder": ng_data_path
        }
    }

    # Save to config.json file
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(new_config, f, indent=2, ensure_ascii=False)
        notification = dbc.Alert("Settings saved successfully!", color="success", duration=3000)
    except Exception as e:
        print(f"Error saving config file: {e}")
        notification = dbc.Alert(f"Error saving settings: {e}", color="danger", duration=5000)

    # Return updated config to store, close modal, and show notification
    return new_config, False, notification


@app.callback(
    Output("settings-modal", "is_open", allow_duplicate=True),
    [Input("open-settings-modal-button", "n_clicks"), Input("cancel-settings-button", "n_clicks")],
    [State("settings-modal", "is_open")],
    prevent_initial_call=True
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

@app.callback(
    [Output("config-waveform-min_count", "value"),
     Output("config-waveform-max_count", "value"),
     Output("config-waveform-interval", "value"),
     Output("config-columns-time", "value"),
     Output("config-columns-assy_number", "value"),
     Output("config-columns-ct", "value"),
     Output("config-columns-plc_signal", "value"),
     Output("config-ng-qr", "value"),
     Output("config-ng-type_col", "value"),
     Output("config-ng-type_name", "value"),
     Output("config-sensor_names", "value"),
     Output("config-paths-data", "value"),
     Output("config-paths-ng_data", "value")],
    [Input("settings-modal", "is_open")],
    [State("config-store", "data")]
)
def populate_settings_modal(is_open, stored_config):
    if not is_open or not stored_config:
        return (dash.no_update,) * 13 # Return no_update for all 13 outputs

    return [
        stored_config.get("waveform", {}).get("min_count"),
        stored_config.get("waveform", {}).get("max_count"),
        stored_config.get("waveform", {}).get("interval"),
        stored_config.get("columns", {}).get("time"),
        stored_config.get("columns", {}).get("assy_number_col_name"),
        stored_config.get("columns", {}).get("ct"),
        stored_config.get("columns", {}).get("plc_signal"),
        stored_config.get("ngdata_columns", {}).get("ng_data_qr_col_name"),
        stored_config.get("ngdata_columns", {}).get("ng_type_col_name"),
        stored_config.get("ngdata_columns", {}).get("ng_type_name"),
        "\n".join(stored_config.get("sensor_name", [])), # Join list to string for textarea
        stored_config.get("paths", {}).get("data_folder"),
        stored_config.get("paths", {}).get("ng_data_folder"),
    ]

# Callback to show/hide N Segments input based on "Segmented Features" selection
@app.callback(
    Output('n-segments-input-div', 'style'),
    [Input('feature-checklist', 'value')]
)
def toggle_n_segments_input(selected_features):
    # Base style for the column, assuming md=4 is handled by className or parent
    # We only toggle 'display'
    base_style = {'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '20px'} # Default style from previous version for consistency
    if selected_features and 'segmented_features' in selected_features:
        base_style['display'] = 'block' 
    else:
        base_style['display'] = 'none'
    return base_style

# Callback to extract features and store them
@app.callback(
    Output('extracted-features-store', 'data'),
    Output('feature-table-output', 'children'),
    Output('feature-extraction-notification', 'children'), # New output for notifications
    [Input('extract-features-button', 'n_clicks')],
    [State('ng-type-dropdown', 'value'),
     State('assy-no-dropdown', 'value'),
     State('feature-checklist', 'value'),
     State('n-segments-input', 'value'),
     State('extracted-features-store', 'data')]
)
def handle_feature_extraction(n_clicks, selected_ng_type, selected_assy_nos,
                              selected_features_names, n_segments, existing_store_data):
    if n_clicks == 0:
        return existing_store_data, html.P("Feature data will appear here after extraction. Select options and click 'Extract Features'."), "" # No notification initially

    # --- Create a compatible DataFrame for feature extraction ---
    # The feature extraction logic expects 'location_id', 'sensor_id', and 'value' columns.
    # We create a temporary DataFrame with these columns from our main sensor_df.

    # Start with a copy of the main DataFrame
    df_for_extraction = sensor_df.copy()

    # Filter based on the main waveform graph selections
    if selected_ng_type != 'all':
        df_for_extraction = df_for_extraction[df_for_extraction['ng_type'] == selected_ng_type]

    # Handle 'all' case for multi-select AssyNo dropdown
    if 'all' not in selected_assy_nos and selected_assy_nos:
        df_for_extraction = df_for_extraction[df_for_extraction['AssyNo'].isin(selected_assy_nos)]

    # Rename columns for compatibility
    df_for_extraction = df_for_extraction.rename(columns={
        "ng_type": "location_id",
        "AssyNo": "sensor_id",
        "Sensor_Value": "value"
    })

    selected_location = selected_ng_type
    selected_sensor_ids = selected_assy_nos

    # Handle validation errors by returning an alert to the notification area
    if not selected_location:
        return existing_store_data, dash.no_update, dbc.Alert("Please select an ng_type.", color="danger", dismissable=True)
    if not selected_sensor_ids:
        return existing_store_data, dash.no_update, dbc.Alert("Please select at least one AssyNo.", color="danger", dismissable=True)
    if not selected_features_names:
        return existing_store_data, dash.no_update, dbc.Alert("Please select at least one feature to extract.", color="danger", dismissable=True)

    # Validate N for segmented features
    if 'segmented_features' in selected_features_names:
        if n_segments is None or not isinstance(n_segments, int) or n_segments <= 0:
            return existing_store_data, dash.no_update, dbc.Alert(f"Invalid number of segments (N={n_segments}). Please provide a positive integer.", color="danger", dismissable=True)

    # Determine sensors to process
    sensors_to_process = []
    if 'all' in selected_sensor_ids:
        sensors_to_process = list(df_for_extraction['sensor_id'].unique())
    else:
        sensors_to_process = [sid for sid in selected_sensor_ids if sid != 'all']

    # Filter waveform data for feature extraction
    wave_df_for_extraction = df_for_extraction[df_for_extraction['sensor_id'].isin(sensors_to_process)]

    if wave_df_for_extraction.empty:
        return [], dash.no_update, dbc.Alert("No waveform data found for selected sensors and location.", color="warning", dismissable=True)

    # --- Feature Extraction Logic ---
    all_extracted_features = []
    for s_id in wave_df_for_extraction['sensor_id'].unique():
        sensor_waveform_data = wave_df_for_extraction[wave_df_for_extraction['sensor_id'] == s_id]['value']
        if sensor_waveform_data.empty: continue
        
        features = calculate_waveform_features(
            series=sensor_waveform_data,
            selected_feature_names=selected_features_names,
            n_segments=n_segments # Pass N regardless, function will ignore if not needed
        )
        
        # Add identifier columns back for context
        features['sensor_id'] = s_id
        # Get the original 'ng_type' for the 'location_id' field
        features['location_id'] = wave_df_for_extraction[wave_df_for_extraction['sensor_id'] == s_id]['location_id'].iloc[0]
        features['result'] = wave_df_for_extraction[wave_df_for_extraction['sensor_id'] == s_id]['result'].iloc[0]
        # Add other relevant columns if needed, e.g., hinban
        features['hinban'] = sensor_df[sensor_df['AssyNo'] == s_id]['hinban'].iloc[0]
        all_extracted_features.append(features)

    if not all_extracted_features:
        return [], dash.no_update, dbc.Alert("Could not extract features for the selected sensors/settings.", color="warning", dismissable=True)

    features_df = pd.DataFrame(all_extracted_features)
    if not features_df.empty:
        features_df['result_numeric'] = numerize_result(features_df['result'])
        # Define order of columns for the output table
        id_cols = ['location_id', 'sensor_id', 'hinban', 'result', 'result_numeric']
        present_id_cols = [col for col in id_cols if col in features_df.columns]
        feature_col_names = sorted([col for col in features_df.columns if col not in present_id_cols])
        ordered_columns = present_id_cols + feature_col_names
        features_df_ordered = features_df[ordered_columns]

        datatable_output = dash_table.DataTable(
            id='interactive-feature-table',
            columns=[{"name": i, "id": i} for i in features_df_ordered.columns],
            data=features_df_ordered.to_dict('records'),
            page_action="native", page_current=0, page_size=10,
            filter_action="native", sort_action="native", sort_mode="multi",
            style_table={'overflowX': 'auto', 'marginTop': '20px'},
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
            style_cell={'height': 'auto', 'minWidth': '90px', 'width': '120px', 'maxWidth': '180px', 'whiteSpace': 'normal', 'textAlign': 'left'}
        )
    else:
        datatable_output = dbc.Alert("No features extracted or data is empty.", color="info", style={'marginTop': '20px'})

    # Success notification
    notification = dbc.Alert("Feature extraction complete!", color="success", duration=4000)
    
    # Store the DataFrame with compatible column names ('location_id', 'sensor_id')
    return features_df.to_dict('records'), datatable_output, notification

# Callback to update feature dropdowns based on stored data
@app.callback(
    Output('x-axis-feature-dropdown', 'options'),
    Output('y-axis-feature-dropdown', 'options'),
    Output('x-axis-feature-dropdown', 'value'),
    Output('y-axis-feature-dropdown', 'value'),
    [Input('extracted-features-store', 'data')]
)
def update_feature_plot_dropdowns(stored_feature_data):
    if not stored_feature_data:
        return [], [], None, None

    df = pd.DataFrame(stored_feature_data)
    # Exclude all identifier columns from feature selection
    potential_features = [col for col in df.columns if col not in ['sensor_id', 'location_id', 'result', 'hinban']]
    options = [{'label': col, 'value': col} for col in potential_features]

    default_x = options[0]['value'] if options else None
    default_y = options[1]['value'] if len(options) > 1 else None

    return options, options, default_x, default_y

# Callback to generate feature visualization graph
@app.callback(
    Output('feature-visualization-graph', 'figure'),
    Output('feature-stats-output', 'children'),
    [Input('extracted-features-store', 'data'),
     Input('x-axis-feature-dropdown', 'value'),
     Input('y-axis-feature-dropdown', 'value'),
     Input('feature-graph-type-dropdown', 'value')]
)
def update_feature_graph_and_stats(stored_feature_data, x_feature, y_feature, graph_type):
    empty_figure = go.Figure()
    empty_figure.update_layout(title_text="Extract features and select graph type/options.",
                               xaxis=dict(showgrid=False, zeroline=False, visible=True, range=[0,1]),
                               yaxis=dict(showgrid=False, zeroline=False, visible=True, range=[0,1]))
    empty_stats = html.P("")

    if not stored_feature_data:
        return empty_figure, empty_stats

    df = pd.DataFrame(stored_feature_data)
    if df.empty:
        return empty_figure, dbc.Alert("No feature data available for visualization.", color="info", dismissable=True, duration=4000)

    fig = empty_figure
    stats_output_content = []

    try:
        if graph_type == 'parallel_coordinates':
            numeric_cols_for_parallel = df.select_dtypes(include=np.number).columns.tolist()

            if 'result_numeric' in numeric_cols_for_parallel and len(numeric_cols_for_parallel) > 1:
                # Exclude all identifier columns
                dimensions = [col for col in numeric_cols_for_parallel if col not in ['sensor_id', 'location_id', 'hinban']]
                if not dimensions:
                    fig.update_layout(title_text="No numeric dimensions found for Parallel Coordinates plot.")
                    return fig, dbc.Alert("No numeric dimensions for Parallel Coordinates.", color="warning", dismissable=True, duration=4000)

                if 'result_numeric' in df.columns:
                     fig = px.parallel_coordinates(
                        df,
                        dimensions=dimensions,
                        color="result_numeric",
                        color_continuous_scale=[(0, "green"), (1, "red")],
                        labels={col: col.replace("_", " ").title() for col in dimensions},
                        title="Parallel Coordinates Plot of Features (OK: Green, NG: Red)"
                    )
                     fig.update_layout(
                         margin=dict(l=80, r=80, t=100, b=80), # Increase margins
                     )
            else:
                fig.update_layout(title_text="Not enough numeric data for Parallel Coordinates plot.")
                stats_output_content = [dbc.Alert("Not enough data for Parallel Coordinates.", color="warning", dismissable=True, duration=4000)]

        elif graph_type == 'scatter':
            if not x_feature or not y_feature: # Check if features are selected for scatter
                fig.update_layout(title_text="Please select X and Y features for Scatter Plot.")
                return fig, dbc.Alert("Select X and Y features for Scatter Plot.", color="info", dismissable=True, duration=4000)
            if x_feature not in df.columns or y_feature not in df.columns: # Check if selected features exist
                fig.update_layout(title_text="Selected feature(s) not found.")
                return fig, dbc.Alert("Selected feature(s) not found in data.", color="warning", dismissable=True, duration=4000)

            fig = px.scatter(df, x=x_feature, y=y_feature, color='result',
                             title=f"Scatter Plot: {x_feature} vs {y_feature}",
                             color_discrete_map={"OK": "green", "NG": "red"},
                             hover_data=['sensor_id', 'location_id', 'hinban'])

            stats_output_content.append(html.H5(f"Descriptive Statistics for '{x_feature}':"))
            ok_data = df[(df['result'] == 'OK') & pd.notna(df[x_feature])][x_feature]
            ng_data = df[(df['result'] == 'NG') & pd.notna(df[x_feature])][x_feature]

            if not ok_data.empty:
                stats_output_content.append(html.H6("OK Group:"))
                stats_output_content.append(html.Pre(ok_data.describe().to_string()))
            if not ng_data.empty:
                stats_output_content.append(html.H6("NG Group:"))
                stats_output_content.append(html.Pre(ng_data.describe().to_string()))
            if ok_data.empty and ng_data.empty:
                stats_output_content.append(dbc.Alert(f"No data for '{x_feature}' to calculate statistics.", color="info", dismissable=True, duration=4000))
        else:
            fig.update_layout(title_text="Invalid graph type selected.")
            stats_output_content = [dbc.Alert("Select a valid graph type.", color="warning", dismissable=True, duration=4000)]

    except Exception as e:
        error_message = f"Error generating {graph_type} plot. Error: {str(e)}"
        fig.update_layout(title_text=error_message)
        stats_output_content = [dbc.Alert(error_message, color="danger", dismissable=True, duration=4000)]

    if fig is not empty_figure:
        fig.update_layout(legend_title_text='Result')

    return fig, html.Div(stats_output_content)

# Callback to control visibility of X/Y axis dropdowns and stats output
@app.callback(
    Output('x-axis-div', 'style'),
    Output('y-axis-div', 'style'),
    Output('feature-stats-output', 'style'),
    [Input('feature-graph-type-dropdown', 'value')]
)
def toggle_axis_selectors_and_stats(graph_type):
    # Using dbc.Col's md parameter for width, so style only controls display here.
    # The parent dbc.Col for these dropdowns already has md=4.
    scatter_style_visible = {'width': '100%', 'display': 'block'} # Ensure full width of its Col, and visible
    stats_style_visible = {'marginTop': '20px', 'display': 'block'}
    hidden_style = {'display': 'none'}

    if graph_type == 'scatter':
        return scatter_style_visible, scatter_style_visible, stats_style_visible
    else: # For parallel_coordinates or any other type
        return hidden_style, hidden_style, hidden_style


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
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols or len(numeric_cols) < 2:
        fig_empty_numeric = go.Figure()
        fig_empty_numeric.update_layout(title_text="Not enough numeric features for correlation heatmap.",
                                        xaxis=dict(showgrid=False, zeroline=False, visible=False),
                                        yaxis=dict(showgrid=False, zeroline=False, visible=False))
        return fig_empty_numeric

    correlation_matrix = df[numeric_cols].corr()
    fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        title="Feature Correlation Heatmap (including Result_Numeric)"
    )
    fig.update_layout(height=600)
    return fig

# Callback to run ML analysis and display results
@app.callback(
    Output('ml-results-output-area', 'children'),
    [Input('run-ml-analysis-button', 'n_clicks')],
    [State('extracted-features-store', 'data'),
     State('ml-model-checklist', 'value')]
)
def run_ml_analysis_and_display(n_clicks, stored_feature_data, selected_models):
    if n_clicks == 0 or not stored_feature_data or not selected_models:
        if n_clicks > 0 and not stored_feature_data:
            return dbc.Alert("Please extract features first before running ML analysis.", color="warning", dismissable=True, duration=4000)
        if n_clicks > 0 and not selected_models:
            return dbc.Alert("Please select at least one ML model to run.", color="danger", dismissable=True, duration=4000)
        return html.P("ML Analysis results will appear here. Select models and click 'Run ML Analysis'.")

    df = pd.DataFrame(stored_feature_data)
    if df.empty or 'result_numeric' not in df.columns:
        return dbc.Alert("Feature data is empty or 'result_numeric' column is missing.", color="danger", dismissable=True, duration=4000)

    X_train, X_test, y_train, y_test, feature_names, error_msg = ml_processor.preprocess_data(df.copy())

    if error_msg:
        return dbc.Alert(f"Error during preprocessing: {error_msg}", color="danger", dismissable=True, duration=4000)
    if X_train is None or X_test is None: # Should be caught by error_msg but as a safeguard
        return dbc.Alert("Failed to preprocess data for ML analysis (unknown reason).", color="danger", dismissable=True, duration=4000)

    results_children = []
    for model_name in selected_models:
        results_children.append(html.H4(f"Results for: {model_name}", className="mt-4"))
        model_results = ml_processor.train_and_evaluate_model(X_train, X_test, y_train, y_test, model_name, feature_names)

        if model_results.get("error"):
            results_children.append(dbc.Alert(f"Error training/evaluating {model_name}: {model_results['error']}", color="danger", dismissable=True, duration=4000))
            continue

        # Feature Importances
        if model_results.get("feature_importances"):
            imp_df = pd.DataFrame(list(model_results["feature_importances"].items()), columns=['Feature', 'Importance']).sort_values(by="Importance", ascending=False)
            top_n = min(len(imp_df), 15)
            fig_imp = px.bar(imp_df.head(top_n), x='Importance', y='Feature', orientation='h', title=f"Top {top_n} Feature Importances")
            fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
            results_children.append(dcc.Graph(figure=fig_imp))

        # Confusion Matrix
        cm = model_results.get("confusion_matrix")
        if cm:
            cm_array = np.array(cm)
            fig_cm = px.imshow(cm_array, text_auto=True,
                               labels=dict(x="Predicted Label", y="True Label", color="Count"),
                               x=['OK (0)', 'NG (1)'], y=['OK (0)', 'NG (1)'],
                               color_continuous_scale='Blues',
                               title="Confusion Matrix")
            fig_cm.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
            results_children.append(dcc.Graph(figure=fig_cm))

        # Metrics Table
        metrics_data = { # Use f-string formatting here for consistency
            "Accuracy": f"{model_results.get('accuracy', 0.0):.4f}",
            "Precision": f"{model_results.get('precision', 0.0):.4f}",
            "Recall": f"{model_results.get('recall', 0.0):.4f}",
            "F1 Score": f"{model_results.get('f1_score', 0.0):.4f}",
            "ROC AUC": f"{model_results.get('roc_auc', 0.0):.4f}"
        }
        # metrics_df = pd.DataFrame([metrics_data]).T.reset_index() # This creates a DataFrame with one row
        metrics_df = pd.DataFrame(list(metrics_data.items()), columns=["Metric", "Value"]) # Correct way for two columns
        
        metrics_table = dbc.Table.from_dataframe(metrics_df, striped=True, bordered=True, hover=True, className="mt-3")
        results_children.append(html.H5("Evaluation Metrics:", className="mt-3"))
        results_children.append(metrics_table)

        # ROC Curve
        roc_data = model_results.get("roc_curve")
        if roc_data and roc_data.get("fpr") is not None and roc_data.get("tpr") is not None:
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=roc_data["fpr"], y=roc_data["tpr"], mode='lines', name=f'ROC Curve (AUC = {model_results.get("roc_auc", 0.0):.2f})'))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Chance', line=dict(dash='dash')))
            fig_roc.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', legend=dict(x=0.6, y=0.1))
            results_children.append(dcc.Graph(figure=fig_roc))

        results_children.append(html.Hr(className="my-4"))

    if not results_children:
        return dbc.Alert("No results to display. Check selections or data.", color="info", dismissable=True, duration=4000)

    return html.Div(results_children)

# Callback for CSV download
@app.callback(
    Output("download-dataframe-csv", "data"),
    [Input("btn-download-csv", "n_clicks")],
    [State('extracted-features-store', 'data')],
    prevent_initial_call=True,
)
def download_csv(n_clicks, stored_feature_data):
    if not stored_feature_data:
        return None

    df = pd.DataFrame(stored_feature_data)
    if df.empty:
        return None

    # The feature store already has 'location_id' and 'sensor_id' from the extraction callback
    id_cols = ['location_id', 'sensor_id', 'hinban', 'result', 'result_numeric']
    present_id_cols = [col for col in id_cols if col in df.columns]
    feature_cols = [col for col in df.columns if col not in present_id_cols]
    final_cols = present_id_cols + sorted(feature_cols)
    df_to_download = df[final_cols]

    return dcc.send_data_frame(df_to_download.to_csv, "extracted_features.csv", index=False)

# Callback to update dependent dropdowns
@app.callback(
    [Output('assy-no-dropdown', 'options'),
     Output('hinban-dropdown', 'options'),
     Output('sensor-type-dropdown', 'options'),
     Output('assy-no-dropdown', 'value'),
     Output('hinban-dropdown', 'value'),
     Output('sensor-type-dropdown', 'value')],
    [Input('ng-type-dropdown', 'value')]
)
def update_dependent_dropdowns(selected_ng_type):
    filtered_df = sensor_df
    if selected_ng_type != 'all':
        filtered_df = sensor_df[sensor_df['ng_type'] == selected_ng_type]

    assy_no_options = [{'label': 'All', 'value': 'all'}] + [{'label': assy, 'value': assy} for assy in filtered_df['AssyNo'].unique()]
    hinban_options = [{'label': 'All', 'value': 'all'}] + [{'label': hinban, 'value': hinban} for hinban in filtered_df['hinban'].unique()]
    sensor_type_options = [{'label': 'All', 'value': 'all'}] + [{'label': st, 'value': st} for st in filtered_df['Sensor_Type'].unique()]

    # Reset dependent dropdowns to 'all' when the parent changes
    return assy_no_options, hinban_options, sensor_type_options, ['all'], 'all', 'all'

# Callback to update graph
@app.callback(
    Output('waveform-graph', 'figure'),
    [Input('ng-type-dropdown', 'value'),
     Input('assy-no-dropdown', 'value'),
     Input('hinban-dropdown', 'value'),
     Input('result-dropdown', 'value'),
     Input('sensor-type-dropdown', 'value'),
     Input('line-width-slider', 'value'),
     Input('opacity-slider', 'value')]
)
def update_graph(selected_ng_type, selected_assy_nos, selected_hinban, selected_result, selected_sensor_type, line_width, opacity):

    filtered_df = sensor_df.copy()

    # Apply filters based on dropdowns
    if selected_ng_type != 'all':
        filtered_df = filtered_df[filtered_df['ng_type'] == selected_ng_type]

    if selected_hinban != 'all':
        filtered_df = filtered_df[filtered_df['hinban'] == selected_hinban]

    if selected_result != 'all':
        filtered_df = filtered_df[filtered_df['result'] == selected_result]

    if selected_sensor_type != 'all':
        filtered_df = filtered_df[filtered_df['Sensor_Type'] == selected_sensor_type]

    # Handle 'all' case for multi-select AssyNo dropdown
    if 'all' not in selected_assy_nos and selected_assy_nos:
        filtered_df = filtered_df[filtered_df['AssyNo'].isin(selected_assy_nos)]

    fig = go.Figure()

    if filtered_df.empty:
        fig.update_layout(title_text="No data found for selected criteria.",
                          xaxis=dict(showgrid=False, zeroline=False, visible=True),
                          yaxis=dict(showgrid=False, zeroline=False, visible=True))
        return fig

    color_map = {'OK': 'green', 'NG': 'red'}

    for assy_no, group in filtered_df.groupby('AssyNo'):
        result = group['result'].iloc[0]
        color = color_map.get(result, 'grey')
        fig.add_trace(go.Scatter(
            x=group['No'],
            y=group['Sensor_Value'],
            mode='lines',
            name=f"{assy_no} ({result})",
            line=dict(width=line_width, color=color),
            opacity=opacity
        ))

    fig.update_layout(
        title="Waveform Viewer",
        xaxis_title="No",
        yaxis_title="Sensor Value",
        legend_title="AssyNo (Result)"
    )
    return fig

if __name__ == '__main__':
    # Define the URL
    URL = "http://127.0.0.1:8050"
    # Open the URL in a new browser tab only in the main process
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        webbrowser.open_new(URL)
    # Run the app
    app.run(debug=True, host='127.0.0.1', port=8050)
