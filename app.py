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

# Generate or load data
print("Generating initial sensor data, please wait...")
sensor_df = generate_sensor_data()
print("Sensor data generation complete. Application is ready.")

# Initialize the Dash app with a DBC theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# App layout using dbc.Container for better spacing and responsiveness
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Sensor Waveform Analysis Dashboard", className="text-center my-4"))),

    dbc.Tabs([
        dbc.Tab(label="Waveform Explorer & Feature Extraction", children=[
            dbc.Row([ # Row for location and sensor dropdowns
                dbc.Col([
                    html.Label("Select Location ID:"),
                    dcc.Dropdown(
                        id='location-dropdown',
                        options=[{'label': 'All Locations', 'value': 'all'}] + [{'label': loc, 'value': loc} for loc in sensor_df['location_id'].unique()],
                        value='all',
                        clearable=False
                    ),
                ], md=6),

                dbc.Col([
                    html.Label("Select Sensor ID(s):"),
                    dcc.Dropdown(
                        id='sensor-dropdown',
                        options=[],
                        value=[],
                        multi=True
                    ),
                ], md=6),
            ], className="mt-3"),

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
    dcc.Store(id='extracted-features-store')
], fluid=True)


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
    [State('location-dropdown', 'value'),
     State('sensor-dropdown', 'value'),
     State('feature-checklist', 'value'),
     State('n-segments-input', 'value'),
     State('extracted-features-store', 'data')]
)
def handle_feature_extraction(n_clicks, selected_location, selected_sensor_ids,
                              selected_features_names, n_segments, existing_store_data):
    if n_clicks == 0:
        return existing_store_data, html.P("Feature data will appear here after extraction. Select options and click 'Extract Features'."), "" # No notification initially

    # Handle validation errors by returning an alert to the notification area
    if not selected_location:
        return existing_store_data, dash.no_update, dbc.Alert("Please select a location.", color="danger", dismissable=True)
    if not selected_sensor_ids:
        return existing_store_data, dash.no_update, dbc.Alert("Please select at least one sensor.", color="danger", dismissable=True)
    if not selected_features_names:
        return existing_store_data, dash.no_update, dbc.Alert("Please select at least one feature to extract.", color="danger", dismissable=True)

    # Validate N for segmented features
    if 'segmented_features' in selected_features_names:
        if n_segments is None or not isinstance(n_segments, int) or n_segments <= 0:
            return existing_store_data, dash.no_update, dbc.Alert(f"Invalid number of segments (N={n_segments}). Please provide a positive integer.", color="danger", dismissable=True)

    # Determine sensors to process
    sensors_to_process = []
    if 'all' in selected_sensor_ids:
        if selected_location == 'all':
            sensors_to_process = list(sensor_df['sensor_id'].unique())
        else:
            sensors_to_process = list(sensor_df[sensor_df['location_id'] == selected_location]['sensor_id'].unique())
    else:
        sensors_to_process = [sid for sid in selected_sensor_ids if sid != 'all']

    # Filter waveform data
    if selected_location == 'all':
        wave_df_for_extraction = sensor_df[sensor_df['sensor_id'].isin(sensors_to_process)]
    else:
        wave_df_for_extraction = sensor_df[
            (sensor_df['location_id'] == selected_location) &
            (sensor_df['sensor_id'].isin(sensors_to_process))
        ]

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

        features['sensor_id'] = s_id
        features['location_id'] = wave_df_for_extraction[wave_df_for_extraction['sensor_id'] == s_id]['location_id'].iloc[0]
        features['result'] = wave_df_for_extraction[wave_df_for_extraction['sensor_id'] == s_id]['result'].iloc[0]
        all_extracted_features.append(features)

    if not all_extracted_features:
        return [], dash.no_update, dbc.Alert("Could not extract features for the selected sensors/settings.", color="warning", dismissable=True)

    features_df = pd.DataFrame(all_extracted_features)
    if not features_df.empty:
        features_df['result_numeric'] = numerize_result(features_df['result'])
        id_cols = ['location_id', 'sensor_id', 'result', 'result_numeric']
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
    potential_features = [col for col in df.columns if col not in ['sensor_id', 'location_id', 'result']]
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
                dimensions = [col for col in numeric_cols_for_parallel if col not in ['sensor_id', 'location_id']]
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
                     # Adjust layout for better label visibility
                     # After further review, tickangle is not supported for parallel coordinates dimensions.
                     # The best approach is to increase margins and rely on hover labels.
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
                             hover_data=['sensor_id', 'location_id'])

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

    id_cols = ['location_id', 'sensor_id', 'result', 'result_numeric']
    feature_cols = [col for col in df.columns if col not in id_cols]
    final_cols = [col for col in id_cols if col in df.columns]
    final_cols.extend(sorted(feature_cols))
    df_to_download = df[final_cols]

    return dcc.send_data_frame(df_to_download.to_csv, "extracted_features.csv", index=False)

# Callback to update sensor dropdown based on location
@app.callback(
    Output('sensor-dropdown', 'options'),
    Output('sensor-dropdown', 'value'),
    [Input('location-dropdown', 'value')]
)
def update_sensor_options(selected_location):
    if not selected_location:
        return [], []

    options = []
    if selected_location == 'all':
        all_sensor_ids = sorted(sensor_df['sensor_id'].unique())
        options = [{'label': 'All Sensors', 'value': 'all'}] + [{'label': sid, 'value': sid} for sid in all_sensor_ids]
    else:
        filtered_sensors = sorted(sensor_df[sensor_df['location_id'] == selected_location]['sensor_id'].unique())
        options = [{'label': 'All Sensors', 'value': 'all'}] + [{'label': sid, 'value': sid} for sid in filtered_sensors]
    return options, []

# Callback to update graph
@app.callback(
    Output('waveform-graph', 'figure'),
    [Input('location-dropdown', 'value'),
     Input('sensor-dropdown', 'value')]
)
def update_graph(selected_location, selected_sensor_ids):
    if not selected_location or not selected_sensor_ids:
        fig = go.Figure()
        fig.update_layout(
            title_text="Please select a location and one or more sensors.",
            xaxis=dict(showgrid=False, zeroline=False, visible=True, range=[0,1]),
            yaxis=dict(showgrid=False, zeroline=False, visible=True, range=[0,1])
        )
        return fig

    query_parts = []
    if selected_location != 'all':
        query_parts.append(f"location_id == '{selected_location}'")

    actual_sensor_ids_to_filter = []
    if 'all' in selected_sensor_ids:
        if selected_location != 'all': # 'all' sensors for a specific location
             actual_sensor_ids_to_filter = list(sensor_df[sensor_df['location_id'] == selected_location]['sensor_id'].unique())
        # If selected_location is 'all' and 'all' sensors, no sensor ID filter is added to query_parts here
    else: # Specific sensors selected
        actual_sensor_ids_to_filter = [sid for sid in selected_sensor_ids if sid != 'all']

    if actual_sensor_ids_to_filter: # Add sensor filter only if specific sensors are chosen or 'all' for specific location
         query_parts.append(f"sensor_id in {actual_sensor_ids_to_filter}")

    if query_parts:
        current_df = sensor_df.query(" and ".join(query_parts))
    else: # This means 'All Locations' and 'All Sensors'
        current_df = sensor_df.copy()

    if current_df.empty:
        fig = go.Figure()
        fig.update_layout(title_text="No data found for selected criteria.",
                          xaxis=dict(showgrid=False, zeroline=False, visible=True, range=[0,1]),
                          yaxis=dict(showgrid=False, zeroline=False, visible=True, range=[0,1]))
        return fig

    # Title Generation
    title_parts = []
    sensors_in_title = current_df['sensor_id'].unique()
    if len(sensors_in_title) > 3: # Shorten if many sensors
        title_parts = [f"{sensors_in_title[0]} ({current_df[current_df['sensor_id'] == sensors_in_title[0]]['result'].iloc[0]})",
                       f"{sensors_in_title[1]} ({current_df[current_df['sensor_id'] == sensors_in_title[1]]['result'].iloc[0]})",
                       f"...and {len(sensors_in_title) - 2} more"]
    else:
        for sid_in_title in sensors_in_title: # Iterate over actual sensors in current_df for title
            result = current_df[current_df['sensor_id'] == sid_in_title]['result'].iloc[0]
            title_parts.append(f"{sid_in_title} ({result})")

    location_str = selected_location if selected_location != 'all' else "All Locations"

    # Refine sensors_str based on actual_sensor_ids_to_filter and selected_sensor_ids
    if 'all' in selected_sensor_ids and not actual_sensor_ids_to_filter and selected_location == 'all':
        sensors_str = "All Sensors (All Locations)"
    elif 'all' in selected_sensor_ids and actual_sensor_ids_to_filter: # 'all' sensors for a specific location
        sensors_str = "All Sensors in selected location"
    elif not title_parts:
        sensors_str = "None Selected / No Data"
    else:
        sensors_str = ", ".join(title_parts)

    title = f"Waveforms for Location: {location_str} - Sensors: {sensors_str}"


    color_discrete_map = {}
    for sensor_id_val in current_df['sensor_id'].unique():
        result_status = current_df[current_df['sensor_id'] == sensor_id_val]['result'].iloc[0]
        color_discrete_map[sensor_id_val] = "green" if result_status == "OK" else "red"

    fig = px.line(current_df,
                  x='timestamp',
                  y='value',
                  color='sensor_id',
                  title=title,
                  labels={'sensor_id': 'Sensor ID', 'value': 'Value', 'timestamp': 'Timestamp'},
                  color_discrete_map=color_discrete_map)

    fig.update_layout(legend_title_text='Sensor ID') # Changed from legend_title
    return fig

if __name__ == '__main__':
    # Define the URL
    URL = "http://127.0.0.1:8050"
    # Open the URL in a new browser tab
    webbrowser.open_new(URL)
    # Run the app
    app.run(debug=True, host='127.0.0.1', port=8050)
