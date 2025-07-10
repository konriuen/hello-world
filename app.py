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
                            {'label': 'Mean', 'value': 'mean'},
                            {'label': 'Std Dev', 'value': 'std'},
                            {'label': 'Max', 'value': 'max'},
                            {'label': 'Min', 'value': 'min'},
                            {'label': 'RMS', 'value': 'rms'},
                            {'label': 'Segmented Features', 'value': 'segmented_features'}
                        ],
                        value=[],
                        labelStyle={'display': 'block'}
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
            ], className="mb-4"),
        ]),

        dbc.Tab(label="Feature Analysis", children=[
            dbc.Row(dbc.Col(html.Div(id='feature-table-output'), width=12, className="mt-3")),
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
    if selected_features and 'segmented_features' in selected_features:
        return {'display': 'inline-block'} # Show
    return {'display': 'none'} # Hide

# Callback to extract features and store them
@app.callback(
    Output('extracted-features-store', 'data'),
    Output('feature-table-output', 'children'),
    [Input('extract-features-button', 'n_clicks')],
    [State('location-dropdown', 'value'),
     State('sensor-dropdown', 'value'),
     State('feature-checklist', 'value'),
     State('n-segments-input', 'value'),
     State('extracted-features-store', 'data')]
)
def handle_feature_extraction(n_clicks, selected_location, selected_sensor_ids,
                              selected_features_names, n_segments, existing_store_data):
    if n_clicks == 0 or not selected_location or not selected_sensor_ids or not selected_features_names:
        if n_clicks > 0:
            if not selected_location:
                return existing_store_data, html.P("Please select a location.", style={'color': 'red'})
            if not selected_sensor_ids:
                return existing_store_data, html.P("Please select at least one sensor.", style={'color': 'red'})
            if not selected_features_names:
                return existing_store_data, html.P("Please select at least one feature to extract.", style={'color': 'red'})
        return existing_store_data, "Feature data will appear here after extraction."

    sensors_to_process = []
    if 'all' in selected_sensor_ids:
        if selected_location == 'all':
            sensors_to_process = list(sensor_df['sensor_id'].unique())
        else:
            sensors_to_process = list(sensor_df[sensor_df['location_id'] == selected_location]['sensor_id'].unique())
    else:
        sensors_to_process = [sid for sid in selected_sensor_ids if sid != 'all']

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
    for s_id in wave_df_for_extraction['sensor_id'].unique():
        sensor_waveform_data = wave_df_for_extraction[wave_df_for_extraction['sensor_id'] == s_id]['value']
        actual_loc_id = wave_df_for_extraction[wave_df_for_extraction['sensor_id'] == s_id]['location_id'].iloc[0]

        if sensor_waveform_data.empty:
            continue

        current_n_segments = n_segments if selected_features_names and 'segmented_features' in selected_features_names else None

        if selected_features_names and 'segmented_features' in selected_features_names:
            if current_n_segments is None or not isinstance(current_n_segments, int) or current_n_segments <= 0:
                return existing_store_data, html.P(f"Invalid number of segments (N={current_n_segments}). Please provide a positive integer for N.", style={'color': 'red'})

        features = calculate_waveform_features(
            series=sensor_waveform_data,
            selected_feature_names=selected_features_names,
            n_segments=current_n_segments
        )

        features['sensor_id'] = s_id
        features['location_id'] = actual_loc_id
        features['result'] = wave_df_for_extraction[wave_df_for_extraction['sensor_id'] == s_id]['result'].iloc[0]
        all_extracted_features.append(features)

    if not all_extracted_features:
        return [], html.P("Could not extract features for the selected sensors/settings.", style={'color': 'orange'})

    features_df = pd.DataFrame(all_extracted_features)

    if not features_df.empty:
        features_df['result_numeric'] = numerize_result(features_df['result'])

    if not features_df.empty:
        id_cols = ['location_id', 'sensor_id', 'result', 'result_numeric']
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
            row_selectable="multi",
            row_deletable=False,
            selected_rows=[],
            page_action="native",
            page_current=0,
            page_size=10,
            style_table={'overflowX': 'auto', 'marginTop': '20px'},
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
        return empty_figure, html.P("No feature data available.")

    fig = empty_figure
    stats_output_content = []

    try:
        if graph_type == 'parallel_coordinates':
            numeric_cols_for_parallel = df.select_dtypes(include=np.number).columns.tolist()

            if 'result_numeric' in numeric_cols_for_parallel and len(numeric_cols_for_parallel) > 1:
                dimensions = [col for col in numeric_cols_for_parallel if col not in ['sensor_id', 'location_id']]
                if not dimensions:
                    fig.update_layout(title_text="No numeric dimensions found for Parallel Coordinates plot.")
                    return fig, html.P("No numeric dimensions for Parallel Coordinates.")

                if 'result_numeric' in df.columns:
                     fig = px.parallel_coordinates(
                        df,
                        dimensions=dimensions,
                        color="result_numeric",
                        color_continuous_scale=[(0, "green"), (1, "red")],
                        labels={col: col.replace("_", " ").title() for col in dimensions},
                        title="Parallel Coordinates Plot of Features (OK: Green, NG: Red)"
                    )
            else:
                fig.update_layout(title_text="Not enough numeric data for Parallel Coordinates plot.")
                stats_output_content = [html.P("Not enough data for Parallel Coordinates.")]

        elif graph_type == 'scatter':
            if not x_feature or not y_feature:
                fig.update_layout(title_text="Please select X and Y features for Scatter Plot.")
                return fig, html.P("Select X and Y features.")
            if x_feature not in df.columns or y_feature not in df.columns:
                fig.update_layout(title_text="Selected feature(s) not found.")
                return fig, html.P("Selected feature(s) not found.")

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
                stats_output_content.append(html.P(f"No data for '{x_feature}' to calculate statistics."))
        else:
            fig.update_layout(title_text="Invalid graph type selected.")
            stats_output_content = [html.P("Select a valid graph type.")]

    except Exception as e:
        error_message = f"Error generating {graph_type} plot. Error: {str(e)}"
        fig.update_layout(title_text=error_message)
        stats_output_content = [html.P(error_message, style={'color': 'red'})]

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
    scatter_style = {'display': 'inline-block'}
    stats_style_visible = {'marginTop': '20px', 'display': 'block'}
    hidden_style = {'display': 'none'}

    if graph_type == 'scatter':
        return scatter_style, scatter_style, stats_style_visible
    else:
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
            return html.P("Please extract features first before running ML analysis.", style={'color': 'orange'})
        if n_clicks > 0 and not selected_models:
            return html.P("Please select at least one ML model to run.", style={'color': 'red'})
        return html.P("ML Analysis results will appear here. Select models and click 'Run ML Analysis'.")

    df = pd.DataFrame(stored_feature_data)
    if df.empty or 'result_numeric' not in df.columns:
        return html.P("Feature data is empty or 'result_numeric' column is missing.", style={'color': 'red'})

    X_train, X_test, y_train, y_test, feature_names, error_msg = ml_processor.preprocess_data(df.copy())

    if error_msg:
        return html.P(f"Error during preprocessing: {error_msg}", style={'color': 'red'})
    if X_train is None or X_test is None:
        return html.P("Failed to preprocess data for ML analysis (unknown reason).", style={'color': 'red'})

    results_children = []
    for model_name in selected_models:
        results_children.append(html.H4(f"Results for: {model_name}"))
        model_results = ml_processor.train_and_evaluate_model(X_train, X_test, y_train, y_test, model_name, feature_names)

        if model_results.get("error"):
            results_children.append(html.P(f"Error training/evaluating {model_name}: {model_results['error']}", style={'color': 'red'}))
            continue

        if model_results.get("feature_importances"):
            imp_df = pd.DataFrame(list(model_results["feature_importances"].items()), columns=['Feature', 'Importance']).sort_values(by="Importance", ascending=False)
            top_n = min(len(imp_df), 15)
            fig_imp = px.bar(imp_df.head(top_n), x='Importance', y='Feature', orientation='h', title=f"Top {top_n} Feature Importances")
            fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
            results_children.append(dcc.Graph(figure=fig_imp))

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

        metrics = {
            "Accuracy": model_results.get("accuracy"),
            "Precision": model_results.get("precision"),
            "Recall": model_results.get("recall"),
            "F1 Score": model_results.get("f1_score"),
            "ROC AUC": model_results.get("roc_auc")
        }
        metrics_df = pd.DataFrame([metrics]).T.reset_index()
        metrics_df.columns = ["Metric", "Value"]
        metrics_table = dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in metrics_df.columns],
            data=metrics_df.to_dict('records'),
            style_cell={'textAlign': 'left'},
            style_header={'fontWeight': 'bold'}
        )
        results_children.append(html.H5("Evaluation Metrics:"))
        results_children.append(metrics_table)

        roc_data = model_results.get("roc_curve")
        if roc_data and roc_data.get("fpr") is not None and roc_data.get("tpr") is not None:
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=roc_data["fpr"], y=roc_data["tpr"], mode='lines', name=f'ROC Curve (AUC = {model_results.get("roc_auc", 0.0):.2f})'))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Chance', line=dict(dash='dash')))
            fig_roc.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', legend=dict(x=0.6, y=0.1))
            results_children.append(dcc.Graph(figure=fig_roc))

        results_children.append(html.Hr())

    if not results_children:
        return html.P("No results to display. Check selections or data.", style={'color': 'orange'})

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
        if selected_location != 'all':
             actual_sensor_ids_to_filter = list(sensor_df[sensor_df['location_id'] == selected_location]['sensor_id'].unique())
    else:
        actual_sensor_ids_to_filter = [sid for sid in selected_sensor_ids if sid != 'all']

    if actual_sensor_ids_to_filter:
         query_parts.append(f"sensor_id in {actual_sensor_ids_to_filter}")

    if query_parts:
        current_df = sensor_df.query(" and ".join(query_parts))
    else:
        current_df = sensor_df.copy()

    if current_df.empty:
        fig = go.Figure()
        fig.update_layout(title_text="No data found for selected criteria.")
        return fig

    title_parts = []
    sensors_in_title = current_df['sensor_id'].unique()
    if len(sensors_in_title) > 5:
        for sid in sensors_in_title[:3]:
            result = current_df[current_df['sensor_id'] == sid]['result'].iloc[0]
            title_parts.append(f"{sid} ({result})")
        title_parts.append(f"...and {len(sensors_in_title) - 3} more sensors")
    else:
        for sid in sensors_in_title:
            result = current_df[current_df['sensor_id'] == sid]['result'].iloc[0]
            title_parts.append(f"{sid} ({result})")

    location_str = selected_location if selected_location != 'all' else "All Locations"
    if 'all' in selected_sensor_ids:
        sensors_str = "All Sensors"
    else:
        sensors_str = ", ".join(title_parts)

    title = f"Waveforms for Location {location_str} - Sensors: {sensors_str}"

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

    fig.update_layout(legend_title="Sensor ID")
    return fig

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8050)
