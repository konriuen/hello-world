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
        html.Label("Select Sensor ID:"),
        dcc.Dropdown(
            id='sensor-dropdown',
            options=[{'label': sid, 'value': sid} for sid in sensor_df['sensor_id'].unique()],
            value=[sensor_df['sensor_id'].unique()[0]],  # Default value (now a list)
            multi=True  # Allow multiple selections
        ),
    ]),

    dcc.Graph(id='waveform-graph')
])

# Callback to update graph
@app.callback(
    Output('waveform-graph', 'figure'),
    [Input('sensor-dropdown', 'value')]
)
def update_graph(selected_sensor_ids): # Renamed to reflect multiple IDs
    if not selected_sensor_ids: # Check if the list is empty
        fig = px.line(title="Select one or more sensors to view their waveforms")
        fig.update_layout(
            xaxis_title="Timestamp",
            yaxis_title="Value",
        )
        return fig

    # Filter DataFrame for selected sensors
    filtered_df = sensor_df[sensor_df['sensor_id'].isin(selected_sensor_ids)]

    # Use Plotly Express for direct plotting with color mapping based on 'sensor_id' and 'result'
    # We need to ensure the 'result' is correctly associated for color.
    # Let's create a title that lists the selected sensors and their results.

    title_parts = []
    for sid in selected_sensor_ids:
        result = sensor_df[sensor_df['sensor_id'] == sid]['result'].iloc[0]
        title_parts.append(f"{sid} ({result})")
    title = f"Waveforms for {', '.join(title_parts)}"

    # Define a color map for results to be used by Plotly Express
    # We will map sensor_id to a color based on its result.
    # Plotly Express's `color` argument usually refers to a column.
    # For line colors, we can iterate and add traces, or customize the color_discrete_map.

    fig = px.line(filtered_df, x='timestamp', y='value', color='sensor_id',
                  title=title, labels={'sensor_id': 'Sensor'})

    # Customize line colors based on the 'result' of each sensor
    # Plotly Express assigns colors automatically. We need to override them.
    # Create a mapping from sensor_id to the desired color (green for OK, red for NG)
    color_discrete_map = {}
    for sensor_id in filtered_df['sensor_id'].unique():
        result_status = sensor_df[sensor_df['sensor_id'] == sensor_id]['result'].iloc[0]
        if result_status == "OK":
            color_discrete_map[sensor_id] = "green"
        elif result_status == "NG":
            color_discrete_map[sensor_id] = "red"
        else:
            color_discrete_map[sensor_id] = "blue" # Fallback

    # Re-create the figure with the explicit color mapping for sensors
    # Note: px.line creates traces per unique value in 'color' column.
    # We need to ensure the legend and colors are correctly applied.

    # A more robust way for custom colors per trace when 'color' is used for grouping:
    # Iterate through selected sensors and add traces individually if px struggles with direct mapping.
    # However, px should handle it if we provide color_discrete_map.
    # Let's try with color_discrete_map first.

    # Rebuild figure with explicit color mapping for the 'sensor_id' column
    fig = px.line(filtered_df, x='timestamp', y='value', color='sensor_id',
                  title=title, labels={'sensor_id': 'Sensor'},
                  color_discrete_map=color_discrete_map)

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
