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

    dcc.Graph(id='waveform-graph')
])

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
