import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import base64
import io
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server

# Global variables
data = None
model = None
X = None

# Layout
app.layout = html.Div([
    # Upload Component
    html.H1("Upload Dataset"),
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    html.Div(id='upload-output'),

    # Target Selection
    html.H3("Select Target Variable"),
    dcc.Dropdown(id='target-dropdown', placeholder="Select Target Variable"),
    html.Div(id='dropdown-output'),

    # Bar Charts
    html.H3("Bar Charts"),
    html.Label("Select Categorical Variable"),
    dcc.RadioItems(id='categorical-radio'),
    dcc.Graph(id='barchart1'),
    dcc.Graph(id='barchart2'),

    # Train Component
    html.H3("Train Model"),
    dcc.Checklist(id='feature-checklist', options=[]),
    html.Button("Train Model", id='train-button'),
    html.Div(id='train-output'),

    # Predict Component
    html.H3("Make Predictions"),
    dcc.Input(id='predict-input', type='text', placeholder='Enter values (comma-separated)'),
    html.Button("Predict", id='predict-button'),
    html.Div(id='predict-output'),
])

# Callbacks

# Handle file upload
@app.callback(
    Output('upload-output', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def handle_file(contents, filename):
    global data
    if contents:
        try:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            data = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            return html.Div([
                html.H5(f"Uploaded File: {filename}"),
                html.Hr(),
                html.Div("Data Preview:"),
                html.Pre(data.head().to_string(), style={'whiteSpace': 'pre-wrap'}),
            ])
        except Exception as e:
            return html.Div([
                html.H5(f"Error processing file: {filename}"),
                html.P(f"{str(e)}")
            ])
    return "No file uploaded yet."

# Update dropdown for target variable
@app.callback(
    Output('target-dropdown', 'options'),
    Input('upload-output', 'children')
)
def update_target_dropdown(_):
    if data is not None:
        numerical_columns = data.select_dtypes(include=['number']).columns
        return [{'label': col, 'value': col} for col in numerical_columns]
    return []

# Update feature checklist
@app.callback(
    Output('feature-checklist', 'options'),
    Input('upload-output', 'children')
)
def update_feature_checklist(_):
    if data is not None:
        feature_columns = data.select_dtypes(include=['number']).columns
        return [{'label': col, 'value': col} for col in feature_columns]
    return []

# Update bar charts
@app.callback(
    [Output('categorical-radio', 'options'), Output('barchart1', 'figure')],
    Input('target-dropdown', 'value')
)
def update_categorical_chart(target):
    if target and data is not None:
        categorical_columns = data.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            avg_values = data.groupby(categorical_columns[0])[target].mean()
            fig1 = {
                'data': [{'x': avg_values.index, 'y': avg_values.values, 'type': 'bar'}],
                'layout': {'title': 'Average Target by Category'}
            }
            options = [{'label': col, 'value': col} for col in categorical_columns]
            return options, fig1
    return [], {}

@app.callback(
    Output('barchart2', 'figure'),
    Input('target-dropdown', 'value')
)
def update_correlation_chart(target):
    if target and data is not None:
        try:
            numerical_data = data.select_dtypes(include=['number'])
            correlations = numerical_data.corr()[target].abs().sort_values(ascending=False).drop(target)
            if correlations.empty:
                return {
                    'data': [],
                    'layout': {'title': 'No Correlations Found'}
                }
            fig2 = {
                'data': [{'x': correlations.index, 'y': correlations.values, 'type': 'bar'}],
                'layout': {'title': 'Correlation Strength with Target'}
            }
            return fig2
        except Exception as e:
            return {
                'data': [],
                'layout': {'title': f'Error Generating Chart: {str(e)}'}
            }
    return {
        'data': [],
        'layout': {'title': 'Select a Target Variable'}
    }

# Train model
@app.callback(
    Output('train-output', 'children'),
    Input('train-button', 'n_clicks'),
    State('feature-checklist', 'value'),
    State('target-dropdown', 'value')
)
def train_model(n_clicks, selected_features, target_variable):
    global model, X
    if n_clicks:
        if not selected_features:
            return "Please select at least one feature."

        if not target_variable:
            return "Please select a target variable."

        try:
            X = data[selected_features]
            y = data[target_variable]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)

            return f"Model trained successfully! R² Score: {r2:.2f}"
        except Exception as e:
            return f"Error in training: {str(e)}"
    return "Click 'Train Model' to train after selecting features and target."

# Predict target
@app.callback(
    Output('predict-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('predict-input', 'value')
)
def predict_target(n_clicks, input_values):
    global model
    if n_clicks:
        try:
            input_data = [float(x) for x in input_values.split(',')]
            input_df = pd.DataFrame([input_data], columns=X.columns)
            prediction = model.predict(input_df)
            return f"Predicted Target Value: {prediction[0]:.2f}"
        except Exception as e:
            return f"Error in prediction: {str(e)}"
    return "No prediction made yet."

if __name__ == "__main__":
    app.run_server(debug=True)
