import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import base64
import io
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server

# Global variables
data = None
model = None
X_columns = []

# Layout
app.layout = html.Div([
    # Upload File and Target Selection Section
    html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Button("Upload File", style={'fontSize': '14px', 'width': '100%'}),
            style={'textAlign': 'center', 'marginBottom': '10px'}
        ),
        html.Div([
            html.Label("Select Target:", style={'fontSize': '14px', 'marginRight': '10px', 'display': 'inline-block'}),
            dcc.Dropdown(id='target-dropdown', placeholder="Select Target Variable", style={'width': '200px', 'display': 'inline-block', 'fontSize': '14px'}),
        ], style={'textAlign': 'center', 'marginBottom': '20px'}),
    ], style={'backgroundColor': '#f0f0f0', 'padding': '10px', 'marginBottom': '20px'}),

    # Graphs Section
    html.Div([
        html.Div([
            dcc.RadioItems(id='categorical-radio', options=[], inline=True, style={'marginBottom': '10px', 'fontSize': '14px'}),
            dcc.Graph(id='barchart1', config={'displayModeBar': False}, style={'height': '400px'}),
        ], style={'display': 'inline-block', 'width': '49%', 'marginRight': '1%', 'verticalAlign': 'top'}),
        html.Div([
            dcc.Graph(id='barchart2', config={'displayModeBar': False}, style={'height': '400px'}),
        ], style={'display': 'inline-block', 'width': '49%', 'verticalAlign': 'top'}),
    ], style={'textAlign': 'center', 'marginBottom': '20px', 'backgroundColor': 'white'}),

    # Train Model Section
    html.Div([
        dcc.Checklist(id='feature-checklist', options=[], inline=True, style={'marginBottom': '10px', 'fontSize': '14px'}),
        html.Button("Train", id='train-button', style={'fontSize': '14px', 'width': '150px', 'marginBottom': '10px'}),
        html.Div(id='train-output', style={'fontSize': '14px'}),
    ], style={'textAlign': 'center', 'marginBottom': '20px', 'backgroundColor': 'white'}),

    # Predict Section
    html.Div([
        dcc.Input(id='predict-input', type='text', placeholder='Enter values (comma-separated)', style={'marginRight': '10px', 'width': '40%', 'fontSize': '14px'}),
        html.Button("Predict", id='predict-button', style={'fontSize': '14px'}),
        html.Div(id='predict-output', style={'display': 'inline-block', 'fontSize': '14px', 'marginLeft': '10px'}),
    ], style={'textAlign': 'center', 'backgroundColor': 'white'}),
])

# Callbacks remain unchanged
@app.callback(
    [Output('target-dropdown', 'options'),
     Output('feature-checklist', 'options'),
     Output('categorical-radio', 'options')],
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
            numerical_columns = data.select_dtypes(include=['number']).columns
            categorical_columns = data.select_dtypes(include=['object']).columns
            return (
                [{'label': col, 'value': col} for col in numerical_columns],
                [{'label': col, 'value': col} for col in data.columns],
                [{'label': col, 'value': col} for col in categorical_columns]
            )
        except Exception:
            return [], [], []
    return [], [], []

@app.callback(
    Output('barchart1', 'figure'),
    [Input('target-dropdown', 'value'),
     Input('categorical-radio', 'value')]
)
def update_barchart1(target, selected_categorical):
    if target and selected_categorical and data is not None:
        try:
            grouped_data = data.groupby(selected_categorical)[target].mean().dropna()
            num_categories = len(grouped_data)
            bar_width = 0.4 if num_categories > 10 else 0.6
            return {
                'data': [{
                    'x': grouped_data.index,
                    'y': grouped_data.values,
                    'type': 'bar',
                    'marker': {'color': 'lightblue'},
                    'width': [bar_width] * num_categories,
                }],
                'layout': {
                    'title': f"Average {target} by {selected_categorical}",
                    'xaxis': {'title': selected_categorical, 'tickangle': 45, 'automargin': True},
                    'yaxis': {'title': f"Average {target}"},
                },
            }
        except Exception as e:
            return {'data': [], 'layout': {'title': f"Error: {str(e)}"}}
    return {'data': [], 'layout': {'title': 'Select both target and categorical variables'}}

@app.callback(
    Output('barchart2', 'figure'),
    Input('target-dropdown', 'value')
)
def update_barchart2(target):
    if target and data is not None:
        numerical_data = data.select_dtypes(include=['number'])
        correlations = numerical_data.corr()[target].abs().sort_values(ascending=False).drop(target)
        return {
            'data': [{'x': correlations.index, 'y': correlations.values, 'type': 'bar'}],
            'layout': {
                'title': f"Correlation Strength with {target}",
                'xaxis': {'title': "Numerical Variables"},
                'yaxis': {'title': "Correlation Strength"},
            },
        }
    return {'data': [], 'layout': {'title': 'Select a Target Variable'}}

@app.callback(
    Output('train-output', 'children'),
    [Input('train-button', 'n_clicks')],
    [State('feature-checklist', 'value'),
     State('target-dropdown', 'value')]
)
def train_model(n_clicks, selected_features, target_variable):
    global model, X_columns
    if n_clicks:
        try:
            X = data[selected_features]
            y = data[target_variable]
            X_columns = selected_features
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', SimpleImputer(strategy='mean'), X.select_dtypes(include=['number']).columns),
                    ('cat', Pipeline(steps=[
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('onehot', OneHotEncoder(handle_unknown='ignore'))
                    ]), X.select_dtypes(include=['object']).columns)
                ],
                remainder='passthrough'
            )
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', LinearRegression())])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            pipeline.fit(X_train, y_train)
            model = pipeline
            y_pred = pipeline.predict(X_test)
            return f"The R² score is: {r2_score(y_test, y_pred):.2f}"
        except Exception as e:
            return f"Error: {str(e)}"
    return "Click 'Train' to train the model."

@app.callback(
    Output('predict-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('predict-input', 'value')]
)
def predict_target(n_clicks, input_values):
    if n_clicks and model:
        try:
            input_data = [float(x.strip()) for x in input_values.split(',')]
            input_df = pd.DataFrame([input_data], columns=X_columns)
            prediction = model.predict(input_df)
            return f"Predicted target is: {prediction[0]:.2f}"
        except Exception as e:
            return f"Error: {str(e)}"
    return "No prediction made yet."

#if __name__ == '__main__':
 #   app.run_server(debug=True)
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8050))  
    app.run_server(debug=True, port=port, host='0.0.0.0')
