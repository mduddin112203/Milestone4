# Machine Learning Web App

A simple web-based application for uploading datasets, selecting features, training machine learning models, and making predictions, built using **Dash**.

---

## Features
1. Upload CSV datasets for processing.
2. Dynamically select target variables and features.
3. Visualize data relationships through bar charts.
4. Train a linear regression model with selected features.
5. Predict target variable values based on input.

---

## Setup Instructions

# 1. Clone the Repository
```bash
git clone <repository-link>
cd <repository-folder>

# Ensure Python 3.8 or above is installed
python3 --version

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # For macOS/Linux
venv\Scripts\activate     # For Windows

# Use the requirements.txt file to install all dependencies
pip install -r requirements.txt

# Run the application
python app.py

# After running the app, you will see output like this:
Dash is running on http://127.0.0.1:8050/

# Open this URL in your web browser.

- Click "Drag and Drop or Select Files" to upload a CSV file.
- Ensure the file has a proper structure with headers.

- Use the dropdown to select the column you want to predict.

- Use the radio buttons to explore relationships between categorical variables and the target variable.
- The second chart shows correlations between numerical variables and the target.

- Use the checklist to select features for training.
- Click "Train Model" to train a regression model.
- An R² score will display upon success.

- Enter feature values (comma-separated) in the "Make Predictions" section.
- Click "Predict" to see the predicted target variable.
