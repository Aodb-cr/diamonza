import gradio as gr
import joblib
import pandas as pd

# Load the model and unique values
model = joblib.load('model.joblib')
unique_values = joblib.load('unique_values.joblib')

# Define the prediction function
def predict(carat, cut, color, clarity, depth, table, x, y, z):
    # Prepare the input data for prediction
    input_data = pd.DataFrame({
        'carat': [float(carat)],
        'cut': [cut],
        'color': [color],
        'clarity': [clarity],
        'depth': [float(depth)],
        'table': [float(table)],
        'x': [float(x)],
        'y': [float(y)],
        'z': [float(z)]
    })
    
    # Perform the prediction
    prediction = model.predict(input_data)
    
    return prediction[0]

# Create the Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Carat"),
        gr.Dropdown(choices=['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'], label="Cut"),
        gr.Dropdown(choices=['D', 'E', 'F', 'G', 'H', 'I', 'J'], label="Color"),
        gr.Dropdown(choices=['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'], label="Clarity"),
        gr.Textbox(label="Depth"),
        gr.Textbox(label="Table"),
        gr.Textbox(label="X"),
        gr.Textbox(label="Y"),
        gr.Textbox(label="Z")
    ],
    outputs="text",
    title="Diamond Price Predictor",
    description="Enter the features of the diamond to predict its price."
)

# Launch the app
interface.launch()
