import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import gradio as gr

# Assuming your data for soil and crop type is already available as a DataFrame
# Encode the Soil Type and Crop Type
encode_soil = LabelEncoder()
encode_crop = LabelEncoder()

# Dummy DataFrame creation (you should replace this with your actual data)
soil_types = ['Sandy','Loamy','Black','Red','Clayey']
crop_types = ['Maize','Sugarcane','Cotton','Tobacco','Paddy','Barley','Wheat','Millets','Oil seeds','Pulses','Ground Nuts']

# Fit encoders to the sample data
encode_soil.fit(soil_types)
encode_crop.fit(crop_types)

# Create DataFrame to store original and encoded values
Soil_Type = pd.DataFrame(zip(encode_soil.classes_, encode_soil.transform(encode_soil.classes_)), columns=['Original', 'Encoded']).set_index('Original')
Crop_Type = pd.DataFrame(zip(encode_crop.classes_, encode_crop.transform(encode_crop.classes_)), columns=['Original', 'Encoded']).set_index('Original')

# Function to predict fertilizer
def predict_fertilizer(temperature, humidity, moisture, soil_type, crop_type, nitrogen, potassium, phosphorous):
    """Predicts the fertilizer based on input parameters."""
    
    # Encode soil type and crop type using the same LabelEncoder used during training
    soil_encoded = Soil_Type.loc[soil_type]['Encoded']
    crop_encoded = Crop_Type.loc[crop_type]['Encoded']

    # Load the trained model (Make sure classifier.pkl exists in the same directory)
    model = joblib.load('classifier.pkl')

    # Make prediction
    ans = model.predict([[temperature, humidity, moisture, soil_encoded, crop_encoded, nitrogen, potassium, phosphorous]])

    # Map prediction to fertilizer type
    fertilizers = {
        0: "NPK 10-26-26",
        1: "GROMOR 14-35-14",
        2: "17-17-17",
        3: "20-20",
        4: "28-28",
        5: "DAP",
        6: "Urea"
    }

    return fertilizers.get(ans[0], "Unknown Fertilizer")

# Gradio interface components
temperature = gr.Number(label="Temperature")
humidity = gr.Number(label="Humidity")
moisture = gr.Number(label="Moisture")
soil_type = gr.Dropdown(choices=list(Soil_Type.index), label="Soil Type")
crop_type = gr.Dropdown(choices=list(Crop_Type.index), label="Crop Type")
nitrogen = gr.Number(label="Nitrogen")
potassium = gr.Number(label="Potassium")
phosphorous = gr.Number(label="Phosphorous")
output = gr.Textbox(label="Recommended Fertilizer")

# Gradio interface setup
iface = gr.Interface(
    fn=predict_fertilizer,
    inputs=[temperature, humidity, moisture, soil_type, crop_type, nitrogen, potassium, phosphorous],
    outputs=output,
    title="Fertilizer Recommendation System",
    description="Enter crop and soil details to get recommended fertilizer."
)

# Launch the Gradio app
if __name__ == "__main__":
    iface.launch()
