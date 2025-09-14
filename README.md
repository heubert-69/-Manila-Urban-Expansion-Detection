# Manila-Urban-Expansion-Detection
A machine learning web application for predicting urban areas from satellite imagery spectral data. This tool uses a pre-trained Random Forest model to classify urban and non-urban areas based on Landsat spectral features.

🌟 Features
📊 CSV-based Prediction: Upload CSV files with spectral features for urban classification

🎯 Pre-trained Model: Uses a Random Forest classifier trained on Manila urban data

📈 Interactive Visualizations: Multiple charts and graphs for result analysis

📱 Web Interface: User-friendly Streamlit interface

📥 Download Results: Export predictions as CSV files

🌍 Spatial Analysis: Optional geographic coordinate support

🎪 Confidence Scoring: Quality assessment for each prediction

🏗️ Technology Stack
Technology	Purpose	Version
Python	Backend language	3.8+
Streamlit interface framework	≥3.50.0
Scikit-learn	Machine learning library	≥1.0.0
Pandas	Data processing	≥1.3.0
NumPy	Numerical computations	≥1.21.0
Matplotlib	Data visualization	≥3.5.0
Pickle	Model serialization	Built-in
Hugging Face	Deployment platform	-
📋 Required CSV Format
Essential Columns:
```csv
B1_coastal,B2_blue,B3_green,B4_red,B5_nir,B6_swir1,B7_swir2,NDVI,NDBI,NDWI,brightness,ratio_swir_nir,ratio_nir_red
```
Optional Columns:
```csv
longitude,latitude  # For spatial visualization
```
Example CSV Structure:
```csv
B1_coastal,B2_blue,B3_green,B4_red,B5_nir,B6_swir1,B7_swir2,NDVI,NDBI,NDWI,brightness,ratio_swir_nir,ratio_nir_red
0.123,0.145,0.167,0.189,0.234,0.456,0.378,0.234,0.456,0.123,0.289,1.234,1.456
0.134,0.156,0.178,0.201,0.245,0.467,0.389,0.245,0.467,0.134,0.301,1.245,1.467
```
🚀 Installation & Setup
Local Development:
Clone the repository:

```bash
git clone <your-repo-url>
cd satellite-urban-prediction
```
Create virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
Install dependencies:

```bash
pip install -r requirements.txt
```
Add your trained model:

```bash
# Place your trained model file in the root directory
# File should be named: model.pkl
Run the application:

python final_app.py
```

🎯 Model Training Information
Expected Features:
The model expects 13 spectral features in this exact order:

B1_coastal - Coastal aerosol band

B2_blue - Blue band

B3_green - Green band

B4_red - Red band

B5_nir - Near Infrared band

B6_swir1 - Short-wave Infrared 1

B7_swir2 - Short-wave Infrared 2

NDVI - Normalized Difference Vegetation Index

NDBI - Normalized Difference Built-up Index

NDWI - Normalized Difference Water Index

brightness - Average brightness

ratio_swir_nir - SWIR to NIR ratio

ratio_nir_red - NIR to Red ratio

Model Architecture:
Algorithm: Random Forest Classifier

Trees: 100 estimators

Max Depth: 10 levels

Training Data: Manila urban/rural areas

Accuracy: >85% on test data

📊 Output Results
Visual Outputs:
Prediction Distribution - Bar chart of urban vs non-urban predictions

Probability Distribution - Histogram of prediction confidence

Spatial Distribution - Geographic plot (if coordinates provided)

Confidence Levels - Quality assessment of predictions

Data Outputs:
Prediction Label: Urban/Non-Urban classification

Probability Score: Confidence score (0-1)

Confidence Level: Qualitative assessment (Low/Medium/High/Very High)

Geographic Coordinates: If provided in input

Downloadable Files:
Complete results CSV with all predictions

Preserves all original input data plus predictions

🎮 How to Use
Prepare Your Data:

Collect spectral data from Landsat imagery

Calculate required indices (NDVI, NDBI, NDWI)

Format as CSV with expected column names

Run Prediction:

Upload CSV file through the web interface

Click "Predict Urban Areas"

View interactive results and visualizations

Analyze Results:

Review prediction statistics

Examine confidence levels

Download results for further analysis

Interpret Results:

Urban areas: High NDBI, moderate brightness

Non-urban: High NDVI (vegetation) or other features

Confidence scores indicate prediction reliability

🔧 Customization
Modifying Expected Features:
Edit the expected_features list in app.py:

```python
expected_features = [
    'B1_coastal', 'B2_blue', 'B3_green', 'B4_red', 
    'B5_nir', 'B6_swir1', 'B7_swir2',
    'NDVI', 'NDBI', 'NDWI', 'brightness', 
    'ratio_swir_nir', 'ratio_nir_red'
]
```
Adding New Visualizations:
Extend the plotting section in predict_urbanization_csv() function:

```python
# Add new subplot
ax5 = plt.subplot(2, 3, 5)  # Adjust grid as needed
ax5.plot(new_data)
ax5.set_title('New Visualization')
```
Model Replacement:
Replace model.pkl with your new model file. Ensure it has:

.model attribute: Trained classifier

.scaler attribute: Fitted StandardScaler

.feature_names attribute: List of expected features

🐛 Troubleshooting
Common Issues:
Missing Model File:

```text
❌ Pickle file urban_model.pkl not found
Solution: Ensure urban_model.pkl is in the root directory
```

CSV Format Error:

```text
❌ Missing features in CSV: B5_nir, NDVI, ...
Solution: Check column names match expected features
```
```text
Memory Issues:
Solution: Reduce sample size or upgrade Hugging Face Space hardware
```
```text
Visualization Errors:
Solution: Check for NaN values in input data
```
```text
Performance Tips:
Use smaller CSV files for testing (<10,000 rows)
```

Pre-calculate spectral indices before upload

Ensure numeric columns don't contain text values

Handle missing values before upload

📈 Example Use Cases
Urban Planning:
Monitor urban expansion over time

Identify potential development areas

Assess urban density patterns

Environmental Research:
- Study urban heat island effects

- Analyze vegetation loss in urban areas

- Monitor water body changes near cities

Academic Projects:
- Remote sensing coursework

- Machine learning demonstrations

- Geographic information systems (GIS) studies

🤝 Contributing
Fork the repository

Create a feature branch

Make your changes

Test thoroughly

Submit a pull request

Development Priorities:
Add support for multiple model types

Implement batch processing for large files

Add temporal analysis capabilities

Include more visualization options

Support for additional satellite data formats

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Landsat Program for satellite imagery data

Scikit-learn team for machine learning tools

Gradio team for the web framework

Hugging Face for deployment platform

📞 Support
For questions and support:

Check the troubleshooting section above

Review example CSV formats

Ensure model file is properly formatted

Verify all dependencies are installed

⭐ If you find this project useful, please give it a star on GitHub!

Built with ❤️ for urban planning, environmental research and financial forecasting
