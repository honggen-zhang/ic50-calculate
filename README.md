# IC50 Calculator

A web-based tool for calculating IC50 values using 3PL, 4PL, or 5PL logistic models with robust fitting and outlier detection.

## Features

- **Multiple Model Support**: 3-Parameter (3PL), 4-Parameter (4PL), and 5-Parameter (5PL) Logistic models
- **Robust Fitting**: Iterative reweighting and Huber loss methods
- **Outlier Detection**: Multiple methods (Cook's distance, IQR, Z-score, Residual-based)
- **Multiple Measurements**: Support for averaging multiple experimental replicates
- **Interactive Visualization**: Real-time plotting with error bars and confidence intervals
- **Data Input Options**: Manual entry, CSV upload, or example data
- **Export Results**: Download plots and results as CSV

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. The app will open in your default web browser at `http://localhost:8501`

3. Choose your input method:
   - **Manual Entry**: Enter concentrations and responses manually
   - **CSV Upload**: Upload a CSV file with your data
   - **Example Data**: Use provided example data

4. Configure model settings in the sidebar:
   - Select model type (3PL, 4PL, or 5PL)
   - Choose robust fitting method
   - Enable/disable outlier detection
   - Select outlier detection method

5. View results:
   - IC50 value and confidence interval
   - R² value
   - Detailed model parameters
   - Interactive plot with fitted curve

6. Download results:
   - Download plot as PNG
   - Download results as CSV

## CSV Format

If uploading a CSV file, ensure it has:
- One column for concentrations
- One or more columns for responses (inhibition %)

Example CSV format:
```csv
Concentration,Measurement1,Measurement2
0.001,-10,-11
0.01,5,1
0.1,10,10
1,40,50
5,75,74
10,85,85
```

## Model Descriptions

- **3PL**: 3-Parameter Logistic model (fixed hill slope = -1)
- **4PL**: 4-Parameter Logistic model (recommended, most commonly used)
- **5PL**: 5-Parameter Logistic model (includes asymmetry parameter)

## Outlier Detection Methods

- **Cook's Distance**: Measures influence of each data point
- **IQR**: Interquartile Range method
- **Z-score**: Standard deviation-based method
- **Residual**: Based on residual analysis

## Notes

- At least 3 data points are required for fitting
- Zero concentrations are excluded by default (can be changed in settings)
- The tool automatically handles multiple measurements by calculating mean and standard deviation

## Deployment to Web (Streamlit Cloud)

Want to share this tool with others via a web link? See [DEPLOYMENT.md](DEPLOYMENT.md) for step-by-step instructions on deploying to Streamlit Cloud (free hosting).

**Quick Summary:**
1. Push your code to GitHub
2. Sign up for Streamlit Cloud (free)
3. Connect your GitHub repository
4. Deploy with one click!

Your app will be accessible at a public URL that anyone can use - no coding experience required for users.
