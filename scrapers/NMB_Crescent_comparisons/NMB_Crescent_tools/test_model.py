import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt


def train_models_for_bins(df, dir_bin_column='dir_bin', output_dir='models/'):
    """
    Train a separate model for each wind direction bin.

    Parameters:
        df (pd.DataFrame): Filtered data.
        dir_bin_column (str): Column for wind direction bins.
        output_dir (str): Directory to save the trained models.

    Returns:
        dict: A dictionary with bin labels as keys and model RMSE as values.
    """
    os.makedirs(output_dir, exist_ok=True)
    bin_models = {}
    bin_metrics = {}

    for bin_label in df[dir_bin_column].unique():
        # Filter data for the current bin
        bin_data = df[df[dir_bin_column] == bin_label]

        if len(bin_data) < 20:  # Skip bins with insufficient data
            print(f"Skipping bin {bin_label} due to insufficient data.")
            continue

        # Define features and target
        X = bin_data[['cres WS']]
        y = bin_data['WS Difference']

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        # Save the model
        model_path = os.path.join(output_dir, f'bin_{bin_label}_model.pkl')
        joblib.dump(model, model_path)

        # Log metrics
        bin_models[bin_label] = model
        bin_metrics[bin_label] = rmse

        #print(f"Model for bin {bin_label}: RMSE = {rmse:.2f} (saved to {model_path})")

    return bin_metrics


def predict_cres_ws(nmb_ws, nmb_dir, dir_bin_column='dir_bin', output_dir='models/'):
    """
    Predict Crescent wind speed (cres WS) given NMB wind speed and direction.

    Parameters:
        nmb_ws (float): Wind speed at NMB.
        nmb_dir (float): Wind direction at NMB.
        dir_bin_column (str): Column for wind direction bins.
        output_dir (str): Directory where the models are stored.

    Returns:
        float: Predicted wind speed at Crescent.
    """
    # Map NMB direction to the correct bin
    bin_size = 5
    dir_bins = np.arange(0, 360 + bin_size, bin_size)
    dir_labels = [(dir_bins[i] + dir_bins[i + 1]) / 2 for i in range(len(dir_bins) - 1)]
    dir_bin = dir_labels[int(nmb_dir / bin_size)]

    # Load the trained model for the bin
    model_path = os.path.join(output_dir, f'bin_{dir_bin}_model.pkl')
    if not os.path.exists(model_path):
        raise ValueError(f"No model found for direction bin {dir_bin}. Ensure the model is trained.")

    model = joblib.load(model_path)

    # Ensure the input has the expected feature name
    X = pd.DataFrame([[nmb_ws]], columns=['cres WS'])
    ws_diff_pred = model.predict(X)[0]


    # Calculate Crescent wind speed
    cres_ws_pred = nmb_ws - ws_diff_pred
    return cres_ws_pred


if __name__ == "__main__":
    # Define paths
    filtered_data_path = '/Users/jayriihiluoma/Documents/python/scrapers/NMB_Crescent_comparisons/NMB_Crescent_data/filtered_combined_data.csv'
    output_dir = '/Users/jayriihiluoma/Documents/python/scrapers/NMB_Crescent_comparisons/NMB_Crescent_data/bin_models'


    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load filtered data
    df = pd.read_csv(filtered_data_path)

    # Ensure 'dir_bin' column is properly formatted
    df['dir_bin'] = df['dir_bin'].astype(str)

    # Train models for each bin
    metrics = train_models_for_bins(df, output_dir=output_dir)

    '''# Interactive example prediction
    while True:
        try:
            # Prompt the user for wind speed and direction
            nmb_avg_ws = float(input("Enter NMB average wind speed (knots): ").strip())
            nmb_max_ws = float(input("Enter NMB max wind speed (knots): ").strip())
            nmb_dir = float(input("Enter NMB wind direction (degrees): ").strip())

            # Predict Crescent average wind speed
            cres_avg_ws = predict_cres_ws(nmb_avg_ws, nmb_dir, output_dir=output_dir)
            
            # Predict Crescent max wind speed
            cres_max_ws = predict_cres_ws(nmb_max_ws, nmb_dir, output_dir=output_dir)

            # Display the predictions
            print(f"\nPredicted Crescent Average Wind Speed: {cres_avg_ws:.2f} knots")
            print(f"Predicted Crescent Max Wind Speed: {cres_max_ws:.2f} knots")

        except ValueError as e:
            # Handle cases where the input is invalid
            print(f"Error: {e}. Please enter valid numerical values.")
        except KeyboardInterrupt:
            # Gracefully exit on Ctrl+C
            print("\nExiting...")
            break
        except Exception as e:
            # Catch any other unexpected errors
            print(f"An unexpected error occurred: {e}")
            break'''

    while True:
        try:
            nmb_avg_ws = float(input("Enter NMB average wind speed (knots): ").strip())
            nmb_max_ws = float(input("Enter NMB max wind speed (knots): ").strip())
            nmb_dir = float(input("Enter NMB wind direction (degrees): ").strip())

            # Predict Crescent wind speeds
            cres_avg_ws = predict_cres_ws(nmb_avg_ws, nmb_dir, output_dir=output_dir)
            cres_max_ws = predict_cres_ws(nmb_max_ws, nmb_dir, output_dir=output_dir)

            # Display predictions
            print(f"\nPredicted Crescent Average Wind Speed: {cres_avg_ws:.2f} knots")
            print(f"Predicted Crescent Max Wind Speed: {cres_max_ws:.2f} knots")

            # Append the data to the Google Sheet
            pred_sheet.append_row([nmb_avg_ws, nmb_max_ws, nmb_dir, cres_avg_ws, cres_max_ws])
            print("Prediction data written to 'pred_cresc' sheet.")

            # End the terminal session after processing one set of inputs
            print("\nExiting after processing the input.")
            break

        except ValueError as e:
            print(f"Error: {e}. Please enter valid numerical values.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break