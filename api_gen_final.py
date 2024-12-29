import pandas as pd
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import io

# Load the trained model
model = tf.keras.models.load_model('lstm_autoencoder_model.keras')

# Initialize FastAPI app
app = FastAPI()

# Define request and response models
class PredictionRequest(BaseModel):
    csv_data: str

class PredictionResponse(BaseModel):
    alert: bool
    date_range: str

# Helper function to read CSV from string
def read_csv_from_string(csv_string: str):
    return pd.read_csv(io.StringIO(csv_string), parse_dates=['Date'], index_col='Date')

@app.post("/predict", response_model=PredictionResponse)
async def predict_anomaly(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content), parse_dates=['Date'], index_col='Date')
    # seq_len: int = Query(10, description="Length of the sequence") 
    # threshold: float = Query(0.1, description="Threshold for anomaly detection (might change in case of further data drift)")

    # Preprocess the data to match the input format of the model
    seq_len=3
    sequences = create_sequences(df.values, seq_len=3)

    # Make predictions
    predictions = model.predict(sequences)

    # Calculate loss
    data_loss = tf.keras.losses.mae(predictions, sequences)
    data_loss_mean =np.mean(data_loss,axis=1)

    # Check for anomalies (you need to define your threshold)
    threshold = 2  # Example threshold
    anomalies = data_loss_mean > threshold

    # Extract date range for anomalies
    results_dated=pd.DataFrame({'Dates':df.index,'Loss':data_loss_mean})
    potential_anom=results_dated.loc[results_dated['Loss']>threshold]

    def find_consecutive_ranges(dates, min_days=3, max_days=14):
        consecutive_ranges = []
        start_date = dates[0]
        prev_date = dates[0]

        for date in dates[1:]:
            if (date - prev_date).days != 1:
                range_length = (prev_date - start_date).days + 1
                if min_days <= range_length <= max_days:
                    consecutive_ranges.append((start_date, prev_date))
                print(f"Range: {start_date} to {prev_date} - Length: {range_length} days")
                start_date = date
            prev_date = date

        range_length = (prev_date - start_date).days + 1
        if min_days <= range_length <= max_days:
            consecutive_ranges.append((start_date, prev_date))
        print(f"Range: {start_date} to {prev_date} - Length: {range_length} days")

        return consecutive_ranges

    # Identify consecutive date ranges in the DataFrame
    consecutive_dates = potential_anom['Dates'].tolist()
    consecutive_ranges = find_consecutive_ranges(consecutive_dates)

    # Print consecutive ranges to debug
    # print(f"Consecutive Ranges: {consecutive_ranges}")

    # Select dates that are part of the desired ranges
    selected_dates = []
    for start_date, end_date in consecutive_ranges:
        selected_dates.extend(pd.date_range(start=start_date, end=end_date).tolist())

    selected_df = potential_anom[potential_anom['Dates'].isin(selected_dates)]

    # Display the selected DataFrame
    print(selected_df.Dates)


    











    # date_range = f"{anomalous_dates.min()} - {anomalous_dates.max()}"

    return PredictionResponse(alert=any(anomalies), date_range=selected_df.Dates)

# Helper function to create sequences
def create_sequences(data, seq_len):
    sequences = []
    for i in range(len(data) - seq_len + 1):
        sequences.append(data[i:i + seq_len])
    return np.array(sequences)
