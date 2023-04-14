# Script that downloads the weather time series dataset recorded by the Max Planck Institute for Biogeochemistry

import tensorflow as tf
import os
import pandas as pd

zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)
csv_stream = pd.read_csv(csv_path)
csv_stream.to_parquet('timeseries-2009-2016.parquet')
