# Noise analysis and prediction in North American subway stations

## Summary
Source code and dataset for our paper titled "Noise Analysis and Prediction in North American Subway Stations". This code works on a dataset that was collected from various subway station across New York City. After cleaning the original collection by removing duplicate or empty rows, we processed the dataset for neural network training. In summary, categorical (e.g. non-numerical) columns were turned into encoded representation to allow the NN distinguish among them and create comparisons when necessary.

## Files
* ```leq_db.py```: standalone script that preprocesses the dataset and trains a neural network to predict Leq_dB levels given 'Borough', 'Track_Type', 'Platform_Occupancy', 'Station_Type', and 'Station_Width'. Provides a ```matplotlib``` visual representation of the learning and prediction process.

* ```lmax_db.py```: standalone script that preprocesses the dataset and trains a neural network to predict Lmax_dB levels given 'Borough', 'Track_Type', 'Platform_Occupancy', 'Station_Type', and 'Station_Width'. Provides a ```matplotlib``` visual representation of the learning and prediction process.

* ```Cleaned_Sound_Measurement_Data.xlsx```: Our dataset which includes but is not limited to "Lmax_dB" and "Leq_dB" noise levels recorded at subway stations. We use some of the other columns to train a model and use it to predict "Lmax_dB" and "Leq_dB" noise levels.

## Usage
This code is made to run easily on Google Colaboratory servers which are provided to the general public free of charge. Simply copy-and-paste the content in each file onto separate notebooks and click run. Or, given all the necessary libraries are installed on your offline workstation, run each scrip: ```py leq_db.py``` or ```py lmax_db.py```

## Requirements
Supported platform: (Preferred) Google Colaboratory, Linux (Since tensorflow does not run on Windows)
Languages: Python
Libraries: Numpy, Pandas, sklearn, tensorflow

## License
This source code is licensed under the MIT Licesnse. PLease mention or give credit to our paper or this repository if you refer to any of the code here.
