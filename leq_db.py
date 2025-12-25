import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Dense, PReLU, LeakyReLU, ReLU, BatchNormalization, ELU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

dataset_loc = "[dataset_location]"
dataframe = pd.read_excel(dataset_loc)
dtlist = dataframe['Datetime']
date = []
time = []
for item in dtlist:
    sdate, stime = str(item).split(sep=' ')
    date.append(sdate)
    time.append(stime)
dataframe['time'] = time
dataframe['date'] = date
dataframe.drop('Datetime', axis=1, inplace=True)

# clean the platform crowd size column. Some 'M' values have a leading whitespace
# Assuming 'Platform_Occupancy' is the column name
platform_occupancy = 'Platform_Occupancy'
for i in range(0, len(dataframe[platform_occupancy])):
    dataframe.loc[i, platform_occupancy] = dataframe[platform_occupancy][i].lstrip()

# standardize time and add an Hour column
dataframe['time'] = pd.to_datetime(dataframe['time'], format='%H:%M:%S')
# Extract the hour from the datetime
dataframe['hour'] = dataframe['time'].dt.hour

categorical_columns = [
    'Borough',
    'Track_Type',
    'Platform_Occupancy',
    'Station_Type',
    'Station_Width',
]
''' Convert non-numerical columns to binary encoding '''
dataframe_encoded = pd.get_dummies(dataframe, columns=categorical_columns)
''' Alternative non-numerical encoding - worse results '''
'''
label_encoder = LabelEncoder()
for category in categorical_columns:
    dataframe[category + '_Encoded'] = label_encoder.fit_transform(dataframe[category])
dataframe = dataframe.drop(columns=categorical_columns)
'''

''' Create X, Y datasets for model training'''
ignore_cols = ['Leq_dB', 'Lmax_dB', 'Platform', 'Operative_Lines', 'date', 'time', 'hour']
X = dataframe_encoded.drop(columns=ignore_cols)
y = dataframe_encoded['Leq_dB']
for col in X.columns:
    if X[col].dtype == 'bool':
        X[col] = X[col].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

''' Construct the neural network, compile, and commence training'''
model = Sequential([
    Dense(128, input_dim=X.shape[1]),
    ELU(alpha=0.1),

    Dense(64),
    ELU(alpha=0.1),

    Dense(32),
    ELU(alpha=0.1),

    Dense(1)  # Output layer
])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test), callbacks=[reduce_lr])

''' Display the mean-absolute-error and learning rate via a visual plot'''
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('Leq_dB Prediction Loss')
plt.legend()
plt.show()
