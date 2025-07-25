import pandas as pd
import numpy as np
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def train_task_classifier(input_csv):

    df = pd.read_csv(input_csv, sep=';', parse_dates=['Date'])
    

    df['TaskType'] = df['Sheets'].apply(lambda x: 'Modeling' if pd.isna(x) else 'Drawings')
    label_encoder = LabelEncoder()
    df['TaskTypeEncoded'] = label_encoder.fit_transform(df['TaskType'])  
    

    df['Duration'] = (df['Date'] - df['Date'].min()).dt.total_seconds()  
    feature_columns = ['Duration']  
    X = df[feature_columns].values
    y = to_categorical(df['TaskTypeEncoded'])  
    

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    
    model = Sequential([
        Dense(32, activation='relu', input_dim=X.shape[1]),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(2, activation='softmax')  
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    
    return model, scaler, label_encoder


def analyze_employee_tasks(input_csv, output_excel, start_time, end_time, model, scaler, label_encoder):

    df = pd.read_csv(input_csv, sep=';', parse_dates=['Date'])
    df['TaskType'] = df['Sheets'].apply(lambda x: 'Modeling' if pd.isna(x) else 'Drawings')
    df = df[(df['Date'] >= start_time) & (df['Date'] <= end_time)]

    df['Duration'] = (df['Date'] - df['Date'].min()).dt.total_seconds()
    feature_columns = ['Duration']  # Same feature columns used for training
    X = scaler.transform(df[feature_columns].values)
    
    predictions = model.predict(X)
    df['PredictedTaskType'] = np.argmax(predictions, axis=1)  # Get the predicted class
    df['PredictedTaskType'] = label_encoder.inverse_transform(df['PredictedTaskType'])

    results = []
    for employee, group in df.groupby('Employee'):
        total_tasks = len(group)
        modeling_count = len(group[group['PredictedTaskType'] == 'Modeling'])
        drawings_count = len(group[group['PredictedTaskType'] == 'Drawings'])
        
        modeling_percent = (modeling_count / total_tasks) * 100
        drawings_percent = (drawings_count / total_tasks) * 100
        
        if abs(modeling_percent - drawings_percent) <= 10:
            main_task = 'Modeling/Drawings'
        elif modeling_percent > drawings_percent:
            main_task = 'Modeling'
        else:
            main_task = 'Drawings'

        results.append({
            'Time Period': f"{start_time.strftime('%Y-%m-%d %H:%M')} - {end_time.strftime('%Y-%m-%d %H:%M')}",
            'Main Task': main_task,
            'Modeling %': round(modeling_percent, 2),
            'Drawings %': round(drawings_percent, 2),
            'User': employee
        })

    result_df = pd.DataFrame(results)
    result_df.to_excel(output_excel, index=False)


input_csv = 'data_csv.csv'
output_excel = 'task_analysis-keras.xlsx'
start_time = datetime(2025, 1, 1, 22, 1)
end_time = datetime(2025, 1, 24, 22, 5)


model, scaler, label_encoder = train_task_classifier(input_csv)


analyze_employee_tasks(input_csv, output_excel, start_time, end_time, model, scaler, label_encoder)
