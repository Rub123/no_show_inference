from sklearn.preprocessing import LabelEncoder
import pandas as pd


def preprocess(df):
    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay']).dt.date.astype('datetime64[ns]')
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay']).dt.date.astype('datetime64[ns]')
    df['Scholarship'] = df['Scholarship'].astype('object')
    df['Hipertension'] = df['Hipertension'].astype('object')
    df['Diabetes'] = df['Diabetes'].astype('object')
    df['Alcoholism'] = df['Alcoholism'].astype('object')
    df['Handcap'] = df['Handcap'].astype('object')
    df['SMS_received'] = df['SMS_received'].astype('object')
    df['ScheduledDay_DOW'] = df['ScheduledDay'].dt.day
    df['AppointmentDay_DOW'] = df['AppointmentDay'].dt.day
    df['ScheduledDay_Month'] = df['ScheduledDay'].dt.month
    df['AppointmentDay_Month'] = df['AppointmentDay'].dt.month
    df['ScheduledDay_Year'] = df['ScheduledDay'].dt.year
    df['AppointmentDay_Year'] = df['AppointmentDay'].dt.year
    x = df['Num_App_Missed']
    df.drop('Num_App_Missed', axis=1, inplace=True )
    df.drop(columns=['PatientId', 'ScheduledDay', 'AppointmentDay', 'AppointmentID'], axis=1, inplace=True)
    for col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    df['Num_App_Missed'] = x
    return df
