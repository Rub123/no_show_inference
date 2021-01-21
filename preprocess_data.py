from sklearn.preprocessing import OneHotEncoder
import pandas as pd

f = ['Gender', 'ScheduledDay', 'AppointmentDay', 'Age', 'Neighbourhood', 'Scholarship', 'Hipertension', 'Diabetes',
     'Alcoholism', 'Handcap', 'SMS_received']

end_f = ['Gender', 'Age', 'Scholarship', 'Hipertension', 'Diabetes',
         'Alcoholism', 'Handcap', 'SMS_received', 'Neighbourhood_AEROPORTO',
         'Neighbourhood_ANDORINHAS', 'Neighbourhood_ANTÔNIO HONÓRIO',
         'Neighbourhood_ARIOVALDO FAVALESSA', 'Neighbourhood_BARRO VERMELHO',
         'Neighbourhood_BELA VISTA', 'Neighbourhood_BENTO FERREIRA',
         'Neighbourhood_BOA VISTA', 'Neighbourhood_BONFIM',
         'Neighbourhood_CARATOÍRA', 'Neighbourhood_CENTRO',
         'Neighbourhood_COMDUSA', 'Neighbourhood_CONQUISTA',
         'Neighbourhood_CONSOLAÇÃO', 'Neighbourhood_CRUZAMENTO',
         'Neighbourhood_DA PENHA', 'Neighbourhood_DE LOURDES',
         'Neighbourhood_DO CABRAL', 'Neighbourhood_DO MOSCOSO',
         'Neighbourhood_DO QUADRO', 'Neighbourhood_ENSEADA DO SUÁ',
         'Neighbourhood_ESTRELINHA', 'Neighbourhood_FONTE GRANDE',
         'Neighbourhood_FORTE SÃO JOÃO', 'Neighbourhood_FRADINHOS',
         'Neighbourhood_GOIABEIRAS', 'Neighbourhood_GRANDE VITÓRIA',
         'Neighbourhood_GURIGICA', 'Neighbourhood_HORTO',
         'Neighbourhood_ILHA DAS CAIEIRAS', 'Neighbourhood_ILHA DE SANTA MARIA',
         'Neighbourhood_ILHA DO BOI', 'Neighbourhood_ILHA DO FRADE',
         'Neighbourhood_ILHA DO PRÍNCIPE',
         'Neighbourhood_ILHAS OCEÂNICAS DE TRINDADE', 'Neighbourhood_INHANGUETÁ',
         'Neighbourhood_ITARARÉ', 'Neighbourhood_JABOUR',
         'Neighbourhood_JARDIM CAMBURI', 'Neighbourhood_JARDIM DA PENHA',
         'Neighbourhood_JESUS DE NAZARETH', 'Neighbourhood_JOANA D´ARC',
         'Neighbourhood_JUCUTUQUARA', 'Neighbourhood_MARIA ORTIZ',
         'Neighbourhood_MARUÍPE', 'Neighbourhood_MATA DA PRAIA',
         'Neighbourhood_MONTE BELO', 'Neighbourhood_MORADA DE CAMBURI',
         'Neighbourhood_MÁRIO CYPRESTE', 'Neighbourhood_NAZARETH',
         'Neighbourhood_NOVA PALESTINA', 'Neighbourhood_PARQUE INDUSTRIAL',
         'Neighbourhood_PARQUE MOSCOSO', 'Neighbourhood_PIEDADE',
         'Neighbourhood_PONTAL DE CAMBURI', 'Neighbourhood_PRAIA DO CANTO',
         'Neighbourhood_PRAIA DO SUÁ', 'Neighbourhood_REDENÇÃO',
         'Neighbourhood_REPÚBLICA', 'Neighbourhood_RESISTÊNCIA',
         'Neighbourhood_ROMÃO', 'Neighbourhood_SANTA CECÍLIA',
         'Neighbourhood_SANTA CLARA', 'Neighbourhood_SANTA HELENA',
         'Neighbourhood_SANTA LUÍZA', 'Neighbourhood_SANTA LÚCIA',
         'Neighbourhood_SANTA MARTHA', 'Neighbourhood_SANTA TEREZA',
         'Neighbourhood_SANTO ANDRÉ', 'Neighbourhood_SANTO ANTÔNIO',
         'Neighbourhood_SANTOS DUMONT', 'Neighbourhood_SANTOS REIS',
         'Neighbourhood_SEGURANÇA DO LAR', 'Neighbourhood_SOLON BORGES',
         'Neighbourhood_SÃO BENEDITO', 'Neighbourhood_SÃO CRISTÓVÃO',
         'Neighbourhood_SÃO JOSÉ', 'Neighbourhood_SÃO PEDRO',
         'Neighbourhood_TABUAZEIRO', 'Neighbourhood_UNIVERSITÁRIO',
         'Neighbourhood_VILA RUBIM', 'schedule_day', 'schedule_month',
         'schedule_year', 'schedule_hour', 'appointment_day',
         'appointment_month', 'appointment_year', 'appointment_hour',
         'appointment_dayofweek', 'day_diff_schedule_appointment']


def preprocess(df, columns=f):
    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay']).dt.date.astype('datetime64[ns]')
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay']).dt.date.astype('datetime64[ns]')
    df['Gender'] = df.Gender.map({'M': 0, 'F': 1})
    X = df[columns]
    categorical = X.describe(include='O').columns

    ohe = OneHotEncoder(sparse=False)
    ohe.fit(X[categorical])

    add_to_X = pd.DataFrame(ohe.transform(X[categorical]), columns=ohe.get_feature_names(categorical), index=X.index)
    X.drop('Neighbourhood', axis=1, inplace=True)

    X = pd.concat([X, add_to_X], axis=1)
    X['schedule_day'] = X.ScheduledDay.dt.day
    X['schedule_month'] = X.ScheduledDay.dt.month
    X['schedule_year'] = X.ScheduledDay.dt.year
    X['schedule_hour'] = X.ScheduledDay.dt.hour

    X['appointment_day'] = X.AppointmentDay.dt.day
    X['appointment_month'] = X.AppointmentDay.dt.month
    X['appointment_year'] = X.AppointmentDay.dt.year
    X['appointment_hour'] = X.AppointmentDay.dt.hour
    X['appointment_dayofweek'] = X.AppointmentDay.dt.dayofweek

    X['day_diff_schedule_appointment'] = (X['AppointmentDay'] - X['ScheduledDay']).dt.days
    X.drop('ScheduledDay', axis=1, inplace=True)
    X.drop('AppointmentDay', axis=1, inplace=True)

    return pd.DataFrame(data=X, columns=end_f).fillna(0)
