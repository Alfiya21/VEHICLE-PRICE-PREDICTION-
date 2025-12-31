from datetime import datetime

def create_features(df):
    current_year = datetime.now().year

    if 'year' in df.columns:
        df['vehicle_age'] = current_year - df['year']

    if 'mileage' in df.columns:
        df['mileage_per_year'] = df['mileage'] / (df['vehicle_age'] + 1)

    df.drop(columns=['year'], inplace=True)
    return df
