import pandas as pd
import numpy as np

def load_and_clean_data(path):
    df = pd.read_csv(path)

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Drop columns with too much noise
    drop_cols = ['name', 'description', 'trim', 'exterior_color', 'interior_color']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Handle missing values
    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include='object').columns

    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)

    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    return df
