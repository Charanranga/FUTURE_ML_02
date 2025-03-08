# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

def preprocess_data(df):
    # Drop duplicate rows if any
    df.drop_duplicates(inplace=True)

    # Fill missing values with default values
    df['title'].fillna('Unknown', inplace=True)
    df['release_date'].fillna('1900-01-01', inplace=True)
    df['overview'].fillna('No overview available', inplace=True)
    df['tagline'].fillna('No tagline', inplace=True)
    df['genres'].fillna('[]', inplace=True)
    df['production_companies'].fillna('[]', inplace=True)
    df['production_countries'].fillna('[]', inplace=True)
    df['spoken_languages'].fillna('[]', inplace=True)
    df['keywords'].fillna('[]', inplace=True)
    df['homepage'].fillna('No homepage', inplace=True)
    df['imdb_id'].fillna('No ID', inplace=True)
    df['backdrop_path'].fillna('No image', inplace=True)
    df['poster_path'].fillna('No image', inplace=True)

    # Convert release_date to datetime format
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

    # Convert categorical columns to category type
    categorical_cols = ['status', 'original_language', 'adult']
    for col in categorical_cols:
        df[col] = df[col].astype('category')

    # Drop columns with excessive missing values (more than 70%)
    threshold = 0.3  # Keep columns with at least 30% non-null values
    df = df.dropna(thresh=int(threshold * len(df)), axis=1)

    return df

# Load the dataset
df = pd.read_csv("/content/drive/MyDrive/TMDB_movie_dataset_v11.csv")
print("🔹 Original Dataset Preview:")
display(df.head())

# Get dataset information before preprocessing
print("\n🔹 Original Dataset Info:")
df.info()
print("\n🔹 Missing Values Count:")
print(df.isnull().sum())
# Get summary statistics of numerical columns
print("\n🔹 Summary Statistics:")
display(df.describe())

# Check for duplicate rows
print("\n🔹 Duplicate Rows Count:", df.duplicated().sum())

# Display column names
print("\n🔹 Column Names:")
print(df.columns)
# Preprocess the dataset
df = preprocess_data(df)

# Save cleaned data
df.to_csv('/content/TMDB_movie_dataset_cleaned.csv', index=False)
print("✅ Preprocessed dataset saved as 'TMDB_movie_dataset_cleaned.csv'")

# Display dataset information after preprocessing
print("\n🔹 Processed Dataset Info:")
# Display processed dataset
print("\n🔹 Processed Data Preview:")
display(df.head())
df.info()

# Check correlation between numerical features
numerical_df = df.select_dtypes(include=np.number)
plt.figure(figsize=(12, 6))
sns.heatmap(numerical_df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Matrix")
plt.show()



print("✅ Data Preprocessing Completed!")