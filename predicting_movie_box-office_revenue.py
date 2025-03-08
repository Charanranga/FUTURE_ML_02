import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Function to preprocess data
def preprocess_data(df):
    """Cleans and prepares the dataset for training."""

    # Select relevant columns and remove missing values
    df = df[['genres', 'budget', 'revenue']].dropna()

    # Ensure revenue is greater than zero to avoid log errors
    df = df[df['revenue'] > 0]

    # Convert budget and revenue to log scale for normalization
    df['log_budget'] = np.log1p(df['budget'])
    df['log_revenue'] = np.log1p(df['revenue'])

    # Encode genres using target mean encoding
    genre_means = df.groupby('genres')['log_revenue'].mean()
    df['genre_encoded'] = df['genres'].map(genre_means)

    # Drop unnecessary columns
    df.drop(columns=['genres', 'budget', 'revenue'], inplace=True)

    return df, genre_means

# Function to train and evaluate the model
def train_model(df):
    """Trains a linear regression model and evaluates performance."""

    df, genre_means = preprocess_data(df)

    # Split data into features (X) and target variable (y)
    X = df.drop(columns=['log_revenue'])
    y = df['log_revenue']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save model & genre encodings
    joblib.dump(model, 'box_office_model.pkl')
    joblib.dump(genre_means, 'genre_means.pkl')

    # Predictions
    y_pred = model.predict(X_test)

    # Model Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Feature importance
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
    feature_importance = feature_importance.sort_values(by='Coefficient', ascending=False)

    # Performance Report
    report = f"""
    🎬 Movie Box Office Revenue Prediction Model 📊
    ------------------------------------------------
    Model Performance:
    - Mean Absolute Error (MAE): {mae:.2f}
    - Root Mean Squared Error (RMSE): {rmse:.2f}
    - R-Squared (R2): {r2:.2f}

    🔥 Key Influencing Factors:
    {feature_importance.to_string(index=False)}
    """
    print(report)

    # Plot feature importance
    plt.figure(figsize=(10, 5))
    sns.barplot(x=feature_importance['Coefficient'], y=feature_importance['Feature'])
    plt.title('Feature Importance in Predicting Box Office Revenue')
    plt.show()

    return model

# Function to predict revenue for a new movie
def predict_revenue(genre, budget):
    """Predicts the box office revenue of a new movie."""

    # Load the trained model & genre encodings
    model = joblib.load('box_office_model.pkl')
    genre_means = joblib.load('genre_means.pkl')

    # Encode genre
    genre_encoded = genre_means.get(genre, genre_means.mean())  # Default to mean if genre is unknown

    # Convert budget to log scale
    log_budget = np.log1p(budget)

    # Prepare input data
    input_data = pd.DataFrame({'log_budget': [log_budget], 'genre_encoded': [genre_encoded]})

    # Predict log revenue and convert back to normal scale
    log_revenue_pred = model.predict(input_data)[0]
    revenue_pred = np.expm1(log_revenue_pred)  # Convert log revenue back to actual revenue

    print(f"🎥 Estimated Box Office Revenue: ${revenue_pred:,.2f}")
    return revenue_pred

# Example usage
if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv('/content/TMDB_movie_dataset_cleaned.csv')  # Ensure your dataset has 'genres', 'budget', 'revenue'

    # Train the model
    train_model(df)

    # Predict revenue for a new movie
    genre = input("Enter movie genre: ")
    budget = float(input("Enter movie budget($): "))
    predict_revenue(genre, budget)