# FUTURE_ML_02

# Predicting Movie Box Office Revenue

## 📌 Project Overview
This project focuses on predicting the box office revenue of movies using **Machine Learning** techniques. By analyzing factors such as genre, cast, production budget, and audience reception, we develop a regression model to estimate a movie's financial performance.

## 📂 Dataset
We use the **TMDB Movie Dataset** sourced from Kaggle:  
🔗 [TMDB Movies Dataset](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies)

The dataset contains:
- **Movie Metadata**: Title, genre, release date, runtime, etc.
- **Financial Information**: Budget, revenue, popularity.
- **Cast & Crew Details**: Actors, directors, producers.
- **Audience Reception**: Vote counts and ratings.

## 🛠 Tools & Technologies
- **Programming**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Machine Learning Techniques**: Regression modeling
- **Platforms**: Jupyter Notebook / Google Colab

## 📊 Methodology
1. **Data Preprocessing**
   - Handling missing values
   - Feature engineering (transforming categorical variables, creating new features)
   - Normalization and encoding

2. **Exploratory Data Analysis (EDA)**
   - Identifying trends and correlations
   - Visualizing key relationships

3. **Model Training & Evaluation**
   - Models tested: Linear Regression, Random Forest, XGBoost
   - Performance metrics: RMSE, R² Score

4. **Predictions & Insights**
   - Analyzing key factors influencing box office performance
   - Business implications and improvements

## 🔥 Results
- The **Random Forest Regressor** achieved the best performance with an **R² score of 0.51**.
- **Mean Absolute Error (MAE): 2.80**
- **Root Mean Squared Error (RMSE): 3.64**
-  🔥 Key Influencing Factors : Feature  Coefficient
-   genre_encoded  : 0.802924
-   log_budget  : 0.241214

  
- The most significant factors affecting revenue were **budget, popularity, and cast**.
- The model can assist movie studios in estimating a film’s potential earnings before release.

## 📌 How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/movie-revenue-prediction.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook Movie_Revenue_Prediction.ipynb
   ```

## 📎 Contributions
We welcome contributions! Feel free to fork, improve, or enhance the model.

## Contact
For any questions, reach out via GitHub Issues or email:[gorantlacharanranga@gmail.com]

🚀 Happy Coding! 🎬



