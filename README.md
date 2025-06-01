# heart_d_prediction

Heart Disease Classification – Model Training & Evaluation

This project involves training machine learning models to predict heart disease based on clinical and demographic features using a dataset (presumably from the UCI Heart Disease dataset).

The script model_training.py handles data preprocessing, visualization, model selection, and evaluation using K-Nearest Neighbors (KNN) and Random Forest Classifier.
📊 Features of the Script

    Exploratory Data Analysis (EDA):

        Shows the first few rows, dataset info, and statistical summary

        Displays a correlation heatmap for numerical features

        Visualizes the distribution of the target variable

    Data Preprocessing:

        One-hot encodes categorical features such as sex, cp, thal, etc.

        Splits data into training and test sets

        Applies standard scaling to continuous numerical features

    Model Training & Evaluation:

        Performs 10-fold cross-validation on KNN for K values from 1 to 20

        Plots accuracy scores for different K values with annotations

        Trains a Random Forest Classifier and evaluates its cross-validation accuracy

        Prints the mean accuracy for both models

🛠️ How to Run

    Make sure you have a dataset.csv file in the same directory with relevant heart disease data.

    Install dependencies (you can create a requirements.txt using pip freeze if needed):

pip install pandas numpy matplotlib seaborn scikit-learn

Run the script:

    python model_training.py

📈 Example Output

    Correlation heatmap

    Target distribution bar chart

    KNN cross-validation accuracy plot

    Mean cross-validation accuracy scores for:

        KNN (best K = 12)

        Random Forest

📁 Dataset Assumption

The dataset is assumed to have features like:

    Numerical: age, trestbps, chol, thalach, oldpeak

    Categorical: sex, cp, fbs, restecg, exang, slope, ca, thal

    Target: target (binary classification)
