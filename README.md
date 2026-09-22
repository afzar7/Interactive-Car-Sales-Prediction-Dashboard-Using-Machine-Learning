Interactive Car Sales Prediction Dashboard Using Machine Learning

An interactive **Car Sales Prediction Dashboard** that uses Machine Learning to predict car sales based on historical data and provides an easy-to-use interface for exploring sales trends, analyzing important factors, and generating predictions.

## 📌 Project Overview

The **Interactive Car Sales Prediction Dashboard** is designed to help users understand car sales data and predict future sales using machine learning techniques.

The system combines:

* 📊 Data analysis and visualization
* 🤖 Machine Learning-based prediction
* 📈 Interactive dashboards
* 🔍 Exploratory Data Analysis (EDA)
* 🎯 User-input-based sales prediction

Users can enter relevant car and sales information through the dashboard and receive a predicted sales value.

---

## ✨ Features

* **Interactive Dashboard** for exploring car sales data
* **Sales Prediction** using a trained Machine Learning model
* **Data Visualization** through charts and graphs
* **Exploratory Data Analysis** of historical sales
* **Feature Analysis** to identify factors influencing sales
* **User-Friendly Interface** for entering prediction parameters
* **Model Performance Evaluation**
* **Real-Time Prediction** based on user inputs

---

## 🧠 Machine Learning

The project uses supervised machine learning to predict car sales from historical data.

### Typical Workflow

```text
Historical Car Sales Data
          ↓
    Data Cleaning
          ↓
 Exploratory Data Analysis
          ↓
 Feature Engineering
          ↓
 Train/Test Split
          ↓
 Machine Learning Model
          ↓
 Model Evaluation
          ↓
    Saved Model
          ↓
 Interactive Dashboard
          ↓
     Sales Prediction
```

Depending on the dataset, models such as the following can be used:

* Linear Regression
* Decision Tree Regression
* Random Forest Regression
* Gradient Boosting
* XGBoost

The final model can be selected based on evaluation metrics such as **MAE, MSE, RMSE, and R² Score**.

---

## 📊 Dataset

The project uses historical car sales data containing information that may include:

* Car Name / Model
* Manufacturer
* Year
* Resale Price
* Fuel Type
* Transmission
* Mileage
* Engine Capacity
* Sales-related attributes

The dataset should be cleaned before training the model.

### Data Preprocessing

Typical preprocessing steps include:

1. Handling missing values
2. Removing duplicate records
3. Detecting and handling outliers
4. Encoding categorical variables
5. Scaling numerical features when required
6. Selecting relevant features
7. Splitting the data into training and testing sets

---

## 📈 Dashboard

The dashboard can provide multiple sections for analyzing and predicting car sales.

### 1. Overview

Displays key statistics such as:

* Total number of records
* Average sales
* Average resale price
* Popular car models
* Sales trends

### 2. Sales Analysis

Interactive visualizations can show:

* Sales by year
* Sales by manufacturer
* Sales by fuel type
* Sales by transmission
* Price distribution
* Mileage distribution

### 3. Feature Analysis

Users can explore relationships between different variables and car sales.

```text
Resale Price  ──┐
Year            ──┤
Mileage         ──┤
Fuel Type       ──┼──> ML Model ──> Predicted Sales
Transmission    ──┤
Engine          ──┤
Other Features  ──┘
```

### 4. Prediction

Users enter the required car details into the dashboard.

The application processes the input and sends it to the trained model.

```text
User Input
    ↓
Preprocessing
    ↓
Trained ML Model
    ↓
Prediction
    ↓
Dashboard Result
```

---

## 🛠️ Technologies Used

| Technology       | Purpose                           |
| ---------------- | --------------------------------- |
| Python           | Programming language              |
| Pandas           | Data manipulation                 |
| NumPy            | Numerical computation             |
| Scikit-learn     | Machine Learning                  |
| Matplotlib       | Data visualization                |
| Seaborn          | Statistical visualization         |
| Streamlit        | Interactive dashboard             |
| Joblib           | Model serialization               |
| Jupyter Notebook | Data analysis and experimentation |

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Interactive-Car-Sales-Prediction-Dashboard.git
```

### 2. Navigate to the Project Directory

```bash
cd Interactive-Car-Sales-Prediction-Dashboard
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Linux/macOS**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

If the dashboard is built using Streamlit:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🧪 Model Training

To train the model:

1. Load the car sales dataset.
2. Perform data preprocessing.
3. Select relevant features.
4. Split the dataset into training and testing sets.
5. Train multiple regression models.
6. Evaluate the models.
7. Select the appropriate model based on evaluation metrics.
8. Save the trained model.

Example:

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

joblib.dump(model, "car_sales_model.pkl")
```

---

## 📏 Model Evaluation

The model can be evaluated using:

### Mean Absolute Error (MAE)

Measures the average absolute difference between actual and predicted values.

### Mean Squared Error (MSE)

Measures the average squared prediction error.

### Root Mean Squared Error (RMSE)

The square root of MSE and provides the error in the same unit as the target.

### R² Score

Measures how well the model explains the variation in the target variable.

Example:

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R² Score:", r2)
```

---

## 📦 Example `requirements.txt`

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
streamlit
joblib
jupyter
```

---

## 🎯 Use Cases

This dashboard can be useful for:

* Car dealerships
* Automobile businesses
* Sales analysts
* Marketing teams
* Business intelligence projects
* Students learning Machine Learning
* Data science projects
* Academic mini/major projects

---

## 🔮 Future Enhancements

Possible improvements include:

* Integration with live car sales data
* Advanced ensemble models
* Deep Learning-based prediction
* Price and demand forecasting
* Customer segmentation
* Recommendation system
* Model explainability using SHAP
* User authentication
* Cloud deployment
* Automated model retraining
* Downloadable prediction reports
* Mobile-friendly dashboard

---

## 📌 Limitations

* Prediction accuracy depends on the quality and size of the dataset.
* Historical trends may not always represent future market conditions.
* Predictions should be treated as estimates rather than guaranteed outcomes.
* Model performance may change when applied to a significantly different dataset.

---

## 👨‍💻 Project Objective

The main objective of this project is to demonstrate how **Machine Learning, Data Analysis, and Interactive Visualization** can be combined to create a practical car sales prediction system.

The project provides both **analytical insights** and **machine learning predictions** through an intuitive dashboard.

---


## ⭐ Acknowledgement

This project demonstrates the application of data science and machine learning concepts to the automobile sales domain.

If you find this project useful, consider giving the repository a ⭐ on GitHub.
