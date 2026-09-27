# Heart Disease Detection using Deep Neural Networks

A deep learning classification project that predicts the presence of heart disease in patients based on clinical and demographic health indicators. Built with **TensorFlow (Keras)** and **Scikit-Learn**, the model uses a Sequential Deep Neural Network enhanced with Dropout layers and regularization to prevent overfitting on tabular medical data.

---

## Dataset Overview (`heart.csv`)

The dataset consists of clinical records across 13 predictor features and 1 target variable:

| Feature | Description |
| :--- | :--- |
| `age` | Age of the patient in years |
| `sex` | Gender (`1` = male; `0` = female) |
| `cp` | Chest pain type (4 values: `0` to `3`) |
| `trestbps` | Resting blood pressure (in mm Hg on admission to the hospital) |
| `chol` | Serum cholesterol in mg/dl |
| `fbs` | Fasting blood sugar > 120 mg/dl (`1` = true; `0` = false) |
| `restecg` | Resting electrocardiographic results (values `0`, `1`, `2`) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (`1` = yes; `0` = no) |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | The slope of the peak exercise ST segment |
| `ca` | Number of major vessels (`0`–`3`) colored by fluoroscopy |
| `thal` | Thalassemia indicator (`1` = normal; `2` = fixed defect; `3` = reversible defect) |
| **`target`** | **Diagnosis of heart disease (`1` = presence; `0` = absence)** |

---

## Methodology & Architecture

1. **Exploratory Data Analysis & Preprocessing:**
   * Partitioned the dataset into training and testing sets using `train_test_split`.
   * Standardized numerical features using Scikit-Learn's `StandardScaler` to ensure uniform gradient descent convergence.
   * Converted target labels for neural network compatibility using `to_categorical`.

2. **Neural Network Architecture (TensorFlow / Keras):**
   * **Model Type:** Keras `Sequential` API
   * **Hidden Layers:** Fully connected `Dense` layers with non-linear activations
   * **Overfitting Prevention:** Integrated `Dropout` layers and kernel `regularizers` (L1/L2)
   * **Optimizer:** Compiled with the `Adam` optimizer

3. **Model Evaluation:**
   * Evaluated performance on unseen test data using precision, recall, and F1-score (`classification_report`).
   * Visualized true vs. false positives/negatives using a `confusion_matrix` heatmap via Seaborn and Matplotlib.

---

## Tech Stack

* **Deep Learning:** TensorFlow, Keras (`Sequential`, `Dense`, `Dropout`, `Adam`, `regularizers`)
* **Machine Learning & Preprocessing:** Scikit-Learn (`StandardScaler`, `train_test_split`, `classification_report`, `confusion_matrix`)
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Environment:** Jupyter Notebook, Python

---

## Repository Structure

```text
├── .gitignore                 # Excludes virtual environment (heart_venv/)
├── README.md                  # Project documentation
├── heart.csv                  # Clinical heart disease dataset
├── heart_disease_NN.ipynb     # Main Jupyter Notebook (EDA, preprocessing, NN training & evaluation)
└── requirements.txt           # Required Python libraries
```

---

## Local Setup & Execution

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/TONY00009/Neural_Network_Heart_Disease.git](https://github.com/TONY00009/Neural_Network_Heart_Disease.git)
   cd Neural_Network_Heart_Disease
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv heart_venv
   heart_venv\Scripts\activate      # Windows
   # source heart_venv/bin/activate # Linux/macOS
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the notebook:**
   Open `heart_disease_NN.ipynb` in VS Code or Jupyter Notebook, select the `heart_venv` kernel, and run all cells.
