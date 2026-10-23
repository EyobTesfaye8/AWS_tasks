# AWS ML Tasks - Week 01: Market Arrivals Data Analysis

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EyobTesfaye8/AWS_tasks/blob/main/week-01/AWS_First_week_task.ipynb)
![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

This repository contains the end-to-end data pipeline, statistical analysis, and exploratory data analysis for agricultural commodity market arrivals as part of the AWS Machine Learning program.

---

## 📁 Repository Structure

All work for Week 01 is structured inside the designated `week-01/` directory:

```text
AWS_ML/
├── .gitignore                      # Configured to ignore virtual environments, cache, checkpoints
├── requirements.txt                # Project dependencies (lists numpy, pandas, matplotlib, etc.)
├── README.md                       # Documentation, setup guide, and instructions
└── week-01/
    ├── AWS_First_week_task.ipynb   # Jupyter Notebook with full executed outputs & visualizations
    ├── main.py                     # Python script with modular named cleaning & NumPy functions
    ├── marketing.csv               # Raw dataset stored in the repository
    └── price_trends.png            # Generated time-series visualization
```

---

## 🚀 Getting Started

Follow the instructions below to set up a clean Python virtual environment and run the code.

### 1. Clone the Repository

```bash
git clone https://github.com/EyobTesfaye8/AWS_tasks.git
cd AWS_tasks
```

### 2. Create and Activate a Virtual Environment (`venv`)

Creating an isolated virtual environment ensures clean dependency management:

* **On Linux / macOS:**
  ```bash
  # Create virtual environment named '.venv'
  python3 -m venv .venv

  # Activate virtual environment
  source .venv/bin/activate
  ```

* **On Windows (Command Prompt / PowerShell):**
  ```powershell
  # Create virtual environment named '.venv'
  python -m venv .venv

  # Activate virtual environment
  .\.venv\Scripts\activate
  ```

> **Note:** The `.venv/` directory is registered in `.gitignore` and is never committed to GitHub.

### 3. Install Dependencies

Install the required packages using `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Key dependencies included:
* **`numpy`**: Fast vectorized array calculations and statistical operations.
* **`pandas`**: Tabular data manipulation and data preprocessing.
* **`matplotlib`**: Time-series visualization and plotting.
* **`requests`**: Safe remote data download fallbacks.
* **`jupyter`** / **`ipykernel`**: Notebook execution environment.

---

## 💻 How to Run

### Option A: Run the Python Pipeline Script

Execute the self-contained pipeline script from the root repository directory:

```bash
python week-01/main.py
```

This will:
1. Load the raw dataset from `week-01/marketing.csv`.
2. Execute each named cleaning function in sequence.
3. Perform NumPy statistical calculations and print formatted outputs.
4. Generate and save the price trends plot to `week-01/price_trends.png`.

---

### Option B: Run the Jupyter Notebook Locally

Launch the Jupyter Notebook server:

```bash
jupyter notebook week-01/AWS_First_week_task.ipynb
```

You can step through the notebook or select **Kernel -> Restart & Run All** to re-run all cells. All outputs and plots are pre-computed and saved in the committed notebook.

---

### Option C: Open Directly in Google Colab

You can run the notebook directly in your browser with zero local installation by clicking the badge below:

👉 **[Open in Google Colab](https://colab.research.google.com/github/EyobTesfaye8/AWS_tasks/blob/main/week-01/AWS_First_week_task.ipynb)**

---

## 🛠️ Data Pipeline & Function Highlights

The codebase implements modular, named functions (`def`) adhering to software engineering best practices:

### 1. Named Data Cleaning Functions

* **`load_raw_data(filepath, fallback_url)`**: Safely loads the dataset from the repository with automated network fallback if missing.
* **`clean_missing_and_duplicates(df)`**: Inspects null values across columns and drops duplicate rows (`df.drop_duplicates()`).
* **`clean_text_fields(df, text_columns)`**: Trims leading/trailing whitespace (`str.strip()`) and standardizes text casing (`str.upper()`) across categorical columns (`market`, `city`, `state`).
* **`convert_column_types(df)`**: Converts dates to `datetime64[ns]` and casts numerical attributes (`quantity`, `priceMin`, `priceMax`, `priceMod`) into explicit numeric types.
* **`detect_and_handle_outliers(df, column, factor=1.5)`**: Identifies outliers using the Interquartile Range (IQR) method and applies Winsorization (capping) at statistical boundaries:
  $$\text{IQR} = Q_3 - Q_1$$
  $$\text{Lower Bound} = \max(0, Q_1 - 1.5 \times \text{IQR}), \quad \text{Upper Bound} = Q_3 + 1.5 \times \text{IQR}$$

### 2. NumPy Statistical Calculations

We demonstrate multiple NumPy calculation functions:
1. **`np.mean()`**: Computes the central average modal price.
2. **`np.median()`**: Robust 50th-percentile calculation unaffected by skewness.
3. **`np.std()`**: Measures price dispersion and market volatility.
4. **`np.percentile()`**: Calculates key statistical quantiles (10th, 25th, 75th, 90th).
5. **`np.where()`**: High-performance vectorized conditional tagging for market price segmentation (classifying prices into *Premium* vs *Standard*).
6. **`np.min()` & `np.max()`**: Pinpoints boundary market prices.

---

## 📊 Summary of Results

| Metric | NumPy Value |
| :--- | :--- |
| **Record Count** | 10,227 records |
| **Average Price (`np.mean`)** | 984.28 INR / quintal |
| **Median Price (`np.median`)** | 747.00 INR / quintal |
| **Price Volatility (`np.std`)** | 818.43 INR / quintal |
| **25th Percentile (`np.percentile`)** | 448.00 INR / quintal |
| **75th Percentile (`np.percentile`)** | 1,248.00 INR / quintal |
| **Price Range (`np.min` to `np.max`)** | 80.00 to 6,400.00 INR / quintal |
