import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Existing functions
def load_data(file_path=None, manual_data=None):
    if file_path:
        if file_path.endswith(".csv"):
            print(f"Loading CSV file: {file_path}")
            return pd.read_csv(file_path)
        elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
            print(f"Loading Excel file: {file_path}")
            return pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file type. Please use CSV or Excel.")
    elif manual_data:
        print("Loading manual data...")
        return pd.DataFrame(manual_data)
    else:
        raise ValueError("Please provide either file_path or manual_data.")

def quick_summary(df):
    print("\n Data Shape:", df.shape)
    print("\n Columns:", df.columns.tolist())
    print("\n Null Values:\n", df.isnull().sum())
    print("\n Data Types:\n", df.dtypes)

# New EDA functions
def plot_target_distribution(df, target_col):
    """
    Plot histogram of the target variable.
    """
    plt.figure(figsize=(8,4))
    sns.histplot(df[target_col], kde=True)
    plt.title(f"Distribution of {target_col}")
    plt.show()

def plot_correlation_matrix(df):
    """
    Plot correlation heatmap.
    """
    plt.figure(figsize=(12,10))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()

def plot_numerical_features(df, features):
    """
    Plot histograms for multiple numerical features.
    """
    df[features].hist(bins=20, figsize=(15,10))
    plt.tight_layout()
    plt.show()
