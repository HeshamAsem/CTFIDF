---

# CTF-IDF and Modified CTF-IDF for Cluster Keyword Extraction

This repository provides Python functions designed to perform **K-Means clustering** on tabular data and then extract the most important keywords from each generated cluster using **TF-IDF (Term Frequency-Inverse Document Frequency)**, along with two **modified versions of CTF-IDF (Class-based Term Frequency-Inverse Document Frequency)**.

This approach is particularly useful in scenarios where you have numerical data that represents entities (e.g., user profiles, product features) and associated text (e.g., comments, descriptions). By clustering based on numerical features, you can then analyze the textual content of each cluster to understand its defining characteristics.

---

## Features

* **K-Means Clustering:** Groups data points based on their numerical features.
* **Standard CTF-IDF:** Identifies relevant terms within each cluster using the classic TF-IDF formulation applied to the concatenated text of a cluster (treated as a "document").
* **Modified CTF-IDF (Version 1):** Introduces a custom adjustment to the IDF component, scaling it based on how frequently a term appears across all original documents (not just within clusters).
* **Modified CTF-IDF (Version 2):** Implements another smoothed IDF formula to enhance robustness and prevent extreme values.
* **Evaluation Metrics:** Calculates Silhouette Score and Cohen's Kappa Score for K-Means clustering.
* **Keyword Visualization:** Prints top keywords for each cluster using both simple word counts and the CTF-IDF methodologies.

---

## Installation

To get started, clone this repository and install the required Python libraries using `pip`:

```bash
git clone [https://github.com/HeshamAsem/CTFIDF.git](https://github.com/HeshamAsem/CTFIDF.git)
cd CTFIDF
pip install pandas numpy scikit-learn nltk
```

### NLTK Stopwords

The first time you run the code, you might need to download the NLTK stopwords:

```python
import nltk
nltk.download('stopwords')
```

---

## Usage

To use these functions, you will need a Pandas DataFrame with numerical features for clustering and a text column for keyword extraction.

**Important:** The functions assume the existence of a `CommentsDict` function or a similar mapping mechanism that converts values from `Unnamed: 0` column to the actual text content. **You must define `CommentsDict` or adjust the code to suit your data loading and text mapping.**

### Example Data Structure

Your input `pd.DataFrame` (`Data`) should ideally contain:

* **Numerical Columns:** Any number of columns with numerical data that K-Means will use for clustering.
* **`'Unnamed: 0'` (or similar ID column):** A column that can be mapped to your text content using `CommentsDict`.
* **`'Type'` (Optional):** A column with ground truth labels (e.g., 'Good', 'Bad') if you wish to evaluate K-Means with Cohen's Kappa Score.
* **`'Text'` (Generated Internally):** This column will be created by the functions based on `Unnamed: 0` and `CommentsDict`.

### How to Run

1.  **Define `CommentsDict`:**
    Make sure your `CommentsDict` (or equivalent) is correctly implemented to map the `Unnamed: 0` column's values to the relevant text.

    ```python
    # Example placeholder for CommentsDict (replace with your actual dictionary/function)
    def CommentsDict(key):
        """
        This function should map a key (from Data['Unnamed: 0']) to its corresponding text.
        Example: If 'Unnamed: 0' contains IDs and you have a dictionary mapping IDs to comments:
        return your_comments_dictionary.get(key, "")
        """
        # For demonstration purposes, returning a dummy string
        return f"This is a sample comment for ID {key}"
    ```

2.  **Prepare your Data:**
    Load your data into a Pandas DataFrame.

    ```python
    import pandas as pd
    import numpy as np

    # Create dummy data for demonstration
    # In a real scenario, you'd load your data from a CSV, database, etc.
    data = {
        'Unnamed: 0': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Feature1': np.random.rand(10),
        'Feature2': np.random.rand(10),
        'Type': ['Good', 'Bad', 'Good', 'Good', 'Bad', 'Good', 'Bad', 'Good', 'Bad', 'Good']
    }
    # For demonstration, let's create a simple mapping for CommentsDict
    global_comments_map = {
        1: "excellent product quality and delivery very fast",
        2: "poor customer service and slow response time",
        3: "great value for money highly recommend",
        4: "amazing features and user friendly interface",
        5: "issues with installation contacted support no help",
        6: "satisfied with purchase works as expected",
        7: "buggy software constant crashes",
        8: "love the design and durability",
        9: "received damaged item terrible packaging",
        10: "happy with performance and reliability"
    }

    def CommentsDict(key):
        return global_comments_map.get(key, "")

    df = pd.DataFrame(data)
    ```

3.  **Call the Functions:**
    Now you can call the functions with your data.

    ```python
    # Assuming the functions MainCTFIDF, ModifiedCTFIDF1, ModifiedCTFIDF2 are imported or defined in your script

    # Define TF-IDF parameters for MainCTFIDF
    tfidf_params = {
        'ngram_range': (1, 1), # Consider single words
        'max_df': 1.0,         # Ignore terms that appear in more than X% of documents
        'min_df': 1,           # Ignore terms that appear in less than X documents
        'norm': 'l2'           # L2 normalization
    }

    # Run MainCTFIDF
    print("--- Running MainCTFIDF ---")
    results_main = MainCTFIDF(df.copy(), N=2, Name="Sample Data", Dict=tfidf_params)
    print("\nMainCTFIDF Results (K-Means Metrics):", results_main)

    # Run ModifiedCTFIDF1
    print("\n--- Running ModifiedCTFIDF1 ---")
    results_mod1 = ModifiedCTFIDF1(df.copy(), N=2, Name="Sample Data")
    print("\nModifiedCTFIDF1 Results (K-Means Metrics):", results_mod1)

    # Run ModifiedCTFIDF2
    print("\n--- Running ModifiedCTFIDF2 ---")
    results_mod2 = ModifiedCTFIDF2(df.copy(), N=2, Name="Sample Data")
    print("\nModifiedCTFIDF2 Results (K-Means Metrics):", results_mod2)
    ```

---

## Function Reference

### `MainCTFIDF(Data, N, Name, Dict)`

Performs K-Means clustering and then extracts keywords using a standard TF-IDF approach (referred to as CTF-IDF in the original context, implying TF-IDF applied to document clusters).

* **`Data` (pd.DataFrame):** Your input DataFrame. Must contain numerical features, `'Unnamed: 0'` (for text mapping), and optionally `'Type'` for evaluation.
* **`N` (int):** The number of clusters `k` for K-Means.
* **`Name` (str):** A descriptive name for your dataset, used in print statements.
* **`Dict` (dict):** A dictionary specifying parameters for `TfidfVectorizer`, e.g., `{'ngram_range': (1,1), 'max_df': 1.0, 'min_df': 1, 'norm': 'l2'}`.
* **Returns:** A dictionary containing K-Means evaluation metrics (`'Kmeans'`: cluster counts, Silhouette Score, Cohen's Kappa Score).

### `ModifiedCTFIDF1(Data, N, Name)`

Implements a modified CTF-IDF approach where the IDF component is adjusted by a factor related to the document frequency of a term across all original documents.

* **`Data` (pd.DataFrame):** Your input DataFrame, similar to `MainCTFIDF`.
* **`N` (int):** The number of clusters `k` for K-Means.
* **`Name` (str):** A descriptive name for your dataset.
* **Returns:** A dictionary containing K-Means evaluation metrics (`'Kmeans'`: cluster counts, Silhouette Score, Cohen's Kappa Score).

### `ModifiedCTFIDF2(Data, N, Name)`

Applies another modified CTF-IDF, where the IDF calculation uses a smoothed formula with a constant `K` (0.3 by default) to mitigate the impact of very common or very rare terms.

* **`Data` (pd.DataFrame):** Your input DataFrame, similar to `MainCTFIDF`.
* **`N` (int):** The number of clusters `k` for K-Means.
* **`Name` (str):** A descriptive name for your dataset.
* **Returns:** A dictionary containing K-Means evaluation metrics (`'Kmeans'`: cluster counts, Silhouette Score, Cohen's Kappa Score).

---

## Contributing

Feel free to open issues or submit pull requests if you have suggestions for improvements or bug fixes.

---

## License

This project is open-sourced under the MIT License. See the `LICENSE` file for more details.


---


## Contact

For any questions or inquiries, please contact [Hesham Asem](https://github.com/HeshamAsem).

