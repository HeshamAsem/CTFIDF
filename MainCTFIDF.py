import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, cohen_kappa_score
from sklearn.feature_extraction.text import TfidfVectorizer
import operator
from nltk.corpus import stopwords
import math
from collections import Counter

# Make sure 'CommentsDict' is defined or passed if it's an external mapping
# Example placeholder for CommentsDict (replace with your actual dictionary)
def CommentsDict(key):
    """
    Placeholder for a function that maps 'Unnamed: 0' values to text comments.
    Replace this with your actual implementation.
    """
    return f"This is a comment for key {key}"


def MainCTFIDF(Data: pd.DataFrame, N: int, Name: str, Dict: dict):
    """
    Performs K-Means clustering and then extracts keywords using TF-IDF (CTF-IDF).

    Args:
        Data (pd.DataFrame): The input DataFrame containing numerical features
                             and a 'Text' column (or 'Unnamed: 0' to be mapped to 'Text').
                             It should also contain a 'Type' column for evaluation.
        N (int): The number of clusters for K-Means.
        Name (str): A name for the dataset, used in print statements.
        Dict (dict): A dictionary containing parameters for TfidfVectorizer:
                     'ngram_range', 'max_df', 'min_df', 'norm'.
    """
    # Drop rows with any missing values
    Data.dropna(inplace=True)

    # Map 'Unnamed: 0' to 'Text' using the CommentsDict function
    # Ensure CommentsDict is defined and accessible in your environment
    Data['Text'] = Data['Unnamed: 0'].map(CommentsDict)

    # Prepare features (X) and target (y)
    # Drop columns not needed for clustering
    X = Data.drop(['Unnamed: 0', 'Type', 'Text'], axis=1)
    y = Data['Type']

    # Normalize numerical features using Z-score standardization
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X - mean) / std

    print(f'For File {Name}, has {X.shape[0]} rows, and {X.shape[1]} Features\n..........................\n')

    # Initialize dictionary to store prediction results
    PredictedDict = {}

    # Convert DataFrame and Series to NumPy arrays for K-Means
    X_array = np.array(X)
    y_array = np.array(y)

    # Convert 'Good'/'Bad' labels to numerical (1/0) if present for Cohen's Kappa
    if 'Good' in y_array:
        CohY = [1 if i == 'Good' else 0 for i in y_array]
    else:
        CohY = y_array # Assume y is already numerical if 'Good' is not present

    # Initialize and fit K-Means model
    kmeans = KMeans(n_clusters=N, n_init=50, random_state=42) # Added random_state for reproducibility
    kmeans.fit(X_array)

    # Get cluster labels from K-Means
    labels_kmeans = kmeans.labels_

    print('\n\n--------------------------------------------------------\n--------------------- FOR KMEANS---------------------- \n--------------------------------------------------------\n\n')
    print('Predicted Values (Cluster Distribution):')
    print(pd.Series(labels_kmeans).value_counts())
    print('..............')

    # Add predicted labels to the original DataFrame
    Data['Predicted'] = labels_kmeans

    # Store K-Means results in PredictedDict
    PredictedDict['Kmeans'] = dict(pd.Series(labels_kmeans).value_counts())

    # Calculate and store Silhouette Score
    SilScore = silhouette_score(X_array, labels_kmeans)
    PredictedDict['Kmeans']['SScore'] = SilScore

    # Calculate and store Cohen's Kappa Score
    CohScore = cohen_kappa_score(CohY, labels_kmeans)
    PredictedDict['Kmeans']['CScore'] = CohScore

    # --- CTF-IDF Calculation ---
    AllText = []
    # Concatenate all text from each cluster into a single string
    for cluster_id in Data['Predicted'].unique():
        cluster_text = ' '.join([str(text) for text in Data[Data['Predicted'] == cluster_id]['Text'].tolist()])
        AllText.append(cluster_text)

    # Create a DataFrame for TF-IDF vectorization
    ThisDF = pd.DataFrame()
    ThisDF['Text'] = AllText
    ThisDF['Cluster'] = Data['Predicted'].unique()

    print("\nDataFrame for TF-IDF Vectorization:")
    print(ThisDF.head())

    # Initialize TfidfVectorizer with parameters from the input dictionary
    VecModel = TfidfVectorizer(ngram_range=Dict['ngram_range'], max_df=Dict['max_df'],
                               min_df=Dict['min_df'], norm=Dict['norm'])

    # Fit and transform the text data to get TF-IDF features
    X_Vec = VecModel.fit_transform(ThisDF['Text'])
    X_Vec = pd.DataFrame.sparse.from_spmatrix(X_Vec)

    # Create a mapping from feature index to word
    Names = {x: y for y, x in VecModel.vocabulary_.items()}

    # Iterate through each cluster to find and print top words
    for e, cluster_id in zip(range(len(Data['Predicted'].unique())), Data['Predicted'].unique()):
        print(f'\nMost Repeated Words in Cluster {cluster_id} (Simple Word Count):')
        # Get all text for the current cluster, join it, split into words,
        # count frequencies, and filter out stopwords
        cluster_words = ' '.join([str(j) for j in Data[Data['Predicted'] == cluster_id]['Text'].tolist()]).split()
        most_common_words = pd.Series(cluster_words).value_counts().keys().tolist()
        filtered_words = [k for k in most_common_words if k.lower() not in stopwords.words('english')][:20]
        print(filtered_words)

        # Extract TF-IDF values for the current cluster
        ResultDict = {m: [n, Names[m]] for m, n in enumerate(X_Vec.loc[e].tolist()) if n != 0}

        # Convert to a word:TF-IDF score dictionary and sort by score
        Words = {k: v for v, k in ResultDict.values()}
        WordSortes = dict(sorted(Words.items(), key=operator.itemgetter(1), reverse=True))

        # Get top 100 words based on TF-IDF, filter stopwords, and round values
        TopWordsValues = {k: np.round(v, 4) for k, v in dict(list(WordSortes.items())[:100]).items()
                          if k.lower() not in stopwords.words('english')}
        TopWords = [k for k, v in TopWordsValues.items()]

        print(f'TopWords using CTFIDF:\n{TopWords}')
        print(f'TopWordsValues using CTFIDF:\n{TopWordsValues}')
        print('........................................')

    return PredictedDict # Return the dictionary containing K-Means evaluation metrics