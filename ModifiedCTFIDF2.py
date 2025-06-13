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

def ModifiedCTFIDF2(Data: pd.DataFrame, N: int, Name: str):
    """
    Performs K-Means clustering and then extracts keywords using a modified CTF-IDF (version 2).
    This modification applies a smoothed IDF formula.

    Args:
        Data (pd.DataFrame): The input DataFrame containing numerical features
                             and a 'Text' column (or 'Unnamed: 0' to be mapped to 'Text').
                             It should also contain a 'Type' column for evaluation.
        N (int): The number of clusters for K-Means.
        Name (str): A name for the dataset, used in print statements.
    """
    # Drop rows with any missing values
    Data.dropna(inplace=True)

    # Map 'Unnamed: 0' to 'Text' using the CommentsDict function
    Data['Text'] = Data['Unnamed: 0'].map(CommentsDict)

    # Prepare features (X) and target (y)
    X = Data.drop(['Unnamed: 0', 'Type', 'Text'], axis=1)
    y = Data['Type']

    # Normalize numerical features
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X - mean) / std

    print(f'For File {Name}, has {X.shape[0]} rows, and {X.shape[1]} Features\n..........................\n')

    PredictedDict = {}
    X_array = np.array(X)
    y_array = np.array(y)

    # Convert 'Good'/'Bad' labels to numerical (1/0)
    if 'Good' in y_array:
        CohY = [1 if i == 'Good' else 0 for i in y_array]
    else:
        CohY = y_array

    # K-Means Clustering
    kmeans = KMeans(n_clusters=N, n_init=50, random_state=42)
    kmeans.fit(X_array)
    labels_kmeans = kmeans.labels_

    print('\n\n--------------------------------------------------------\n--------------------- FOR KMEANS---------------------- \n--------------------------------------------------------\n\n')
    print('Predicted Values (Cluster Distribution):')
    print(pd.Series(labels_kmeans).value_counts())
    print('..............')

    Data['Predicted'] = labels_kmeans
    PredictedDict['Kmeans'] = dict(pd.Series(labels_kmeans).value_counts())
    PredictedDict['Kmeans']['SScore'] = silhouette_score(X_array, labels_kmeans)
    PredictedDict['Kmeans']['CScore'] = cohen_kappa_score(CohY, labels_kmeans)

    # --- Prepare Text Data for CTF-IDF ---
    AllText = []
    for cluster_id in Data['Predicted'].unique():
        cluster_text = ' '.join([str(text) for text in Data[Data['Predicted'] == cluster_id]['Text'].tolist()])
        AllText.append(cluster_text)

    ThisDF = pd.DataFrame()
    ThisDF['Text'] = AllText
    ThisDF['Cluster'] = Data['Predicted'].unique()

    print("\nDataFrame for TF-IDF Vectorization:")
    print(ThisDF.head())

    Documents = ThisDF['Text'].tolist() # List of cluster documents

    # --- Custom TF-IDF Helper Functions ---
    def tokenize(text: str):
        """Converts text to lowercase and splits into tokens."""
        text = text.lower()
        tokens = text.split()
        return tokens

    def compute_tf(word_dict: Counter, doc: list):
        """Computes Term Frequency (TF) for a given document."""
        tf_dict = {}
        doc_length = len(doc)
        for word, count in word_dict.items():
            tf_dict[word] = count / float(doc_length)
        return tf_dict

    def compute_idf(term_freq_documents: list[Counter], original_documents: list[str]):
        """
        Computes Inverse Document Frequency (IDF) using a smoothed formula.
        The formula is K + (1-K) * log(1 + (N / (1 + doc_count)))
        """
        N_docs = len(term_freq_documents) # Number of cluster documents
        idf_dict = dict.fromkeys(term_freq_documents[0].keys(), 0)

        # Count documents containing each word
        for document_word_counts in term_freq_documents:
            for word, count in document_word_counts.items():
                if count > 0:
                    idf_dict[word] = idf_dict.get(word, 0) + 1

        K_constant = 0.3 # Smoothing constant
        for word, doc_count in idf_dict.items():
            # Apply the smoothed IDF formula
            idf_dict[word] = (K_constant) + ((1 - K_constant) * math.log(1 + (N_docs / (1 + float(doc_count)))))
        return idf_dict

    def compute_tfidf(tf: dict, idf: dict):
        """Computes TF-IDF by multiplying TF and IDF."""
        tfidf = {}
        for word, val in tf.items():
            tfidf[word] = val * idf[word]
        return tfidf

    # Process documents for custom TF-IDF
    tokenized_documents = [tokenize(doc) for doc in Documents]
    term_freq_docs = [Counter(doc) for doc in tokenized_documents]
    tf_docs = [compute_tf(doc, tokenized_documents[i]) for i, doc in enumerate(term_freq_docs)]
    idf = compute_idf(term_freq_docs, Documents)
    tfidf_docs = [compute_tfidf(tf_doc, idf) for tf_doc in tf_docs]

    # --- Standard TfidfVectorizer for comparison (if needed, otherwise remove) ---
    Dict_standard_tfidf = {'ngram_range': (1, 1), 'max_df': 1.0, 'min_df': 1, 'norm': 'l2'}
    VecModel = TfidfVectorizer(ngram_range=Dict_standard_tfidf['ngram_range'],
                               max_df=Dict_standard_tfidf['max_df'],
                               min_df=Dict_standard_tfidf['min_df'],
                               norm=Dict_standard_tfidf['norm'])
    X_Vec = VecModel.fit_transform(ThisDF['Text'])
    X_Vec = pd.DataFrame.sparse.from_spmatrix(X_Vec)
    Names = {x: y for y, x in VecModel.vocabulary_.items()}

    # --- Print Top Words for each Cluster ---
    for e, cluster_id, tfidf_scores_for_cluster in zip(range(len(Data['Predicted'].unique())),
                                                      Data['Predicted'].unique(),
                                                      tfidf_docs):
        print(f'\nMost Repeated Words in Cluster {cluster_id} (Simple Word Count):')
        cluster_words = ' '.join([str(j) for j in Data[Data['Predicted'] == cluster_id]['Text'].tolist()]).split()
        most_common_words = pd.Series(cluster_words).value_counts().keys().tolist()
        filtered_words = [k for k in most_common_words if k.lower() not in stopwords.words('english')][:20]
        print(filtered_words)

        # Top words using Modified CTF-IDF
        Words2 = tfidf_scores_for_cluster
        WordSortes2 = dict(sorted(Words2.items(), key=operator.itemgetter(1), reverse=True))
        TopWordsValues2 = {k: np.round(v, 4) for k, v in dict(list(WordSortes2.items())[:100]).items()
                           if k.lower() not in stopwords.words('english')}
        TopWords2 = [k for k, v in TopWordsValues2.items()]
        print(f'TopWords using Modified CTFIDF:\n{TopWords2}')

        # Top words using standard CTFIDF (for comparison)
        ResultDict_standard = {m: [n, Names[m]] for m, n in enumerate(X_Vec.loc[e].tolist()) if n != 0}
        Words_standard = {k: v for v, k in ResultDict_standard.values()}
        WordSortes_standard = dict(sorted(Words_standard.items(), key=operator.itemgetter(1), reverse=True))
        TopWordsValues_standard = {k: np.round(v, 4) for k, v in dict(list(WordSortes_standard.items())[:100]).items()
                                   if k.lower() not in stopwords.words('english')}
        TopWords_standard = [k for k, v in TopWordsValues_standard.items()]
        print(f'TopWords using CTFIDF (for comparison):\n{TopWords_standard}')

        print('\n........................................\n')

    return PredictedDict