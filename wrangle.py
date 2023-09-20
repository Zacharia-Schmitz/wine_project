import os
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, levene, mannwhitneyu
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


def check_columns(df, reports=False, graphs=False):
    """
    This function takes a pandas dataframe as input and returns
    a dataframe with information about each column in the dataframe. For
    each column, it returns the column name, the number of
    unique values in the column, the unique values themselves,
    the number of null values in the column, the proportion of null values,
    the data type of the column, and the range of the column if it is float or int. The resulting dataframe is sorted by the
    'Number of Unique Values' column in ascending order.

    Args:
    - df: pandas dataframe

    Returns:
    - pandas dataframe
    """
    print(f"Total rows: {df.shape[0]}")
    print(f"Total columns: {df.shape[1]}")
    if reports == True:
        describe = df.describe().round(2)
        pd.DataFrame(describe)
        print(describe)
    if graphs == True:
        df.hist(bins=20, figsize=(10, 10))
        plt.show()
    data = []
    # Loop through each column in the dataframe
    for column in df.columns:
        # Append the column name, number of unique values, unique values, number of null values, proportion of null values, and data type to the data list
        if df[column].dtype in ["float64", "int64"]:
            data.append(
                [
                    column,
                    df[column].dtype,
                    df[column].nunique(),
                    df[column].isna().sum(),
                    df[column].isna().mean().round(5),
                    df[column].unique(),
                    df[column].describe()[["min", "max", "mean"]].values,
                ]
            )
        else:
            data.append(
                [
                    column,
                    df[column].dtype,
                    df[column].nunique(),
                    df[column].isna().sum(),
                    df[column].isna().mean().round(5),
                    df[column].unique(),
                    None,
                ]
            )
    # Create a pandas dataframe from the data list, with column names 'Column Name', 'Number of Unique Values', 'Unique Values', 'Number of Null Values', 'Proportion of Null Values', 'dtype', and 'Range' (if column is float or int)
    # Sort the resulting dataframe by the 'Number of Unique Values' column in ascending order
    return pd.DataFrame(
        data,
        columns=[
            "col_name",
            "dtype",
            "num_unique",
            "num_null",
            "pct_null",
            "unique_values",
            "range (min, max, mean)",
        ],
    )


def load_wine_data():
    """
    Loads the wine data from the two csvs if they exist, otherwise imports the two files and creates wine.csv.

    Returns:
    df (pandas.DataFrame): The wine data as a DataFrame.
    """
    red = pd.read_csv("winequality_red.csv")
    white = pd.read_csv("winequality_white.csv")
    # Create a seperator to differentiate
    red["is_red"] = 1
    white["is_red"] = 0
    # Merge them and cache
    df = pd.concat([red, white], axis=0)
    return df


def box_plotter(df):
    """
    Generates a box plot for all columns in a dataframe using matplotlib.
    """
    for col in df.columns:
        if col != "quality" and col != "is_red":
            try:
                plt.figure(figsize=(12, 1))
                plt.boxplot(df[col], vert=False)
                plt.title(col)
                plt.show()
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                print(
                    f"Number of results in lower quartile: {len(df[df[col] < lower_bound])} ({(len(df[df[col] < lower_bound])/len(df))*100:.2f}%)"
                )
                print(
                    f"Number of results in inner quartile: {len(df[(df[col] >= lower_bound) & (df[col] <= upper_bound)])} ({(len(df[(df[col] >= lower_bound) & (df[col] <= upper_bound)])/len(df))*100:.2f}%)"
                )
                print(
                    f"Number of results in upper quartile: {len(df[df[col] > upper_bound])} ({(len(df[df[col] > upper_bound])/len(df))*100:.2f}%)"
                )
            except:
                print(
                    f"Error: Could not generate box plot for column {col}. Skipping to next column..."
                )
                plt.close()
                continue


def split_data(df, random_state=123):
    """Split into train, validate, test with a 60% train, 20% validate, 20% test"""
    train_validate, test = train_test_split(df, test_size=0.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=0.25, random_state=123)

    print(f"train: {len(train)} ({round(len(train)/len(df)*100)}% of {len(df)})")
    print(
        f"validate: {len(validate)} ({round(len(validate)/len(df)*100)}% of {len(df)})"
    )
    print(f"test: {len(test)} ({round(len(test)/len(df)*100)}% of {len(df)})")
    return train, validate, test


def plot_quality_vs_feature(df, feature):
    """
    Plots the relationship between each feature and the quality of wine.

    Parameters:
    df (pandas.DataFrame): The wine data as a DataFrame.
    feature (str): The name of the feature to plot against quality.

    Returns:
    None
    """
    # Loop through each column in the DataFrame
    for col in df.columns:
        if col != "quality" and col != "is_red":
            try:
                # Create a line plot of the feature vs quality
                plt.figure(figsize=(15, 7))
                sns.lineplot(data=df, x=feature, y=col)

                # Test for significant difference between quality groups using Mann-Whitney U test
                group1 = df[df["quality_binned"] == "lower"][col]
                group2 = df[df["quality_binned"] == "upper"][col]
                stat, p = mannwhitneyu(group1, group2)

                # Add the test result to the plot title
                plt.title(
                    f"{col} vs {feature} (Mann-Whitney U test: U={stat:.4f}, p={p:.4f})",
                    fontsize=16,
                )
                plt.show()
            except:
                continue


def test_normality_and_variance(df, target="quality"):
    """
    Tests for normality and equal variance of each feature in the DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to test.

    Returns:
    None
    """
    # Ask for the target variable
    target = target

    # Test for normality and equal variance
    for col in df.columns:
        if col != "quality" and col != "is_red":
            stat, p = shapiro(df[col])
            print(f"{col}: {'Normal' if p > 0.05 else 'Not normal'}")

            stat2, p2 = levene(df[target], df[col])
            print(f"{col}: {'Equal variance' if p2 > 0.05 else 'Not equal variance'}\n")


def mm_scale_data(X_train, X_validate, X_test):
    """
    Scales the training, validation, and test data using MinMaxScaler.

    Parameters:
    X_train (pandas.DataFrame): The training data as a DataFrame.
    X_validate (pandas.DataFrame): The validation data as a DataFrame.
    X_test (pandas.DataFrame): The test data as a DataFrame.

    Returns:
    tuple: A tuple containing the scaled training, validation, and test data.
    """
    # Create a MinMaxScaler object
    scaler = MinMaxScaler()

    # Fit the scaler on X_train
    scaler.fit(X_train)

    # Transform X_train, X_validate, and X_test using the scaler
    X_train_scaled = scaler.transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)
    X_test_scaled = scaler.transform(X_test)

    # Return a tuple containing the scaled data
    return X_train_scaled, X_validate_scaled, X_test_scaled


def plot_elbow_and_silhouette(df):
    """
    Plots the elbow curve and silhouette score for a range of cluster numbers.

    Parameters:
    df (pandas.DataFrame): The training data as a DataFrame.

    Returns:
    None
    """
    # Create an empty list to store the sum of squared distances for each number of clusters
    elbow = []

    # Iterate over the range of possible numbers of clusters
    for i in range(1, 10):
        # Create a KMeans object with the current number of clusters
        kmeans = KMeans(n_clusters=i, random_state=123)

        # Fit the KMeans object to the training data
        kmeans.fit(df)

        # Append the sum of squared distances to the elbow list
        elbow.append(kmeans.inertia_)

    # Create a line plot of the sum of squared distances for each number of clusters
    plt.plot(range(1, 10), elbow, marker="o", markersize=15)
    plt.title("The Elbow Method")
    plt.xlabel("Number of clusters")
    plt.show()

    l1 = [2, 3, 4, 5, 6, 7, 8, 9]
    silhouette_avg = []
    for i in l1:
        kmeans = KMeans(n_clusters=i, random_state=123)
        kmeans.fit(df)
        cluster_labels = kmeans.labels_
        pca = PCA(n_components=min(i, df.shape[1]))
        reduced_X = pd.DataFrame(data=pca.fit_transform(df))
        silhouette_avg.append(silhouette_score(reduced_X, kmeans.labels_))

    plt.plot(l1, silhouette_avg, marker="o", markersize=15)
    plt.xlabel("Values of K")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette analysis For Optimal k")
    plt.show()


def plot_rfc_scores(rfc_scores):
    """
    Plots the accuracy scores for the training and validation data.

    Parameters:
    rfc_scores (pandas.DataFrame): The dataframe containing the accuracy scores.

    Returns:
    None
    """
    # Sort the dataframe by 'acc_diff' and reset the index
    rfc_scores.sort_values(by=["acc_diff"], inplace=True)
    rfc_scores.reset_index(inplace=True)

    # Create a line plot of the accuracy scores for the training and validation data
    plt.figure(figsize=(8, 5))
    plt.plot(rfc_scores.index, rfc_scores.acc_train, label="Train", marker="o")
    plt.plot(rfc_scores.index, rfc_scores.acc_val, label="Validate", marker="o")
    plt.fill_between(
        rfc_scores.index, rfc_scores.acc_train, rfc_scores.acc_val, alpha=0.2
    )
    plt.xlabel("Model Number as Index in DF", fontsize=10)
    plt.ylabel("Accuracy", fontsize=14)
    plt.title(f"Classification Model Performance: RandomForest", fontsize=18)
    plt.legend(title="Scores", fontsize=12)
    plt.xlim(0, 210)
    plt.ylim(0.52, 0.56)
    plt.xticks(np.arange(0, 220, 20))
    plt.show()


def merge_and_filter_dataframes():
    """
    Merges the dataframes and filters the data to only show models with a difference less than 0.1.
    Sorts the data by 'acc_val'.

    Parameters:
    None

    Returns:
    pandas.DataFrame: A dataframe containing the merged and filtered data.
    """
    # Merge the dataframes
    rfc_scores = pd.concat(
        [
            rfc_scores_robust_3,
            rfc_scores_mm_3,
            rfc_scores_standard_3,
            rfc_scores_robust_4,
            rfc_scores_mm_4,
            rfc_scores_robust_5,
            rfc_scores_mm_5,
            rfc_scores_standard_5,
        ]
    )

    # Round the accuracy scores to 3 decimals
    rfc_scores["acc_val"] = [round(float(x), 3) for x in rfc_scores["acc_val"]]
    rfc_scores["acc_train"] = [round(float(x), 3) for x in rfc_scores["acc_train"]]
    rfc_scores["acc_diff"] = [round(float(x), 3) for x in rfc_scores["acc_diff"]]

    # Filter the data to only show models with a difference less than 0.1
    rfc_scores_10 = rfc_scores[rfc_scores["acc_diff"] < 0.1]

    # Sort the data by 'acc_val'
    rfc_scores_10_sorted = rfc_scores_10.sort_values(by="acc_val", ascending=False)

    # Return the merged and filtered dataframe
    return rfc_scores_10_sorted


def cluster_and_model_test(
    X_train,
    y_train,
    X_validate,
    y_validate,
    scaler="standard",
    n_clusters=5,
    model="all",
    n_estimators=100,
    max_depth=6,
    min_samples_split=2,
    min_samples_leaf=2,
):
    """
    Clusters the data using KMeans with the given number of clusters, adds the cluster labels to the original data,
    and trains and evaluates RandomForestClassifier, KNeighborsClassifier, LogisticRegression, and DecisionTreeClassifier
    models on the data.

    Parameters:
    X_train (pandas.DataFrame): The training data to cluster and model.
    y_train (pandas.Series): The training target variable.
    X_validate (pandas.DataFrame): The validation data to cluster and model.
    y_validate (pandas.Series): The validation target variable.
    n_clusters (int): The number of clusters to use for KMeans clustering.
    n_estimators (int): The number of trees in the random forest.
    max_depth (int): The maximum depth of the decision trees in the random forest.
    min_samples_split (int): The minimum number of samples required to split an internal node in the decision trees.
    min_samples_leaf (int): The minimum number of samples required to be at a leaf node in the decision trees.

    Returns:
    scores_df (pandas.DataFrame): A DataFrame of classification reports for each model.
    """
    scaler_name = scaler

    if scaler_name == "minmax":
        scaler = MinMaxScaler()
    elif scaler_name == "standard":
        scaler = StandardScaler()
    elif scaler_name == "robust":
        scaler = RobustScaler()

    # Scale the data
    X_train_scaled = scaler.fit_transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)

    # Cluster the data using KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=123)
    kmeans.fit(X_train_scaled)
    X_train_clustered = kmeans.transform(X_train_scaled)
    X_validate_clustered = kmeans.transform(X_validate_scaled)

    # Add the cluster labels to the original data
    X_train_clustered_df = pd.DataFrame(
        X_train_clustered, columns=[f"cluster_{i}" for i in range(n_clusters)]
    )
    X_train_clustered_df.index = X_train.index
    X_train_clustered_df = pd.concat([X_train, X_train_clustered_df], axis=1)

    X_validate_clustered_df = pd.DataFrame(
        X_validate_clustered, columns=[f"cluster_{i}" for i in range(n_clusters)]
    )
    X_validate_clustered_df.index = X_validate.index
    X_validate_clustered_df = pd.concat([X_validate, X_validate_clustered_df], axis=1)

    # Train and evaluate models
    if model == "all":
        models = [
            RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
            ),
            KNeighborsClassifier(),
            LogisticRegression(),
            DecisionTreeClassifier(),
        ]
        scores = dict()
    elif model == "rfc":
        models = [
            RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
            )
        ]
        scores = dict()
    elif model == "knc":
        models = [KNeighborsClassifier()]
        scores = dict()
    elif model == "lr":
        models = [LogisticRegression()]
        scores = dict()
    elif model == "dtc":
        models = [DecisionTreeClassifier()]
        scores = dict()

    for m in models:
        m.fit(X_train_clustered_df, y_train)
        y_train_pred = m.predict(X_train_clustered_df)
        y_validate_pred = m.predict(X_validate_clustered_df)

        report_train = classification_report(y_train, y_train_pred, output_dict=True)
        report_validate = classification_report(
            y_validate, y_validate_pred, output_dict=True
        )

        scores[str(m)] = {
            "acc_train": accuracy_score(y_train, y_train_pred),
            "acc_test": accuracy_score(y_validate, y_validate_pred),
            "scaler": scaler_name,
            "clusters": n_clusters,
            "prec_train": report_train["weighted avg"]["precision"],
            "prec_test": report_validate["weighted avg"]["precision"],
            "recall_train": report_train["weighted avg"]["recall"],
            "recall_test": report_validate["weighted avg"]["recall"],
            "f1_train": report_train["weighted avg"]["f1-score"],
            "f1_test": report_validate["weighted avg"]["f1-score"],
            "supp_train": report_train["weighted avg"]["support"],
            "supp_test": report_validate["weighted avg"]["support"],
        }

    scores_df = pd.DataFrame(scores).transpose()
    scores_df.index.name = "Model"

    return scores_df


def hyper_tuning(
    X_train, y_train, X_validate, y_validate, scaler_name="standard", n_clusters=3
):
    """
    Trains and evaluates a RandomForestClassifier model using different hyperparameters.

    Parameters:
    X_train (pandas.DataFrame): The training data to cluster and model.
    y_train (pandas.Series): The training target variable.
    X_validate (pandas.DataFrame): The validation data to cluster and model.
    y_validate (pandas.Series): The validation target variable.
    scaler_name (str): The name of the scaler to use. Must be one of 'minmax', 'standard', or 'robust'.
    n_clusters (int): The number of clusters to use for KMeans clustering.

    Returns:
    scores_df (pandas.DataFrame): A DataFrame of classification reports for each model.
    """
    # Define the scaler
    if scaler_name == "minmax":
        scaler = MinMaxScaler()
    elif scaler_name == "standard":
        scaler = StandardScaler()
    elif scaler_name == "robust":
        scaler = RobustScaler()

    # Scale the data
    X_train_scaled = scaler.fit_transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)

    # Cluster the data using KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=123)
    kmeans.fit(X_train_scaled)
    X_train_clustered = kmeans.transform(X_train_scaled)
    X_validate_clustered = kmeans.transform(X_validate_scaled)

    # Add the cluster labels to the original data
    X_train_clustered_df = pd.DataFrame(
        X_train_clustered, columns=[f"cluster_{i}" for i in range(n_clusters)]
    )
    X_train_clustered_df.index = X_train.index
    X_train_clustered_df = pd.concat([X_train, X_train_clustered_df], axis=1)

    X_validate_clustered_df = pd.DataFrame(
        X_validate_clustered, columns=[f"cluster_{i}" for i in range(n_clusters)]
    )
    X_validate_clustered_df.index = X_validate.index
    X_validate_clustered_df = pd.concat([X_validate, X_validate_clustered_df], axis=1)

    # Define the hyperparameters to search over
    n_estimators = [100, 200, 300]
    max_depth = [3, 6, 10]
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]

    # Create a list of all possible combinations of hyperparameters
    hyperparameter_grid = list(
        itertools.product(n_estimators, max_depth, min_samples_split, min_samples_leaf)
    )

    # Train and evaluate models using different hyperparameters
    scores = dict()

    for params in hyperparameter_grid:
        rfc = RandomForestClassifier(
            n_estimators=params[0],
            max_depth=params[1],
            min_samples_split=params[2],
            min_samples_leaf=params[3],
        )
        rfc.fit(X_train_clustered_df, y_train)
        y_train_pred = rfc.predict(X_train_clustered_df)
        y_validate_pred = rfc.predict(X_validate_clustered_df)

        report_train = classification_report(y_train, y_train_pred, output_dict=True)
        report_validate = classification_report(
            y_validate, y_validate_pred, output_dict=True
        )

        scores[str(params)] = {
            "acc_train": accuracy_score(y_train, y_train_pred),
            "acc_val": accuracy_score(y_validate, y_validate_pred),
            "acc_diff": abs(
                accuracy_score(y_train, y_train_pred)
                - accuracy_score(y_validate, y_validate_pred)
            ),
            "scaler": scaler_name,
            "clusters": n_clusters,
            "prec_train": report_train["weighted avg"]["precision"],
            "prec_val": report_validate["weighted avg"]["precision"],
            "recall_train": report_train["weighted avg"]["recall"],
            "recall_val": report_validate["weighted avg"]["recall"],
            "f1_train": report_train["weighted avg"]["f1-score"],
            "f1_val": report_validate["weighted avg"]["f1-score"],
            "supp_train": report_train["weighted avg"]["support"],
            "supp_val": report_validate["weighted avg"]["support"],
        }

    scores_df = pd.DataFrame(scores).transpose()
    scores_df.index.name = "Hyperparameters"

    return scores_df


def rfc_results(
    X_train, y_train, X_validate, y_validate, scaler_names, n_clusters_list
):
    results = []
    for scaler_name in scaler_names:
        for n_clusters in n_clusters_list:
            scores = w.hyper_tuning(
                X_train,
                y_train,
                X_validate,
                y_validate,
                scaler_name=scaler_name,
                n_clusters=n_clusters,
            )
            scores["scaler_name"] = scaler_name
            scores["n_clusters"] = n_clusters
            results.append(scores)
    df = pd.DataFrame(results)
    return df


def calculate_baseline_accuracy(y_train, y_validate):
    """
    Calculates the baseline accuracy for a classification problem.

    Parameters:
    y_train (pandas.Series): The training target variable.
    y_validate (pandas.Series): The validation target variable.

    Returns:
    None
    """
    # Calculate the baseline accuracy
    baseline_acc = y_train.mean()

    # Calculate the accuracy of the baseline prediction on the validation set
    baseline_pred = [y_train.mode()[0]] * len(y_validate)
    baseline_acc = accuracy_score(y_validate, baseline_pred)

    # Print the baseline accuracy on the validation set
    print(f"Baseline accuracy on validation set: {baseline_acc:.4f}")
