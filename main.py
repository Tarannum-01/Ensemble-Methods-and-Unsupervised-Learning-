# %%
#### EDA and dataset detail ###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Set random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Load datasets
student_data = pd.read_csv('Student_performance_data.csv')
dermatology_data = pd.read_csv('dermatology_database_1.csv')

# Preprocess 'age' in Dermatology Data
dermatology_data['age'] = pd.to_numeric(dermatology_data['age'], errors='coerce')
print(f"Missing values in 'age': {dermatology_data['age'].isnull().sum()}")

# Fill missing values in 'age' with median
dermatology_data['age'].fillna(dermatology_data['age'].median(), inplace=True)

# Standardize numeric features
scaler = StandardScaler()
student_scaled = scaler.fit_transform(student_data.select_dtypes(include=['float64', 'int64']))
dermatology_scaled = scaler.fit_transform(dermatology_data.drop(columns=['class']))

# EDA: Dataset Summary
print("\nStudent Data Info:")
print(student_data.info())
print(student_data.describe())

print("\nDermatology Data Info:")
print(dermatology_data.info())
print(dermatology_data.describe())

# EDA: Missing Values
print("\nMissing Values in Student Data:")
print(student_data.isnull().sum())

print("\nMissing Values in Dermatology Data:")
print(dermatology_data.isnull().sum())

# EDA: Outlier Detection
plt.figure(figsize=(15, 8))
student_data.boxplot()
plt.title('Student Data - Outlier Detection')
plt.show()

plt.figure(figsize=(15, 8))
dermatology_data.boxplot()
plt.title('Dermatology Data - Outlier Detection')
plt.show()

# EDA: Correlation Heatmaps
plt.figure(figsize=(12, 8))
sns.heatmap(student_data.corr(), annot=True, cmap='coolwarm')
plt.title('Student Data - Correlation Heatmap')
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(dermatology_data.corr(), annot=True, cmap='coolwarm')
plt.title('Dermatology Data - Correlation Heatmap')
plt.show()

# %%
### K-Means Clustering & Hierarchical Clustering  for 2 dataset ###
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import numpy as np

# Standardize Data
scaler = StandardScaler()
student_scaled = scaler.fit_transform(student_data.select_dtypes(include=['float64', 'int64']))
dermatology_scaled = scaler.fit_transform(dermatology_data.drop(columns=['class']))

# Elbow Plot Function (Side-by-Side)
def elbow_plot_side_by_side(student_data, dermatology_data):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    for ax, data, title in zip(axes, [student_data, dermatology_data], ["Student Data", "Dermatology Data"]):
        distortions = []
        for k in range(1, 11):  # Test k from 1 to 10
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data)
            distortions.append(kmeans.inertia_)
        ax.plot(range(1, 11), distortions, marker='o')
        ax.set_title(f'Elbow Method ({title})')
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('Distortion')
    
    plt.tight_layout()
    plt.show()

# Hierarchical Clustering - Dendrogram Function (Side-by-Side)
def dendrogram_side_by_side(student_data, dermatology_data):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for ax, data, title in zip(axes, [student_data, dermatology_data], ["Student Data", "Dermatology Data"]):
        linked = linkage(data, method='ward')
        dendrogram(linked, ax=ax, truncate_mode='lastp', p=10, show_leaf_counts=True)
        ax.set_title(f'Hierarchical Clustering Dendrogram ({title})')
        ax.set_xlabel('Data Points')
        ax.set_ylabel('Distance')
    
    plt.tight_layout()
    plt.show()

# Clustering Visualization (Side-by-Side)
def clustering_visualization_side_by_side(student_data, dermatology_data, student_clusters, dermatology_clusters, title1, title2):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    # Scatter Plot for Student Data
    axes[0].scatter(student_data[:, 0], student_data[:, 1], c=student_clusters, cmap='viridis', s=50, alpha=0.7)
    axes[0].set_title(f'{title1}')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    
    # Scatter Plot for Dermatology Data
    axes[1].scatter(dermatology_data[:, 0], dermatology_data[:, 1], c=dermatology_clusters, cmap='viridis', s=50, alpha=0.7)
    axes[1].set_title(f'{title2}')
    axes[1].set_xlabel('Feature 1')
    
    plt.tight_layout()
    plt.show()

# Perform K-Means Clustering
kmeans_student = KMeans(n_clusters=3, random_state=42)
student_clusters = kmeans_student.fit_predict(student_scaled)
student_silhouette = silhouette_score(student_scaled, student_clusters)

kmeans_dermatology = KMeans(n_clusters=6, random_state=42)
dermatology_clusters = kmeans_dermatology.fit_predict(dermatology_scaled)
dermatology_silhouette = silhouette_score(dermatology_scaled, dermatology_clusters)

# Perform Hierarchical Clustering
student_linked = linkage(student_scaled, method='ward')
student_hierarchical_clusters = fcluster(student_linked, t=3, criterion='maxclust')
student_hierarchical_silhouette = silhouette_score(student_scaled, student_hierarchical_clusters)

dermatology_linked = linkage(dermatology_scaled, method='ward')
dermatology_hierarchical_clusters = fcluster(dermatology_linked, t=6, criterion='maxclust')
dermatology_hierarchical_silhouette = silhouette_score(dermatology_scaled, dermatology_hierarchical_clusters)

# Print Silhouette Scores
print(f"K-Means Silhouette Score (Student Data): {student_silhouette:.3f}")
print(f"K-Means Silhouette Score (Dermatology Data): {dermatology_silhouette:.3f}")
print(f"Hierarchical Silhouette Score (Student Data): {student_hierarchical_silhouette:.3f}")
print(f"Hierarchical Silhouette Score (Dermatology Data): {dermatology_hierarchical_silhouette:.3f}")

# Generate Elbow Plots
elbow_plot_side_by_side(student_scaled, dermatology_scaled)

# Generate Dendrograms
dendrogram_side_by_side(student_scaled, dermatology_scaled)

# Visualize K-Means Clustering
clustering_visualization_side_by_side(
    student_scaled, dermatology_scaled, 
    student_clusters, dermatology_clusters,
    "K-Means Clusters (Student Data)", "K-Means Clusters (Dermatology Data)"
)

# Visualize Hierarchical Clustering
clustering_visualization_side_by_side(
    student_scaled, dermatology_scaled, 
    student_hierarchical_clusters, dermatology_hierarchical_clusters,
    "Hierarchical Clusters (Student Data)", "Hierarchical Clusters (Dermatology Data)"
)


# %%
###### PCA & UMAP  for 2 dataset ###
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# PCA with Scree Plot and Histograms
def run_pca_with_visuals(data, n_components=5, title="Dataset"):
    pca = PCA(n_components=n_components, random_state=42)
    transformed_data = pca.fit_transform(data)
    explained_variance = pca.explained_variance_ratio_ * 100

    # Scree Plot
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, n_components + 1), explained_variance, alpha=0.7, color='skyblue', label='% Variance Explained')
    plt.title(f'Scree Plot for {title}')
    plt.xlabel('Principal Components (PC1, PC2, ...)')
    plt.ylabel('Explained Variance (%)')
    plt.xticks(range(1, n_components + 1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Explained Variance by Principal Components ({title}):", explained_variance)

    # Histograms for Principal Components
    fig, axes = plt.subplots(1, n_components, figsize=(20, 4), sharey=True)
    for i in range(n_components):
        sns.histplot(transformed_data[:, i], bins=30, kde=True, ax=axes[i], color='skyblue')
        axes[i].set_title(f'PC{i+1}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
    plt.suptitle(f'Principal Component Histograms ({title})', y=1.05)
    plt.tight_layout()
    plt.show()

    return transformed_data, explained_variance

# Run PCA on datasets
student_pca, student_pca_variance = run_pca_with_visuals(student_scaled, n_components=5, title="Student Data")
dermatology_pca, dermatology_pca_variance = run_pca_with_visuals(dermatology_scaled, n_components=5, title="Dermatology Data")

# Perform UMAP for comparison
def run_umap(data, n_neighbors=15, min_dist=0.1, n_components=2, title="Dataset"):
    umap_reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=42)
    transformed_data = umap_reducer.fit_transform(data)
    return transformed_data

# Run UMAP
student_umap = run_umap(student_scaled, title="Student Data")
dermatology_umap = run_umap(dermatology_scaled, title="Dermatology Data")

# Side-by-Side Comparison for PCA and UMAP Projections
def plot_dimensionality_reduction(pca_data, umap_data, clusters, title):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # PCA Projection
    sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=clusters, palette="tab10", ax=axes[0], s=50)
    axes[0].set_title(f'PCA Projection ({title})')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].legend(title="Cluster")

    # UMAP Projection
    sns.scatterplot(x=umap_data[:, 0], y=umap_data[:, 1], hue=clusters, palette="tab10", ax=axes[1], s=50)
    axes[1].set_title(f'UMAP Projection ({title})')
    axes[1].set_xlabel('UMAP Dimension 1')
    axes[1].set_ylabel('UMAP Dimension 2')
    axes[1].legend(title="Cluster")

    plt.tight_layout()
    plt.show()

# Side-by-side comparison plots
plot_dimensionality_reduction(student_pca, student_umap, student_clusters, "Student Data")
plot_dimensionality_reduction(dermatology_pca, dermatology_umap, dermatology_clusters, "Dermatology Data")

# %%
#####################################
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import umap

# Optimize PCA to retain 95% variance
def optimized_pca(data, variance_threshold=0.95):
    pca = PCA(n_components=variance_threshold, random_state=42)
    transformed_data = pca.fit_transform(data)
    explained_variance = pca.explained_variance_ratio_ * 100  

    # Scree Plot with cumulative variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, label='Individual Variance')
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', color='red', label='Cumulative Variance')
    plt.axhline(y=variance_threshold, color='green', linestyle='--', label='Threshold')
    plt.title('Optimized PCA Scree Plot')
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance (%)')
    plt.legend()
    plt.grid()
    plt.show()

    print(f"Retained {len(transformed_data[0])} components to explain {variance_threshold*100}% variance.")
    return transformed_data, explained_variance, pca

# PCA Analysis
student_pca_optimized, student_pca_variance, student_pca_model = optimized_pca(student_scaled, variance_threshold=0.95)
dermatology_pca_optimized, dermatology_pca_variance, dermatology_pca_model = optimized_pca(dermatology_scaled, variance_threshold=0.95)

# Combine PCA with UMAP for hybrid reduction
def hybrid_pca_umap(data, n_neighbors=15, min_dist=0.1):
    # PCA for noise reduction
    pca = PCA(n_components=0.95, random_state=42)
    pca_data = pca.fit_transform(data)
    # UMAP for non-linear separation
    umap_reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    umap_data = umap_reducer.fit_transform(pca_data)

    return umap_data

# Hybrid PCA + UMAP
student_hybrid = hybrid_pca_umap(student_scaled)
dermatology_hybrid = hybrid_pca_umap(dermatology_scaled)

# Validate Performance of Optimized PCA and Hybrid PCA + UMAP
print(f"Student PCA: {len(student_pca_optimized[0])} components retained.")
print(f"Dermatology PCA: {len(dermatology_pca_optimized[0])} components retained.")

# %%
import pandas as pd

# Convert PCA results into a DataFrame
student_pca_df = pd.DataFrame(student_pca, columns=[f"PC{i+1}" for i in range(student_pca.shape[1])])
dermatology_pca_df = pd.DataFrame(dermatology_pca, columns=[f"PC{i+1}" for i in range(dermatology_pca.shape[1])])

# Save to CSV
student_pca_df.to_csv('Student_Data_PCA_Reduced.csv', index=False)
dermatology_pca_df.to_csv('Dermatology_Data_PCA_Reduced.csv', index=False)

print("PCA reduced datasets saved as CSV!")

# Convert UMAP results into a DataFrame
student_umap_df = pd.DataFrame(student_umap, columns=["UMAP_Dim1", "UMAP_Dim2"])
dermatology_umap_df = pd.DataFrame(dermatology_umap, columns=["UMAP_Dim1", "UMAP_Dim2"])

# Save to CSV
student_umap_df.to_csv('Student_Data_UMAP_Reduced.csv', index=False)
dermatology_umap_df.to_csv('Dermatology_Data_UMAP_Reduced.csv', index=False)

print("UMAP reduced datasets saved as CSV!")

# Add clusters to PCA data
student_pca_df['Cluster'] = student_clusters
dermatology_pca_df['Cluster'] = dermatology_clusters

# Add clusters to UMAP data
student_umap_df['Cluster'] = student_clusters
dermatology_umap_df['Cluster'] = dermatology_clusters

# %%
import pandas as pd

# Load PCA-reduced datasets
student_pca_reduced = pd.read_csv('Student_Data_PCA_Reduced.csv')
dermatology_pca_reduced = pd.read_csv('Dermatology_Data_PCA_Reduced.csv')

# Load UMAP-reduced datasets
student_umap_reduced = pd.read_csv('Student_Data_UMAP_Reduced.csv')
dermatology_umap_reduced = pd.read_csv('Dermatology_Data_UMAP_Reduced.csv')

# Display sample data
print("Sample of PCA-reduced Student Dataset:")
print(student_pca_reduced.head())

print("\nSample of UMAP-reduced Dermatology Dataset:")
print(dermatology_umap_reduced.head())

# %%
### K-Means Clustering & Hierarchical Clustering  for 2 reduced dataset ###
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt

# Load PCA-reduced and UMAP-reduced datasets
student_pca = pd.read_csv('Student_Data_PCA_Reduced.csv')
dermatology_pca = pd.read_csv('Dermatology_Data_PCA_Reduced.csv')

student_umap = pd.read_csv('Student_Data_UMAP_Reduced.csv')
dermatology_umap = pd.read_csv('Dermatology_Data_UMAP_Reduced.csv')

# Elbow Plot Function
def elbow_plot_side_by_side(data_pca, data_umap, title_pca, title_umap):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax, data, title in zip(axes, [data_pca, data_umap], [title_pca, title_umap]):
        distortions = []
        for k in range(1, 11):  # Test k from 1 to 10
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data)
            distortions.append(kmeans.inertia_)
        ax.plot(range(1, 11), distortions, marker='o')
        ax.set_title(f'Elbow Method ({title})')
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('Distortion')
    plt.tight_layout()
    plt.show()

# Dendrogram Function
def dendrogram_side_by_side(data_pca, data_umap, title_pca, title_umap):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, data, title in zip(axes, [data_pca, data_umap], [title_pca, title_umap]):
        linked = linkage(data, method='ward')
        dendrogram(linked, ax=ax, truncate_mode='lastp', p=10, show_leaf_counts=True)
        ax.set_title(f'Hierarchical Clustering Dendrogram ({title})')
        ax.set_xlabel('Data Points')
        ax.set_ylabel('Distance')
    plt.tight_layout()
    plt.show()

# Clustering Visualization
def clustering_visualization_side_by_side(data_pca, data_umap, clusters_pca, clusters_umap, title_pca, title_umap):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # PCA Projection
    axes[0].scatter(data_pca.iloc[:, 0], data_pca.iloc[:, 1], c=clusters_pca, cmap='viridis', s=50, alpha=0.7)
    axes[0].set_title(f'{title_pca}')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')

    # UMAP Projection
    axes[1].scatter(data_umap.iloc[:, 0], data_umap.iloc[:, 1], c=clusters_umap, cmap='viridis', s=50, alpha=0.7)
    axes[1].set_title(f'{title_umap}')
    axes[1].set_xlabel('UMAP Dimension 1')
    axes[1].set_ylabel('UMAP Dimension 2')

    plt.tight_layout()
    plt.show()

# Perform K-Means Clustering
def kmeans_clustering(data, n_clusters, title):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    silhouette = silhouette_score(data, clusters)
    print(f"K-Means Silhouette Score ({title}): {silhouette:.3f}")
    return clusters

# Perform Hierarchical Clustering
def hierarchical_clustering(data, n_clusters, title):
    linked = linkage(data, method='ward')
    clusters = fcluster(linked, t=n_clusters, criterion='maxclust')
    silhouette = silhouette_score(data, clusters)
    print(f"Hierarchical Silhouette Score ({title}): {silhouette:.3f}")
    return clusters

# Run clustering on PCA-Reduced Data
student_kmeans_pca = kmeans_clustering(student_pca, n_clusters=3, title="Student Data (PCA)")
dermatology_kmeans_pca = kmeans_clustering(dermatology_pca, n_clusters=6, title="Dermatology Data (PCA)")
student_hierarchical_pca = hierarchical_clustering(student_pca, n_clusters=3, title="Student Data (PCA)")
dermatology_hierarchical_pca = hierarchical_clustering(dermatology_pca, n_clusters=6, title="Dermatology Data (PCA)")

# Run clustering on UMAP-Reduced Data
student_kmeans_umap = kmeans_clustering(student_umap, n_clusters=3, title="Student Data (UMAP)")
dermatology_kmeans_umap = kmeans_clustering(dermatology_umap, n_clusters=6, title="Dermatology Data (UMAP)")
student_hierarchical_umap = hierarchical_clustering(student_umap, n_clusters=3, title="Student Data (UMAP)")
dermatology_hierarchical_umap = hierarchical_clustering(dermatology_umap, n_clusters=6, title="Dermatology Data (UMAP)")

# Generate Elbow Plots
elbow_plot_side_by_side(student_pca, student_umap, "Student Data (PCA)", "Student Data (UMAP)")
elbow_plot_side_by_side(dermatology_pca, dermatology_umap, "Dermatology Data (PCA)", "Dermatology Data (UMAP)")

# Generate Dendrograms
dendrogram_side_by_side(student_pca, student_umap, "Student Data (PCA)", "Student Data (UMAP)")
dendrogram_side_by_side(dermatology_pca, dermatology_umap, "Dermatology Data (PCA)", "Dermatology Data (UMAP)")

# Visualize Clustering Results
clustering_visualization_side_by_side(student_pca, student_umap, student_kmeans_pca, student_kmeans_umap,
                                      "K-Means Clusters (Student Data, PCA)", "K-Means Clusters (Student Data, UMAP)")
clustering_visualization_side_by_side(dermatology_pca, dermatology_umap, dermatology_kmeans_pca, dermatology_kmeans_umap,
                                      "K-Means Clusters (Dermatology Data, PCA)", "K-Means Clusters (Dermatology Data, UMAP)")
clustering_visualization_side_by_side(student_pca, student_umap, student_hierarchical_pca, student_hierarchical_umap,
                                      "Hierarchical Clusters (Student Data, PCA)", "Hierarchical Clusters (Student Data, UMAP)")
clustering_visualization_side_by_side(dermatology_pca, dermatology_umap, dermatology_hierarchical_pca, dermatology_hierarchical_umap,
                                      "Hierarchical Clusters (Dermatology Data, PCA)", "Hierarchical Clusters (Dermatology Data, UMAP)")

# %%
### Ensemble method###
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# Function to preprocess the Dermatology dataset
def preprocess_dermatology_data(data):
    data.replace('?', pd.NA, inplace=True)  # Replace '?' with NaN
    data = data.dropna()  # Drop rows with missing values
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')  # Convert to numeric
    return data

# Function to align features and target
def align_features_and_target(features, target):
    return features.loc[target.index], target

# Function to tune, train, and evaluate models
def tune_and_train_models(X_train, X_test, y_train, y_test, dataset_name, dataset_type):
    print(f"\nTraining models on {dataset_name} ({dataset_type}) dataset...")
    
    # Hyperparameter grids
    param_grids = {
        "AdaBoost": {"n_estimators": [50, 100, 150], "learning_rate": [0.01, 0.1, 1]},
        "RandomForest": {"n_estimators": [50, 100, 150], "max_depth": [None, 10, 20], "min_samples_split": [2, 5, 10]},
        "GradientBoosting": {"n_estimators": [50, 100, 150], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 5, 7]},
    }
    
    results = {}

    for model_name, param_grid in param_grids.items():
        print(f"Tuning {model_name}...")
        if model_name == "AdaBoost":
            model = AdaBoostClassifier(random_state=42)
        elif model_name == "RandomForest":
            model = RandomForestClassifier(random_state=42)
        elif model_name == "GradientBoosting":
            model = GradientBoostingClassifier(random_state=42)
        
        grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)
        
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        
        print(f"Best Params for {model_name} on {dataset_name} ({dataset_type}): {grid.best_params_}")
        print(f"Accuracy for {model_name} on {dataset_name} ({dataset_type}): {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        results[model_name] = {
            "best_params": grid.best_params_,
            "accuracy": accuracy_score(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix - {model_name} ({dataset_name} {dataset_type})")
        plt.show()
        
    return results

# Load datasets
student_data = pd.read_csv('Student_performance_data.csv')
dermatology_data = pd.read_csv('dermatology_database_1.csv')

# Preprocess datasets
dermatology_data = preprocess_dermatology_data(dermatology_data)

# Define target variables
y_student = student_data["GradeClass"]  # Replace "GradeClass" with actual target column name
y_dermatology = dermatology_data["class"]  # Replace "class" with actual target column name

# Drop target columns
X_student = student_data.drop(columns=["GradeClass"])
X_dermatology = dermatology_data.drop(columns=["class"])

# Load PCA and UMAP datasets
student_pca = pd.read_csv("Student_Data_PCA_Reduced.csv")
dermatology_pca = pd.read_csv("Dermatology_Data_PCA_Reduced.csv")
student_umap = pd.read_csv("Student_Data_UMAP_Reduced.csv")
dermatology_umap = pd.read_csv("Dermatology_Data_UMAP_Reduced.csv")

# Align features and targets for reduced datasets
student_pca, y_student_pca = align_features_and_target(student_pca, y_student)
dermatology_pca, y_dermatology_pca = align_features_and_target(dermatology_pca, y_dermatology)
student_umap, y_student_umap = align_features_and_target(student_umap, y_student)
dermatology_umap, y_dermatology_umap = align_features_and_target(dermatology_umap, y_dermatology)

# Split datasets
datasets = {
    "Original": [
        train_test_split(X_student, y_student, test_size=0.3, random_state=42, stratify=y_student),
        train_test_split(X_dermatology, y_dermatology, test_size=0.3, random_state=42, stratify=y_dermatology),
    ],
    "PCA": [
        train_test_split(student_pca, y_student_pca, test_size=0.3, random_state=42, stratify=y_student_pca),
        train_test_split(dermatology_pca, y_dermatology_pca, test_size=0.3, random_state=42, stratify=y_dermatology_pca),
    ],
    "UMAP": [
        train_test_split(student_umap, y_student_umap, test_size=0.3, random_state=42, stratify=y_student_umap),
        train_test_split(dermatology_umap, y_dermatology_umap, test_size=0.3, random_state=42, stratify=y_dermatology_umap),
    ],
}

# Analyze for Original, PCA, and UMAP datasets
for dataset_type, [(X_train_student, X_test_student, y_train_student, y_test_student), (X_train_dermatology, X_test_dermatology, y_train_dermatology, y_test_dermatology)] in datasets.items():
    print(f"\nAnalysis for {dataset_type} Dataset:")
    print("\nStudent Dataset:")
    student_results = tune_and_train_models(X_train_student, X_test_student, y_train_student, y_test_student, "Student", dataset_type)
    print("\nDermatology Dataset:")
    dermatology_results = tune_and_train_models(X_train_dermatology, X_test_dermatology, y_train_dermatology, y_test_dermatology, "Dermatology", dataset_type)

# %%
import pandas as pd

try:
    student_pca_reduced = pd.read_csv('Student_Data_PCA_Reduced.csv')
    dermatology_pca_reduced = pd.read_csv('Dermatology_Data_PCA_Reduced.csv')

    # Load UMAP-reduced datasets
    student_umap_reduced = pd.read_csv('Student_Data_UMAP_Reduced.csv')
    dermatology_umap_reduced = pd.read_csv('Dermatology_Data_UMAP_Reduced.csv')

    # Display sample data
    print("Sample of PCA-reduced Student Dataset:")
    print(student_pca_reduced.head())

    print("\nSample of PCA-reduced Dermatology Dataset:")
    print(dermatology_pca_reduced.head())

    print("\nSample of UMAP-reduced Student Dataset:")
    print(student_umap_reduced.head())

    print("\nSample of UMAP-reduced Dermatology Dataset:")
    print(dermatology_umap_reduced.head())

except FileNotFoundError as e:
    print(f"File not found: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

# Ensure the indices of the PCA and UMAP datasets align with the original target
dermatology_pca_reduced, y_dermatology_pca = align_features_and_target(dermatology_pca_reduced, y_dermatology)
dermatology_umap_reduced, y_dermatology_umap = align_features_and_target(dermatology_umap_reduced, y_dermatology)

# Split datasets for Dermatology
datasets_dermatology = {
    "Original": train_test_split(X_dermatology, y_dermatology, test_size=0.3, random_state=42, stratify=y_dermatology),
    "PCA": train_test_split(dermatology_pca_reduced, y_dermatology_pca, test_size=0.3, random_state=42, stratify=y_dermatology_pca),
    "UMAP": train_test_split(dermatology_umap_reduced, y_dermatology_umap, test_size=0.3, random_state=42, stratify=y_dermatology_umap),
}

# Verify the shapes of datasets
print("Original Dataset Shapes:", X_dermatology.shape, y_dermatology.shape)
print("PCA-Reduced Dataset Shapes:", dermatology_pca_reduced.shape, y_dermatology_pca.shape)
print("UMAP-Reduced Dataset Shapes:", dermatology_umap_reduced.shape, y_dermatology_umap.shape)

assert len(dermatology_pca_reduced) == len(y_dermatology_pca), "PCA dataset and target do not match!"
assert len(dermatology_umap_reduced) == len(y_dermatology_umap), "UMAP dataset and target do not match!"
print("PCA-Reduced Target Distribution:", y_dermatology_pca.value_counts())
print("UMAP-Reduced Target Distribution:", y_dermatology_umap.value_counts())

# %%
#### Train test Error,Learning curve and ROC curve #####

import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score, roc_curve, auc, mean_squared_error
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from itertools import cycle

# Function to preprocess Dermatology Dataset
def preprocess_dermatology_data(data):
    data.replace('?', np.nan, inplace=True)  # Replace '?' with NaN
    data = data.apply(pd.to_numeric, errors='coerce')  # Convert columns to numeric
    data.fillna(data.mean(), inplace=True)  # Impute missing values with column mean
    return data

# Function to plot Learning Curve
def plot_learning_curve(model, X, y, title):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.figure()
    plt.plot(train_sizes, train_mean, 'o-', label="Training Accuracy")
    plt.plot(train_sizes, test_mean, 'o-', label="Validation Accuracy")
    plt.title(f"Learning Curve - {title}")
    plt.xlabel("Training Samples")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.show()

# Function to plot Multiclass ROC Curve
def plot_roc_curve_multiclass(model, X_test, y_test, title):
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    n_classes = y_test_bin.shape[1]
    if n_classes < 2:
        print(f"Cannot plot ROC curve for {title}: Less than 2 classes.")
        return

    y_score = model.predict_proba(X_test)
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f"Class {i} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.title(f"ROC Curve - {title}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

# Function to calculate Train/Test Errors
def calculate_train_test_error(model, X_train, X_test, y_train, y_test, model_name, dataset_name):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_error = mean_squared_error(y_train, y_train_pred)
    test_error = mean_squared_error(y_test, y_test_pred)

    print(f"{model_name} Train Error on {dataset_name}: {train_error:.4f}")
    print(f"{model_name} Test Error on {dataset_name}: {test_error:.4f}")

# Function to train and evaluate models
def train_and_evaluate(X_train, X_test, y_train, y_test, dataset_name, dataset_type):
    models = {
        "AdaBoost": AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    }

    for model_name, model in models.items():
        print(f"\nTraining {model_name} on {dataset_name} ({dataset_type}) dataset...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"{model_name} Accuracy: {accuracy:.4f}")
        calculate_train_test_error(model, X_train, X_test, y_train, y_test, model_name, f"{dataset_name} - {dataset_type}")
        plot_learning_curve(model, X_train, y_train, f"{model_name} ({dataset_name} - {dataset_type})")
        
        if hasattr(model, "predict_proba"):
            plot_roc_curve_multiclass(model, X_test, y_test, f"{model_name} ({dataset_name} - {dataset_type})")

# Load and preprocess datasets
student_data = pd.read_csv("Student_performance_data.csv")
dermatology_data = pd.read_csv("dermatology_database_1.csv")
dermatology_data = preprocess_dermatology_data(dermatology_data)

# Original datasets
X_student = student_data.drop(columns=["GradeClass"])
y_student = student_data["GradeClass"]
X_dermatology = dermatology_data.drop(columns=["class"])
y_dermatology = dermatology_data["class"]

# PCA and UMAP datasets
student_pca = pd.read_csv("Student_Data_PCA_Reduced.csv")
dermatology_pca = pd.read_csv("Dermatology_Data_PCA_Reduced.csv")
student_umap = pd.read_csv("Student_Data_UMAP_Reduced.csv")
dermatology_umap = pd.read_csv("Dermatology_Data_UMAP_Reduced.csv")

# Splitting datasets
datasets_student = {
    "Original": train_test_split(X_student, y_student, test_size=0.3, random_state=42, stratify=y_student),
    "PCA": train_test_split(student_pca, y_student, test_size=0.3, random_state=42, stratify=y_student),
    "UMAP": train_test_split(student_umap, y_student, test_size=0.3, random_state=42, stratify=y_student)
}

datasets_dermatology = {
    "Original": train_test_split(X_dermatology, y_dermatology, test_size=0.3, random_state=42, stratify=y_dermatology),
    "PCA": train_test_split(dermatology_pca, y_dermatology, test_size=0.3, random_state=42, stratify=y_dermatology),
    "UMAP": train_test_split(dermatology_umap, y_dermatology, test_size=0.3, random_state=42, stratify=y_dermatology)
}

# Analyze datasets
for dataset_type, (X_train, X_test, y_train, y_test) in datasets_student.items():
    print(f"\n--- Student Dataset ({dataset_type}) ---")
    train_and_evaluate(X_train, X_test, y_train, y_test, "Student", dataset_type)

for dataset_type, (X_train, X_test, y_train, y_test) in datasets_dermatology.items():
    print(f"\n--- Dermatology Dataset ({dataset_type}) ---")
    train_and_evaluate(X_train, X_test, y_train, y_test, "Dermatology", dataset_type)

# %%
####  visualize accuracy and test errors ###
def plot_model_comparison(results, dataset_name):
    models = ["AdaBoost", "RandomForest", "GradientBoosting"]
    metrics = ["Original", "PCA", "UMAP"]

    accuracies = {model: [] for model in models}
    test_errors = {model: [] for model in models}

    # Extract results
    for metric in metrics:
        for model in models:
            accuracies[model].append(results[metric][model]['accuracy'])
            test_errors[model].append(results[metric][model]['test_error'])

    # Plot side-by-side bar charts
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accuracy Plot
    for idx, model in enumerate(models):
        ax[0].bar(x + idx * width, accuracies[model], width, label=model)
    ax[0].set_title(f"{dataset_name} - Accuracy Comparison")
    ax[0].set_xticks(x + width)
    ax[0].set_xticklabels(metrics)
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()
    ax[0].grid()

    # Test Error Plot
    for idx, model in enumerate(models):
        ax[1].bar(x + idx * width, test_errors[model], width, label=model)
    ax[1].set_title(f"{dataset_name} - Test Error Comparison")
    ax[1].set_xticks(x + width)
    ax[1].set_xticklabels(metrics)
    ax[1].set_ylabel("Test Error")
    ax[1].legend()
    ax[1].grid()

    plt.tight_layout()
    plt.show()

# Example Usage
student_results = {
    "Original": {"AdaBoost": {"accuracy": 0.9331, "test_error": 0.3942},
                 "RandomForest": {"accuracy": 0.9248, "test_error": 0.3593},
                 "GradientBoosting": {"accuracy": 0.9345, "test_error": 0.3593}},
    "PCA": {"AdaBoost": {"accuracy": 0.8524, "test_error": 0.2855},
            "RandomForest": {"accuracy": 0.9304, "test_error": 0.2145},
            "GradientBoosting": {"accuracy": 0.9248, "test_error": 0.2047}},
    "UMAP": {"AdaBoost": {"accuracy": 0.5362, "test_error": 2.1964},
             "RandomForest": {"accuracy": 0.7340, "test_error": 0.4749},
             "GradientBoosting": {"accuracy": 0.6783, "test_error": 0.7256}}
}

dermatology_results = {
    "Original": {"AdaBoost": {"accuracy": 0.8818, "test_error": 0.5545},
                 "RandomForest": {"accuracy": 0.9727, "test_error": 0.1091},
                 "GradientBoosting": {"accuracy": 0.9182, "test_error": 0.3818}},
    "PCA": {"AdaBoost": {"accuracy": 0.9364, "test_error": 0.2273},
            "RandomForest": {"accuracy": 0.9273, "test_error": 0.2909},
            "GradientBoosting": {"accuracy": 0.9273, "test_error": 0.3818}},
    "UMAP": {"AdaBoost": {"accuracy": 0.9273, "test_error": 0.2364},
             "RandomForest": {"accuracy": 0.9273, "test_error": 0.2364},
             "GradientBoosting": {"accuracy": 0.9364, "test_error": 0.2000}}
}

# Generate plots
plot_model_comparison(student_results, "Student Dataset")
plot_model_comparison(dermatology_results, "Dermatology Dataset")



# %%
print_messages()


