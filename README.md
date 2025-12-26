# Ensemble-Methods-and-Unsupervised-Learning-
Ensemble Classification with PCA/UMAP and Clustering: Student Performance &amp; Dermatology Datasets”
# Thesis / Project: Ensemble Methods + Unsupervised Learning  
**Datasets:** Student Performance & Dermatology (UCI)  
**Author:** Tarannum Mithila  

## Overview  
This repository contains my end-to-end machine learning workflow using two real-world datasets: **Student_performance_data.csv** and **dermatology_database_1.csv**.  
The project combines **Exploratory Data Analysis (EDA)**, **clustering (K-Means and Hierarchical)**, **dimensionality reduction (PCA and UMAP)**, and **ensemble classification models** to compare performance across original and reduced feature spaces.

## Key Objectives  
- Perform EDA to understand dataset structure, missing values, outliers, and correlations  
- Apply **K-Means** and **Hierarchical Clustering** to identify natural groupings in data  
- Compare **PCA vs UMAP** for dimensionality reduction and visualization  
- Train and tune ensemble models:
  - **AdaBoost**
  - **Random Forest**
  - **Gradient Boosting**
- Evaluate models using:
  - Accuracy and classification report  
  - Confusion matrix  
  - Learning curves  
  - Train vs test error  
  - Multiclass ROC curves (when applicable)

## Workflow Summary  
1. **EDA**
   - Dataset info, summary statistics, missing-value checks  
   - Boxplots for outliers  
   - Correlation heatmaps  
2. **Clustering (Original Data)**
   - Elbow method to select k  
   - Silhouette score evaluation  
   - Dendrogram visualization (Ward linkage)  
3. **Dimensionality Reduction**
   - PCA (fixed components + optimized PCA to retain 95% variance)  
   - UMAP (nonlinear 2D projection)  
   - Hybrid **PCA + UMAP** for noise reduction + nonlinear separation  
4. **Clustering (Reduced Data)**
   - K-Means and Hierarchical clustering on PCA- and UMAP-reduced datasets  
5. **Ensemble Classification**
   - GridSearchCV tuning for each model and dataset type (Original / PCA / UMAP)  
   - Evaluation via confusion matrices, learning curves, ROC curves, and error analysis  
6. **Result Comparison**
   - Bar charts comparing accuracy and test error across dataset types  

## Files  
- `Student_performance_data.csv` — original student dataset  
- `dermatology_database_1.csv` — dermatology dataset (age cleaned + missing handled)  
- `Student_Data_PCA_Reduced.csv` — PCA reduced student dataset  
- `Dermatology_Data_PCA_Reduced.csv` — PCA reduced dermatology dataset  
- `Student_Data_UMAP_Reduced.csv` — UMAP reduced student dataset  
- `Dermatology_Data_UMAP_Reduced.csv` — UMAP reduced dermatology dataset  

## How to Run  
1. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn umap-learn scipy
