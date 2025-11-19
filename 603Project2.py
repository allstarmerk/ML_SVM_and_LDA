import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif #not using chi square for task 1(i)
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# DATA LOADING AND PREPROCESSING
def preprocess_data(filepath):

    #Load the pollution dataset and preprocess it.
    df = pd.read_csv(filepath)
    
    # Separate features and labels
    X = df.iloc[:, :9].values
    y = df.iloc[:, -1].values
    
    #negative values to zero
    X[X < 0] = 0
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    feature_names = df.columns[:9].tolist()
    
    print(f"Dataset shape: {X.shape}")
    print(f"Classes: {le.classes_}")
    print(f"Feature names: {feature_names}\n")
    
    return X, y_encoded, feature_names, le

# TASK 1
def univariate_selection(X, y, feature_names, k_values=[1, 2, 3]): #Perform univariate feature selection using ANOVA
   
    results = {}
    print("\n 1.) Univariate Feature Selection (ANOVA F-statistic):")
    print("-" * 60)
    
    for k in k_values:
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X, y)
        selected_indices = selector.get_support(indices=True)
        selected_features = [feature_names[i] for i in selected_indices]
        
        results[k] = {
            'features': selected_features,
            'indices': selected_indices,
            'selector': selector
        }
        
        print(f"Kf = {k}: {selected_features}")
    
    return results


def perform_rf_importance_selection(X, y, feature_names, k_values=[1, 2, 3], random_state=42): #Perform feature selection using Random Forest feature importance.
   
    results = {}
    print("\n 2.) Feature Importance Scores (Random Forest):")
    
    
    # Train Random Forest and get importances
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    rf.fit(X, y)
    feature_importances = rf.feature_importances_
    importance_indices = np.argsort(feature_importances)[::-1]
    
    for k in k_values:
        selected_indices = importance_indices[:k]
        selected_features = [feature_names[i] for i in selected_indices]
        importances = [feature_importances[i] for i in selected_indices]
        
        results[k] = {
            'features': selected_features,
            'indices': selected_indices,
            'importances': importances
        }
        
        print(f"Kf = {k}: {selected_features}")
        print(f"         Importances: {[f'{imp:.4f}' for imp in importances]}")
    
    return results


def task1_feature_selection(X, y, feature_names): #Executes Task 1: 
    
    print("\n")
    print("TASK 1: FEATURE SELECTION")
    print("\n")
    
    univariate_results = univariate_selection(X, y, feature_names)
    rf_results = perform_rf_importance_selection(X, y, feature_names)
    
    return univariate_results, rf_results

# TASK 2

def perform_pca(X, k_values=[1, 2, 3]): #Perform Principal Component Analysis.
    
    results = {}
    print("\n 1.) PCA:")
    print("-" * 60)
    
    for k in k_values:
        pca = PCA(n_components=k)
        X_transformed = pca.fit_transform(X)
        explained_var = np.sum(pca.explained_variance_ratio_)
        
        results[k] = {
            'X_transformed': X_transformed,
            'pca': pca,
            'explained_variance': explained_var
        }
        
        print(f"Ks = {k}: Shape = {X_transformed.shape}, "
              f"Explained Variance = {explained_var:.4f}")
    
    return results


def perform_lda(X, y, k_values=[1, 2, 3]): #Perform Linear Discriminant Analysis.

    results = {}
    print("\n 2.) LDA:")
    print("-" * 12)
    
    for k in k_values:
        lda = LDA(n_components=k)
        X_transformed = lda.fit_transform(X, y)
        
        results[k] = {
            'X_transformed': X_transformed,
            'lda': lda
        }
        
        print(f"Ks = {k}: Shape = {X_transformed.shape}")
    
    return results


def task2_dimensionality_reduction(X, y): #Executes Task 2: PCA and LDA
    print("\n")
    print("TASK 2: PCA AND LDA")
    print("\n")
    
    pca_results = perform_pca(X)
    lda_results = perform_lda(X, y)
    
    return pca_results, lda_results


# CLASSIFICATION HELPERS Functions
def evaluate_classifier(X, y, classifier, n_iterations=50, test_size=0.25): #Evaluates classifier using multiple random train test splits.
    
    scores = []
    
    for i in range(n_iterations):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=i
        )
        
        classifier.fit(X_train, y_train)
        scores.append(classifier.score(X_test, y_test))
    
    return np.mean(scores)


def classify_with_methods(X, y, method_name, n_iterations=50): #Classifys using both SVM and Naive Bayes.
    
    # SVM
    svm = SVC(kernel='rbf', random_state=42)
    svm_acc = evaluate_classifier(X, y, svm, n_iterations)
    
    # Gaussian Naive Bayes
    nb = GaussianNB()
    nb_acc = evaluate_classifier(X, y, nb, n_iterations)
    
    return {'SVM': svm_acc, 'NaiveBayes': nb_acc}


# TASK 3
def task3_classify_feature_selection(X, y, univariate_results, rf_results, #Executes Task 3
                                     k_values=[1, 2, 3], n_iterations=50):

    print("\n")
    print("TASK 3: CLASSIFICATION WITH FEATURE SELECTION")
    print("\n")
    print(f"\nRunning {n_iterations} iterations for each configuration")
    
    results = {
        'univariate': {'SVM': {}, 'NaiveBayes': {}},
        'rf_importance': {'SVM': {}, 'NaiveBayes': {}}
    }
    
    for k in k_values:
        print(f"\n--- Kf = {k} ---")
        
        # Get selected features for univariate method
        selector = SelectKBest(score_func=f_classif, k=k)
        X_univariate = selector.fit_transform(X, y)
        
        # Get selected features for RF importance method
        X_rf = X[:, rf_results[k]['indices']]
        
        # Classify with univariate selection
        uni_scores = classify_with_methods(X_univariate, y, "Univariate", n_iterations)
        results['univariate']['SVM'][k] = uni_scores['SVM']
        results['univariate']['NaiveBayes'][k] = uni_scores['NaiveBayes']
        
        # Classify with RF importance selection
        rf_scores = classify_with_methods(X_rf, y, "RF Importance", n_iterations)
        results['rf_importance']['SVM'][k] = rf_scores['SVM']
        results['rf_importance']['NaiveBayes'][k] = rf_scores['NaiveBayes']
        
        print(f"  Univariate - SVM: {uni_scores['SVM']:.4f}, "
              f"NB: {uni_scores['NaiveBayes']:.4f}")
        print(f"  RF Import  - SVM: {rf_scores['SVM']:.4f}, "
              f"NB: {rf_scores['NaiveBayes']:.4f}")
    
    return results

# TASK 4: CLASSIFICATION WITH PCA AND LDA
def task4_classify_dimensionality_reduction(X, y, pca_results, lda_results,      #Executes Task 4
                                           k_values=[1, 2, 3], n_iterations=50):
    print("\n" + "="*80)
    print("TASK 4: CLASSIFICATION WITH PCA AND LDA")
    print("="*80)
    print(f"\nRunning {n_iterations} iterations for each configuration...")
    
    results = {
        'PCA': {'SVM': {}, 'NaiveBayes': {}},
        'LDA': {'SVM': {}, 'NaiveBayes': {}}
    }
    
    for k in k_values:
        print(f"\n--- Ks = {k} ---")
        
        # Get transformed data
        X_pca = pca_results[k]['X_transformed']
        X_lda = lda_results[k]['X_transformed']
        
        # Classify with PCA
        pca_scores = classify_with_methods(X_pca, y, "PCA", n_iterations)
        results['PCA']['SVM'][k] = pca_scores['SVM']
        results['PCA']['NaiveBayes'][k] = pca_scores['NaiveBayes']
        
        # Classify with LDA
        lda_scores = classify_with_methods(X_lda, y, "LDA", n_iterations)
        results['LDA']['SVM'][k] = lda_scores['SVM']
        results['LDA']['NaiveBayes'][k] = lda_scores['NaiveBayes']
        
        print(f"  PCA - SVM: {pca_scores['SVM']:.4f}, "
              f"NB: {pca_scores['NaiveBayes']:.4f}")
        print(f"  LDA - SVM: {lda_scores['SVM']:.4f}, "
              f"NB: {lda_scores['NaiveBayes']:.4f}")
    
    return results

# RESULTS
def print_summary(results_task3, results_task4): #Print summary.
    print("\n")
    print("SUMMARY OF ALL RESULTS")
    print("="*12)
    
    # Task 3 Summary
    print("\nTask 3 - Feature Selection Results:")
    print("-" * 12)
    print("Method          Classifier    Kf=1    Kf=2    Kf=3")
    print("-" * 12)
    for method in ['univariate', 'rf_importance']:
        method_name = "Univariate" if method == 'univariate' else "RF Importance"
        for clf in ['SVM', 'NaiveBayes']:
            clf_name = "Naive Bayes" if clf == 'NaiveBayes' else clf
            scores = [results_task3[method][clf][k] for k in [1, 2, 3]]
            print(f"{method_name:15} {clf_name:12}  "
                  f"{scores[0]:.4f}  {scores[1]:.4f}  {scores[2]:.4f}")
    
    # Task 4 Summary
    print("\nTask 4 - Dimensionality Reduction Results:")
    print("-" * 60)
    print("Method    Classifier    Ks=1    Ks=2    Ks=3")
    print("-" * 60)
    for method in ['PCA', 'LDA']:
        for clf in ['SVM', 'NaiveBayes']:
            clf_name = "Naive Bayes" if clf == 'NaiveBayes' else clf
            scores = [results_task4[method][clf][k] for k in [1, 2, 3]]
            print(f"{method:8}  {clf_name:12}  "
                  f"{scores[0]:.4f}  {scores[1]:.4f}  {scores[2]:.4f}")


def analyze_results(results_task3, results_task4):
    print("\n" + "="*80)
    print("TASK 5: summary")
    print("="*12)
    
      #results aggregation
    all_results = []
    
    for method in ['univariate', 'rf_importance']:
        for clf in ['SVM', 'NaiveBayes']:
            for k in [1, 2, 3]:
                all_results.append((
                    f"FS-{method}-{clf}", 
                    k, 
                    results_task3[method][clf][k]
                ))
    
    for method in ['PCA', 'LDA']:
        for clf in ['SVM', 'NaiveBayes']:
            for k in [1, 2, 3]:
                all_results.append((
                    f"DR-{method}-{clf}", 
                    k, 
                    results_task4[method][clf][k]
                ))
    
    # Sort by accuracy
    all_results.sort(key=lambda x: x[2], reverse=True)
    
    print("\nTop 5 Configurations by Accuracy:")
    print("-" * 60)
    for i, (config, k, acc) in enumerate(all_results[:5], 1):
        print(f"{i}. {config:30} K={k}  Accuracy: {acc:.4f}")



def main():
   
    # Load and preprocess data
    X, y_encoded, feature_names, le = preprocess_data('pollution_dataset.csv')
    
    # Task 1: Feature Selection
    univariate_results, rf_results = task1_feature_selection(X, y_encoded, feature_names)
    
    # Task 2: PCA and LDA
    pca_results, lda_results = task2_dimensionality_reduction(X, y_encoded)
    
    # Task 3: Classification with Feature Selection
    results_task3 = task3_classify_feature_selection(
        X, y_encoded, univariate_results, rf_results
    )
    
    # Task 4: Classification with PCA and LDA
    results_task4 = task4_classify_dimensionality_reduction(
        X, y_encoded, pca_results, lda_results
    )
    
    # Print summary
    print_summary(results_task3, results_task4)
    analyze_results(results_task3, results_task4)
    
    print("\n")
    print("Finished running")
    print("=")
    
    return {
        'univariate': univariate_results,
        'rf': rf_results,
        'pca': pca_results,
        'lda': lda_results,
        'task3': results_task3,
        'task4': results_task4
    }


if __name__ == "__main__":
    results = main()