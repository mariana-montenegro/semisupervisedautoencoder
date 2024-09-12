# Internal
from sklearn.calibration import cross_val_predict
from sklearn.metrics import f1_score, roc_auc_score
from dataloader.dataloader import DataLoader
from models_ML import feature_selection, apply_pca, plot_pca_scatter, plot_decision_regions

# External
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# Define best parameters from grid search
best_params = {
    'KNN': {'n_neighbors': 6},
    'RF': {'max_depth': 1, 'n_estimators': 15},
    'AB': {'n_estimators': 10, 'learning_rate': 1},
    'SVM': {'C': 1000, 'gamma': 1e-5}
}

# Define classifiers with best parameters
classifiers = {
    'KNN': KNeighborsClassifier(**best_params['KNN']),
    'RF': RandomForestClassifier(**best_params['RF']),
    'AB': AdaBoostClassifier(**best_params['AB']),
    'SVM': SVC(kernel='rbf', **best_params['SVM'], probability=True)
}

# Define alphas for Lasso feature selection
lasso_alphas = [0.001, 0.003, 0.005]

# Define k values for Pearson correlation feature selection
pearson_k_values = [3, 5, 10]

# Define k values for ANOVA feature selection
anova_k_values = [50, 100, 200]

# MACHINE LEARNING - BINARY APPROACH
def run_train_model(with_feature_selection=False, classifier_name='SVM', fs_method='lasso', alpha_or_k=None):
    print(f'\nRunning model {classifier_name} with{"out" if not with_feature_selection else ""} feature selection (method: {fs_method}, value: {alpha_or_k})...')
    
    print('\nLoading data...')
    x, y = DataLoader().load()
    
    print('\nPre-processing data...')
    x_scaled = DataLoader.pre_process_data(x)
    
    if with_feature_selection:
        print('\nFeature selection...')
        if fs_method == 'lasso':
            x_selected_list = feature_selection(x_scaled, y, fs_method='lasso', lasso_alphas=[alpha_or_k])
        elif fs_method == 'pearson':
            x_selected_list = feature_selection(x_scaled, y, fs_method='pearson', k_values=[alpha_or_k])
        elif fs_method == 'anova':
            x_selected_list = feature_selection(x_scaled, y, fs_method='anova', k_values=[alpha_or_k])
        x_selected = x_selected_list[0]
    else:
        x_selected = x_scaled

    print('\nSplitting data into training and test sets...')
    x_train, x_test, y_train, y_test = train_test_split(x_selected, y, test_size=0.3, random_state=42, stratify=y)  

    # Apply PCA for visualization
    print('\nApplying PCA for visualization...')
    x_pca_train = apply_pca(x_train)
    x_pca_test = apply_pca(x_test)

    # Plotting PCA scatter plot
    print('\nPlotting PCA scatter plot...')
    plot_pca_scatter(x_pca_train, y_train)

    print(f'\nTraining {classifier_name} classifier with 5-fold cross-validation...')
    classifier = classifiers[classifier_name]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_accuracies = cross_val_score(classifier, x_pca_train, y_train, cv=cv, scoring='f1')
    for fold_idx, train_accuracy in enumerate(train_accuracies, 1):
        print(f'Fold {fold_idx}: Training Accuracy = {train_accuracy:.2%}')
    avg_train_accuracy = np.mean(train_accuracies)
    print(f'\nAverage Training Accuracy across 5 folds: {avg_train_accuracy:.2%}')
    
    # Fit the final model on the entire PCA-transformed training set
    classifier.fit(x_pca_train, y_train)

    # Evaluate model on test set
    print('\nEvaluating model on test set...')
    y_pred = classifier.predict(x_pca_test)
    y_pred_prob = classifier.predict_proba(x_pca_test)[:, 1]

    # Evaluate model
    f1 = f1_score(y_test, y_pred)
    print(f'F1 Score on Test Set: {f1:.2f}')
    auc = roc_auc_score(y_test, y_pred_prob)
    print(f'AUC on Test Set: {auc:.2f}')

    # Plot decision boundary
    plot_decision_regions(x_pca_train, y_train, classifier=classifier, title=f'Decision Regions for non-linear SVM ({fs_method} - k:{alpha_or_k})')

def run():    
    classifiers_to_test = ['SVM']
    for clf_name in classifiers_to_test:
        run_train_model(with_feature_selection=True, classifier_name=clf_name, fs_method='anova', alpha_or_k=50)

if __name__ == '__main__':
    run()
