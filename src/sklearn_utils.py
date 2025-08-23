# =========================================================================== #
# >>> Module of functions and classes preprocessing data train and data test. #                                        
# =========================================================================== #

# ======================================================== #
# Imports:                                                 #
# ======================================================== #
# Scikit-Learn Preprocessing / Metrics
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
# Scikit-Learn Models
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# ======================================================== #
# Scikit-Learn - Class                                     #
# ======================================================== #
class SKLearn:

    # Inicialize Class
    def __int__(
        self,
        name: str = 'SKLearn'
    ):
        self.name = name

    # ======================================================== #
    # Scikit-Learn Models - Function/Pipeline                  #
    # ======================================================== #
    def SKModels(
        self,
    ):
        """
        Create and return a list of machine learning models wrapped in sklearn Pipelines.

        Returns
        -------
        list of tuple
            A list where each element is a tuple containing:
            - model name (str)
            - sklearn Pipeline object
        """
        try:

            models = [
                (
                    'Logistic Regression',
                    Pipeline([
                        ('model', LogisticRegression(
                            random_state = 33,
                            class_weight = 'balanced',
                        )),
                    ]),
                ),
                (
                    'Decision Tree Classifier',
                    Pipeline([
                        ('model', DecisionTreeClassifier(
                            random_state = 33,
                            class_weight = 'balanced',
                            max_depth = 5,
                            criterion = 'gini',
                            min_impurity_decrease = 0.001,
                        )),
                    ]),
                ),
                (
                    'Random Forest Classifier',
                    Pipeline([
                        ('model', RandomForestClassifier(
                            random_state = 33,
                            class_weight = 'balanced',
                            n_estimators = 100,
                            max_depth = 10,
                            min_samples_split = 2,
                        )),
                    ]),
                ),
                (
                    'KNeighbors Classifier',
                    Pipeline([
                        ('model', KNeighborsClassifier(
                            n_neighbors = 5,
                            weights = 'distance',
                            metric = 'minkowski',
                        )),
                    ]),
                ),
                (
                    'Support Vector Machine Classifier',
                    Pipeline([
                        ('model', SVC(
                            random_state = 33,
                            class_weight = 'balanced',
                            C = 1.0,
                            kernel = 'rbf',
                            gamma = 'scale',
                            probability = True,
                        )),
                    ]),
                ),
                (
                    'Gradient Boosting Classifier',
                    Pipeline([
                        ('model', GradientBoostingClassifier(
                            random_state = 33,
                            n_estimators = 200,
                            max_depth = 3,
                            learning_rate = 0.1,
                            subsample = 0.7,
                        )),
                    ]),
                ),
            ]

            return models

        except Exception as e:
            print(f'[ERROR] Failed to create models: {str(e)}')


    # ======================================================== #
    # Cross Validation ML - Function                           #
    # ======================================================== #
    def CrossValidationML(
        self,
        models,
        x_train,
        y_train,
        n_splits: int = 5,
        random_state: int = 33,
        shuffle: bool = True,
        scoring: str = 'roc_auc',  
    ):
        """
        Performs stratified K-fold cross-validation on a list of machine learning models.

        This function receives multiple models (e.g., pipelines), trains and evaluates them
        using Stratified K-Fold cross-validation, and prints the mean and standard deviation 
        of the chosen scoring metric for each model.

        Args:
            models (list): A list of tuples with (model_name, model_pipeline), where each 
                model_pipeline follows the scikit-learn API (i.e., implements .fit and .predict).
            x_train (array-like): Feature set used for training and validation.
            y_train (array-like): Target labels corresponding to x_train.
            n_splits (int, optional): Number of folds for cross-validation. Defaults to 5.
            random_state (int, optional): Random seed for reproducibility. Defaults to 33.
            shuffle (bool, optional): Whether to shuffle the data before splitting. Defaults to True.
            scoring (str, optional): Scoring metric to use (e.g., 'roc_auc', 'accuracy', 'f1'). 
                Defaults to 'roc_auc'.

        Prints:
            For each model:
                - Name of the model
                - Mean cross-validation score
                - Standard deviation of the score across folds

        Returns:
            None
        """
        try:
            
            # Cross Validation
            cv = StratifiedKFold(n_splits = n_splits, random_state = random_state, shuffle = shuffle)

            # Metrics Cross validation
            results, names = [], []

            for name, pipeline in models:

                cv_results = cross_val_score(
                    pipeline,
                    x_train, 
                    y_train,
                    cv = cv,
                    scoring = scoring

                )
                results.append(cv_results)
                names.append(name)

                print('#' * 30)
                print(f'✅ {name}: ')
                print(f'🎯 AUC ROC Mean: {cv_results.mean():.6f} ')
                print(f'🟣 STD Metrics: {cv_results.std():.6f}\n')

        except Exception as e:
            print(f'[ERROR] Failed to run ml cross validation {str(e)}')