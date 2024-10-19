import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

class DecisionTree:
    '''
    A site-specific decision tree model using sklearn's Pipeline
    '''
    def __init__(self, site, cat_features, num_features, target, df) -> None:
        super().__init__()
        self.site = site
        self.cat_features = cat_features
        self.num_features = num_features
        self.target = target
        
        # Filter data by site
        self.data = df[df['site'] == self.site]

    def transform_split_train_score_eval(self):
        '''
        - Transforms features using pipelines
        - Splits into training and test set
        - Fits model using training data
        - Makes predictions on test set
        - Evaluates model performance
        '''
        # Define X and y
        X_site = self.data[self.cat_features + self.num_features]
        y_site = self.data[self.target]
        
        # Define the column transformer for both numerical and categorical features
        preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value=0)),  # Replace missing values with 0
                ('scaler', StandardScaler())  # Standardize numerical features
            ]), self.num_features),
            ('cat', OrdinalEncoder(), self.cat_features)  # Encode categorical features
        ]
        )

        # Create the pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', DecisionTreeRegressor(max_depth=5))  # Set the depth
        ])
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_site, y_site, test_size=0.25, random_state=42)
        
        # Fit the pipeline to training data
        pipeline.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)

        # Evaluate model performance
        self.evaluate_model(y_train, y_pred_train, y_test, y_pred_test)
        
        # Access and display feature importances
        feature_importances_ = pipeline.named_steps['regressor'].feature_importances_

        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()

        # Combine feature names with their importances
        feature_importance_dict = dict(zip(feature_names, feature_importances_))

        # Sort by importance
        self.sorted_feature_importances = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

        plt.figure(figsize=(12, 6))
        sns.barplot(x=[importance for name, importance in self.sorted_feature_importances],
                    y=[name for name, importance in self.sorted_feature_importances],
                    palette='viridis')
        plt.title('Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.show()

    def evaluate_model(self, y_train, y_pred_train, y_test, y_pred_test):
        '''Evaluates the model and prints the results.'''
        # Mean Absolute Error (MAE)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        mae_train = mean_absolute_error(y_train, y_pred_train)

        # Root Mean Squared Error (RMSE)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))

        # R-squared (R²)
        r2_test = r2_score(y_test, y_pred_test)
        r2_train = r2_score(y_train, y_pred_train)

        # Print the evaluation metrics
        print(f"Mean Absolute Error (MAE) Test: {mae_test}")
        print(f"Mean Absolute Error (MAE) Train: {mae_train}")
        print(f"Root Mean Squared Error (RMSE) Test: {rmse_test}")
        print(f"Root Mean Squared Error (RMSE) Train: {rmse_train}")
        print(f"R-squared (R²) Test: {r2_test}")
        print(f"R-squared (R²) Train: {r2_train}")

        # MAPE Calculation
        mean_flow = np.mean(self.data[self.target])
        print(f'Mean Absolute Percentage Error (MAPE) Test: {100 * mae_test / mean_flow:.1f}%')
        print(f'Mean Absolute Percentage Error (MAPE) Train: {100 * mae_train / mean_flow:.1f}%')


