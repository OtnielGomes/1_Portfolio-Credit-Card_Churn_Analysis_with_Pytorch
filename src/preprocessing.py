############################################################################################################
### >>> Module of functions and classes preprocessing data train and data test.                          ###
############################################################################################################

# Imports:
# Scikit-Learn Preprocessing
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

############################################################################################################

### Preprocessing Data Class ###
class PreprocessingData:

    ### Inicialize atributes ###
    def __init__(
        self,
        name: str = 'PreprocessingData',
    ):
        self.name = name
    
    ### ML Classic Preprocessing Function/Pipeline ###
    def MLClassicPreprocessing(
        self,
    ):
        try:

            # Lists of type features:

            # Standard Scaler Features
            std_scaler_features = [
                'credit_limit', 'total_amt_chng_q4_q1', 'total_ct_chng_q4_q1', 
                'avg_utilization_ratio', 'customer_age','dependent_count', 'months_on_book', 
                'total_relationship_count', 'months_inactive_12_mon', 'contacts_count_12_mon', 
                'total_revolving_bal', 'total_trans_amt', 'total_trans_ct'
            ]
            
            # One Hot Encoder Features
            one_hot_features = ['gender', 'marital_status']

            # Ordinal Features
            ordinal_features = ['education_level', 'income_category', 'card_category']

            # Create Ordinal pipeline
            # Create Ordinal order
            ordinal_features_order = {
            'education_level': ['Unknown', 'Uneducated', 'High School', 'College', 'Graduate', 'Post-Graduate', 'Doctorate'],
            'income_category': ['Unknown', 'Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +'],
            'card_category': ['Blue', 'Silver', 'Gold', 'Platinum']
            }
            ordinal_pipeline = Pipeline(
                steps = [
                    (
                        'ordinal_encoder', 
                        OrdinalEncoder(
                            categories = [
                                ordinal_features_order['education_level'], 
                                ordinal_features_order['income_category'],
                                ordinal_features_order['card_category']
                            ]
                        ),
                    ),
                    (
                        'standard_scaler', 
                        StandardScaler()
                    ),
                ]   
            )

            ml_preprocessor = ColumnTransformer(
                transformers = [
                    ('one_hot_features', OneHotEncoder(), one_hot_features),
                    ('ordinal_features', ordinal_pipeline, ordinal_features),
                    ('std_scaler_features', StandardScaler(), std_scaler_features),
                    
                ], 
                remainder = 'passthrough'
            )
            
            return ml_preprocessor
        
        except Exception as e:
            print(f'[ERROR] Failed to create pipeline with preprocessed data: {str(e)}.')
    
    ### Pytorch Neural Networks Preprocessing Function/Pipeline ###
    def NeuralNetWorkPreprocessing(
        self,
    ):
        try:

            # Lists of type features:

            # MixMax features
            minmax_scaler_features = [
                'months_on_book', 'customer_age', 'dependent_count',
                'total_relationship_count', 'months_inactive_12_mon',
                'contacts_count_12_mon', 'total_revolving_bal', 'avg_utilization_ratio', 
                'total_amt_chng_q4_q1', 'total_ct_chng_q4_q1', 'total_trans_ct', 
            ]

            # Robust features
            robust_scaler_features = ['credit_limit', 'total_trans_amt']

            # Nominal features
            nominal_features = ['gender', 'marital_status']

            # Ordinal Features
            ordinal_features = ['education_level', 'income_category', 'card_category']

            # Create Ordinal pipeline
            # Create Ordinal order
            ordinal_features_order = {
            'education_level': ['Unknown', 'Uneducated', 'High School', 'College', 'Graduate', 'Post-Graduate', 'Doctorate'],
            'income_category': ['Unknown', 'Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +'],
            'card_category': ['Blue', 'Silver', 'Gold', 'Platinum']
            }
            ordinal_pipeline = Pipeline(
                steps = [
                    (
                        'ordinal_encoder', 
                        OrdinalEncoder(
                            categories = [
                                ordinal_features_order['education_level'], 
                                ordinal_features_order['income_category'],
                                ordinal_features_order['card_category']
                            ]
                        ),
                    ),
                ]   
            )

            # Final Preprocessor
            nn_preprocessor = ColumnTransformer(
                transformers = [
                    ('nominal_features', OrdinalEncoder(), nominal_features),
                    ('ordinal_features', ordinal_pipeline, ordinal_features),
                    ('minmax_scaler_features', MinMaxScaler(), minmax_scaler_features),
                    ('robust_scaler_features', RobustScaler(), robust_scaler_features),        
                ], 
                remainder = 'passthrough'
            ) 
            
            return nn_preprocessor
        
        except Exception as e:
            print(f'[ERROR] Failed to create pipeline with preprocessed data: {str(e)}.')