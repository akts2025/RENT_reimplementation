from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

from rent import RENTFeatureSelection

#Example of using default settings for linear regression
seed = 1

#generate dataset with 100 features, but only 10 features are informative
X, y = make_regression(random_state=seed)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)

#Fit model without feature selection
unfiltered_pipeline = make_pipeline(StandardScaler(), ElasticNet())
unfiltered_pipeline.fit(X_train, y_train)
unfiltered_predictions = unfiltered_pipeline.predict(X_test)
print(f'MSE of model with all features: {mean_squared_error(y_test, unfiltered_predictions)}')

#Fit model with feature selection
output_df_scaler = StandardScaler()
output_df_scaler.set_output(transform='pandas')
rent_pipeline = make_pipeline(output_df_scaler, RENTFeatureSelection(), ElasticNet())
rent_pipeline.fit(X_train, y_train)
rent_predictions = rent_pipeline.predict(X_test)
#MSE is improved after feature selection
print(f'MSE of model with selected features: {mean_squared_error(y_test, rent_predictions)}')
print(f'{len(rent_pipeline["rentfeatureselection"].selected_features_)} features selected')
print(rent_pipeline['rentfeatureselection'].get_feature_selection_df())