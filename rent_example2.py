from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.metrics import f1_score

from rent import RENTFeatureSelection

#Example of customizing arguments for logistic regression
seed = 1

#generate dataset with 20 features, but only 2 features are informative
X, y = make_classification(random_state=seed)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=seed)

#Fit model without feature selection
unfiltered_pipeline = make_pipeline(StandardScaler(), LogisticRegression())
unfiltered_pipeline.fit(X_train, y_train)
unfiltered_predictions = unfiltered_pipeline.predict(X_test)
print(f'F1 score of model with all features: {f1_score(y_test, unfiltered_predictions)}')

#Prepare arguments for feature selection
class ModLogReg (LogisticRegression):
    def fit (self, X, y):
        super().fit(X, y)
        self.coef_ = self.coef_[0]
        return self

enet_classification_model_args = {'penalty' : 'elasticnet',
                                  'l1_ratio' : 0.5,
                                  'C' : 0.5,
                                  'fit_intercept' : False,
                                  'solver' : 'saga',}

def pct_resample_stratified (features, outcome, resample_pct=0.8, **kwargs):
    #resample a percentage samples from inputs with stratification based on outcome
    resample_len = round(len(features) * resample_pct)
    return resample(features, outcome, n_samples=resample_len, stratify=outcome, **kwargs)

resampler_args = {'resample_pct' : 0.8,
                  'replace' : True,}

#Fit model with feature selection
output_df_scaler = StandardScaler()
output_df_scaler.set_output(transform='pandas')
enet_classifcation_model = ModLogReg
rent_args = {
    'enet_model' : enet_classifcation_model, 
    'enet_model_args' : enet_classification_model_args,
    'resampler' : pct_resample_stratified,
    'resampler_args' : resampler_args,
}
rent_pipeline = make_pipeline(output_df_scaler, RENTFeatureSelection(**rent_args), LogisticRegression())
rent_pipeline.fit(X_train, y_train)
rent_predictions = rent_pipeline.predict(X_test)
#F1 score is improved after feature selection
print(f'F1 score of model with selected features: {f1_score(y_test, rent_predictions)}')
print(f'{len(rent_pipeline["rentfeatureselection"].selected_features_)} features selected')
print(rent_pipeline['rentfeatureselection'].get_feature_selection_df())