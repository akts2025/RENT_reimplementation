from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample

from sksurv.datasets import load_breast_cancer
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

from rent import RENTFeatureSelection

#Example of feature selection for survival analysis with scikit-survival
seed = 1

#Import breast cancer dataset from scikit-survival
X, y = load_breast_cancer()

#Ordinal encoding for text-based categorical features
encode_list = [('encode_text', OrdinalEncoder(), ['er', 'grade'])]
encode_features = ColumnTransformer(encode_list, remainder='passthrough', verbose_feature_names_out=False)
encode_features.set_output(transform='pandas')
X = encode_features.fit_transform(X)

#Custom stratification scheme based on survival status and time
def survival_stratify (s):
    if s[0] == False:
        return 0
    
    if s[1] <= 1000:
        return 4
    elif s[1] <= 3000:
        return 3
    elif s[1] <= 5000:
        return 2
    else: 
        return 1

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=[survival_stratify(s) for s in y], random_state=seed)
test_event = [s[0] for s in y_test]
test_time = [s[1] for s in y_test]

#Fit model without feature selection
unfiltered_pipeline = make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis())
unfiltered_pipeline.fit(X_train, y_train)
unfiltered_predictions = unfiltered_pipeline.predict(X_test)
print(f'C-index of model with all features: {concordance_index_censored(test_event, test_time, unfiltered_predictions)[0]}')

#Prepare arguments for feature selection
class ModCoxnet (CoxnetSurvivalAnalysis):
    def fit (self, X, y):
        super().fit(X, y)
        self.coef_ = super()._get_coef(None)[0]
        return self

#Use default arguments provided by scikit-survival
enet_classification_model_args = {}

def pct_resample_survival_stratified (features, outcome, resample_pct=0.8, **kwargs):
    #resample a percentage samples from inputs with stratification based on survival status and time
    resample_len = round(len(features) * resample_pct)
    return resample(features, outcome, n_samples=resample_len, stratify=[survival_stratify(s) for s in outcome], **kwargs)

resampler_args = {'resample_pct' : 0.8,
                  'replace' : True,}

#Fit model with feature selection
output_df_scaler = StandardScaler()
output_df_scaler.set_output(transform='pandas')
enet_classifcation_model = ModCoxnet
rent_args = {
    'enet_model' : enet_classifcation_model, 
    'enet_model_args' : enet_classification_model_args,
    'resampler' : pct_resample_survival_stratified,
    'resampler_args' : resampler_args,
}
rent_pipeline = make_pipeline(output_df_scaler, RENTFeatureSelection(**rent_args), CoxnetSurvivalAnalysis())
rent_pipeline.fit(X_train, y_train)
rent_predictions = rent_pipeline.predict(X_test)
#C-index is improved after feature selection
print(f'C-index of model with selected features: {concordance_index_censored(test_event, test_time, rent_predictions)[0]}')
print(f'{len(rent_pipeline["rentfeatureselection"].selected_features_)} features selected')
print(rent_pipeline["rentfeatureselection"].selected_features_)