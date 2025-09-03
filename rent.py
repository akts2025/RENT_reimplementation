import warnings
from collections import defaultdict

import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.linear_model import ElasticNet

from pct_resample import pct_resample
from rent_selection_criteria import coef_nonzero, coef_signstable, coef_ttest

class RENTFeatureSelection (TransformerMixin):
    def __init__ (self, enet_model=None, enet_model_args=None, resampler=None, resampler_args=None, iter_count=500, selection_criteria=None, selection_thresholds=None, selection_thresholds_to_pass=None):
        self.init_default_args(enet_model, enet_model_args, resampler, resampler_args, iter_count, selection_criteria, selection_thresholds, selection_thresholds_to_pass)

    def init_default_args (self, enet_model, enet_model_args, resampler, resampler_args, iter_count, selection_criteria, selection_thresholds, selection_thresholds_to_pass):
        if enet_model is None:
            self.enet_model_ = ElasticNet
            self.enet_model_args_ = {'fit_intercept' : False}
        else:
            self.enet_model_ = enet_model
            self.enet_model_args_ = enet_model_args
        
        if resampler is None:
            #resample 80% of dataset without replacement by default
            self.resampler_ = pct_resample
            self.resampler_args_ = {'resample_pct' : 0.8, 'replace' : True}
        else:
            self.resampler_ = resampler
            self.resampler_args_ = resampler_args

        if not isinstance(iter_count, int):
            warnings.warn('iter_count in RENTFeatureSelection is not an integer!')
        self.iter_count_ = iter_count

        if selection_criteria is None:
            #usual RENT criteria is
            #percentage of iterations with non-zero coef
            #coefs have stable signs across iterations
            #t-test for coef=0 across iterations
            self.selection_criteria_ = (coef_nonzero, coef_signstable, coef_ttest)
        else:
            self.selection_criteria_ = selection_criteria

        if selection_thresholds is None:
            self.selection_thresholds_ = (0.9, 0.9, 0.975)
        else:
            self.selection_thresholds_ = selection_thresholds
        
        if selection_thresholds_to_pass is None:
            #all thresholds must pass by default
            self.selection_thresholds_to_pass_ = len(self.selection_criteria_)
        else:
            if not isinstance(selection_thresholds_to_pass, int):
                warnings.warn('selection_thresholds_to_pass in RENTFeatureSelection is not an integer!')
            self.selection_thresholds_to_pass_ = selection_thresholds_to_pass
        
    def fit (self, X, y):
        self.selected_features_ = self.run_feature_selection(X, y)
        return self

    def run_feature_selection (self, features, outcome):
        #get enet model coefficents from resampled training data
        coef_dict = defaultdict(list)
        for current_iter_count in range(self.iter_count_):
            resample_features, resample_outcome = self.resampler_(features, outcome, random_state=current_iter_count, **self.resampler_args_)
            enet_model = self.enet_model_(**self.enet_model_args_).fit(resample_features, resample_outcome)
            enet_features = enet_model.feature_names_in_
            enet_coef = enet_model.coef_
            for feature, coef in zip(enet_features, enet_coef):
                coef_dict[feature].append(coef)

        #calculate tau and determine selected features
        self.tau_dict_ = {}
        output_list = []
        for k in coef_dict.keys():
            tau_list = []
            for criteria in self.selection_criteria_:
                tau_list.append(criteria(coef_dict[k]))

            self.tau_dict_[k] = tau_list

            include_decision = [t>=thres for t, thres in zip(tau_list, self.selection_thresholds_)]

            if sum(include_decision) >= self.selection_thresholds_to_pass_:
                output_list.append(k)

        if len(output_list) == 0:
            warnings.warn('Zero features were selected by RENTFeatureSelection!')
        
        return output_list

    def transform (self, X):
        return X[self.selected_features_]

    def get_feature_selection_df (self):
        tau_df = pd.DataFrame.from_dict(self.tau_dict_)
        tau_df = tau_df.transpose()
        tau_df.columns = [f'tau{i}' for i in range(1, len(tau_df.columns)+1)]
        tau_df['selected'] = [i in self.selected_features_ for i in tau_df.index]
        return tau_df