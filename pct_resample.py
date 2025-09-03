from sklearn.utils import resample

def pct_resample (features, outcome, resample_pct=0.8, **kwargs):
    #resample a percentage samples from inputs
    resample_len = round(len(features) * resample_pct)
    return resample(features, outcome, n_samples=resample_len, **kwargs)