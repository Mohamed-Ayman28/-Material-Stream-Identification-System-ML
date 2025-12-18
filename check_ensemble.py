import joblib

ens = joblib.load('models/ensemble_enhanced.pkl')
print('Ensemble estimators:')
for name, est in ens.estimators:
    print(f'  {name}: {type(est).__name__}')
    print(f'    n_features_in_: {getattr(est, "n_features_in_", "N/A")}')
