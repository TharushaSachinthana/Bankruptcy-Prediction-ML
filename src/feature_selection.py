from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.linear_model import LogisticRegression

def select_features(X, y, output_path):
    chi2_selector = SelectKBest(chi2, k=10).fit(X, y)
    chi2_features = X.columns[chi2_selector.get_support()]
    rfe_model = LogisticRegression(max_iter=1000)
    rfe_selector = RFE(rfe_model, n_features_to_select=10).fit(X, y)
    rfe_features = X.columns[rfe_selector.support_()]
    selected_features = list(set(chi2_features) | set(rfe_features))
    with open(output_path, 'w') as f:
        f.write('\n'.join(selected_features))
    return X[selected_features]
