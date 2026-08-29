from src.train import load_data, candidate_searches
from sklearn.model_selection import StratifiedKFold

def test_dataset_contract():
    X,y=load_data("Heart_Disease_Prediction.csv")
    assert X.shape==(270,13)
    assert set(y.unique())=={0,1}

def test_searches_construct():
    cv=StratifiedKFold(5,shuffle=True,random_state=42)
    searches=candidate_searches(cv)
    assert {"logistic_regression","random_forest"}==set(searches)
