from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
CATEGORICAL=["Sex","Chest pain type","FBS over 120","EKG results","Exercise angina","Slope of ST","Number of vessels fluro","Thallium"]
NUMERIC=["Age","BP","Cholesterol","Max HR","ST depression"]
def build_model():
    num=Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())])
    cat=Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))])
    prep=ColumnTransformer([("num",num,NUMERIC),("cat",cat,CATEGORICAL)])
    return Pipeline([("preprocess",prep),("classifier",LogisticRegression(max_iter=2000,class_weight="balanced"))])
