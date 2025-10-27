import pandas as pd
import numpy as np
import joblib 
import pickle
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
df=pd.read_csv('heart.csv').drop_duplicates()

x=df.drop(['HeartDisease'],axis=1)
y=df['HeartDisease'].map({0:'No',1:'Yes'})

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

num_attr=['Age','RestingBP','Cholesterol','FastingBS', 'MaxHR', 'Oldpeak']
cat_attr=['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']

num_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy='mean')),
    ('scaler',StandardScaler())
])
cat_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('encoder',OneHotEncoder(sparse_output=False))
])
preprocessing=ColumnTransformer(transformers=[
    ('cat',cat_pipeline,cat_attr),
    ('num',num_pipeline,num_attr)
])
pipeline=Pipeline([
    ('preprocessing',preprocessing),
    ('model',KNeighborsClassifier(n_neighbors=5))
  
])
pipeline.fit(x_train,y_train)
y_pred=pipeline.predict(x_test)
print('accuracy score :',accuracy_score(y_test,y_pred)*100)

print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))  # Optional: show confusion matrix

result=pd.DataFrame({
    'Actual':y_test.values,
    'Prediction':y_pred
})

result.to_csv('Output.csv',index=False)
print('save the file...')

joblib.dump(pipeline,'heart.pkl')