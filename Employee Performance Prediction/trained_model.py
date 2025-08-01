import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import pickle

df = pd.read_csv('garments_worker_productivity.csv') 
print(df.head())

print(df.describe())

print(df.shape)

print(df.info())

print(df.isnull().sum())

df.drop(['wip'],axis=1,inplace=True)
print(df.head())
print(df.columns)


df['date']=pd.to_datetime(df['date'])
print(df.date)


df['month']=df['date'].dt.month
df.drop(['date'],axis=1,inplace=True)
print(df.month)



print(df['department'].value_counts())
df['department']=df['department'].apply(lambda x: 'finishing' if x.replace(" ","")=='finishing' else 'sweing')
print(df['department'].value_counts())

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in ['quarter', 'department', 'day', 'team']:
    df[col] = le.fit_transform(df[col])


x=df.drop(['actual_productivity'],axis=1)
y=df['actual_productivity']
X=x.to_numpy()
print(X)


x_train,x_test,y_train,y_test=train_test_split(X,y,train_size=0.8,random_state=0)


linear=LinearRegression()
linear.fit(x_train, y_train)
predict_linear=linear.predict(x_test)
print("mean_square_error: ",mean_squared_error(y_test,predict_linear))
print("mean_absolute_error: ",mean_absolute_error(y_test,predict_linear))
print("R2 score{}: ".format(r2_score(y_test,predict_linear)))
r2_linear=r2_score(y_test,predict_linear)
print(r2_linear)

random=RandomForestRegressor(n_estimators=100, random_state=42)
random.fit(x_train, y_train)
predict_random=random.predict(x_test)
print("mean_square_error: ",mean_squared_error(y_test,predict_random))
print("mean_absolute_error: ",mean_absolute_error(y_test,predict_random))
print("R2 score{}: ".format(r2_score(y_test,predict_random)))
r2_random=r2_score(y_test,predict_random)
print(r2_random)

xgb= XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb.fit(x_train, y_train)
predict_xgb=xgb.predict(x_test)
print("mean_square_error: ",mean_squared_error(y_test,predict_xgb))
print("mean_absolute_error: ",mean_absolute_error(y_test,predict_xgb))
print("R2 score{}: ".format(r2_score(y_test,predict_xgb)))
r2_xgb=r2_score(y_test,predict_xgb)
print(r2_xgb)


best_model = None
best_score = max(r2_linear, r2_random, r2_xgb)

if best_score == r2_linear:
    best_model = linear
    print("Best model: Linear Regression")
elif best_score == r2_random:
    best_model = random
    print("Best model: Random Forest")
else:
    best_model = xgb
    print("Best model: XGBoost")

# Save best model to use in Flask
with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("Model saved as model.pkl")
print(x.columns)

from sklearn.metrics import confusion_matrix, classification_report, f1_score

# Function to map productivity score to classes
def categorize_output(predictions):
    return [
        0 if val <= 0.3 else 1 if val <= 0.7 else 2
        for val in predictions
    ]

# Categorize y_test and predictions
y_true_cat = categorize_output(y_test)

# Linear Regression
y_pred_linear = categorize_output(linear.predict(x_test))
print("Linear Regression")

print("Classification Report:\n", classification_report(y_true_cat, y_pred_linear))
print("F1 Score:", f1_score(y_true_cat, y_pred_linear, average='weighted'))
print("-" * 50)

# Random Forest
y_pred_rf = categorize_output(random.predict(x_test))
print("Random Forest")

print("Classification Report:\n", classification_report(y_true_cat, y_pred_rf))
print("F1 Score:", f1_score(y_true_cat, y_pred_rf, average='weighted'))
print("-" * 50)

# XGBoost
y_pred_xgb = categorize_output(xgb.predict(x_test))
print("XGBoost Regressor")
print("Classification Report:\n", classification_report(y_true_cat, y_pred_xgb))
print("F1 Score:", f1_score(y_true_cat, y_pred_xgb, average='weighted'))
