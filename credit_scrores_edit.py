import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import joblib

df = pd.read_csv(r'C:\Users\emman\OneDrive\Documents\credit_scores\credit_scores.csv')

df.drop(columns=["Name", "SSN", "ID", "Customer_ID"], inplace=True)

target = df['Credit_Score']
df.drop(columns=['Credit_Score'], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2, random_state=1)

num_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
cat_cols = X_train.select_dtypes(include=['object']).columns

num_imputer = SimpleImputer(strategy='mean')
X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
X_test[num_cols] = num_imputer.transform(X_test[num_cols])

scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])
    label_encoders[col] = le

parameters = {
    'kernel': ['linear', 'rbf'],
    'C': [0.01, 10, 20]
}
svc = SVC()
clf = GridSearchCV(svc, parameters, scoring='accuracy', cv=5)
clf.fit(X_train, y_train)

best_model = clf.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Best Model Accuracy: {accuracy}')

df_full = pd.read_csv(r'C:\Users\emman\OneDrive\Documents\credit_scores.csv')
df_full.drop(columns=["Name", "SSN", "ID", "Customer_ID"], inplace=True)
target_full = df_full['Credit_Score']
df_full.drop(columns=['Credit_Score'], inplace=True)

df_full[num_cols] = num_imputer.fit_transform(df_full[num_cols])
df_full[num_cols] = scaler.fit_transform(df_full[num_cols])
df_full[cat_cols] = cat_imputer.fit_transform(df_full[cat_cols])

for col in cat_cols:
    df_full[col] = label_encoders[col].fit_transform(df_full[col])

best_model.fit(df_full, target_full)

joblib.dump(best_model, 'best_svm_model.joblib')
