import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

# ----------------- Paths -----------------
processed_dir = Path("data/processed")
output_csv = Path("C:/Users/utente/OneDrive/Documents/Fake News Article/data/processed/baseline_predictions.csv")

# ----------------- Load processed data -----------------
train = pd.read_csv(processed_dir / "train.csv")
val = pd.read_csv(processed_dir / "val.csv")
test = pd.read_csv(processed_dir / "test.csv")

X_train, y_train = train['text'], train['label']
X_val, y_val = val['text'], val['label']
X_test, y_test = test['text'], test['label']

# ----------------- TF-IDF vectorization -----------------
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

# ----------------- Logistic Regression -----------------
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_vec, y_train)

y_pred_val = lr.predict(X_val_vec)
y_pred_test = lr.predict(X_test_vec)

print("=== Logistic Regression - Validation ===")
print(classification_report(y_val, y_pred_val))
print("Confusion matrix:")
print(confusion_matrix(y_val, y_pred_val))

print("=== Logistic Regression - Test ===")
print(classification_report(y_test, y_pred_test))

# ----------------- Gradient Boosted Trees -----------------
gbt = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
gbt.fit(X_train_vec.toarray(), y_train)

y_pred_val_gbt = gbt.predict(X_val_vec.toarray())
y_pred_test_gbt = gbt.predict(X_test_vec.toarray())

print("=== Gradient Boosted Trees - Validation ===")
print(classification_report(y_val, y_pred_val_gbt))
print("Confusion matrix:")
print(confusion_matrix(y_val, y_pred_val_gbt))

print("=== Gradient Boosted Trees - Test ===")
print(classification_report(y_test, y_pred_test_gbt))


