
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score

# 1. Load Dataset
print("Loading dataset...")
df = pd.read_excel(r"D:\project_assignment\dataset\bank-additional-full.csv.xlsx")
print("First 2 rows of dataset:")
print(df.head(2))  # Sanity check

# 2. Data Visualization and Exploration
print("\nClass Distribution:")
sns.countplot(x='y', data=df)
plt.title("Subscription Distribution")
plt.savefig("subscription_distribution.png")

# Correlation analysis (only for numerical data)
print("\nGenerating correlation heatmap...")
numeric_corr = df.corr(numeric_only=True)
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")

# 3. Preprocessing
print("\nChecking for missing values...")
print(df.isnull().sum())

print("\nEncoding categorical variables...")
df_encoded = pd.get_dummies(df, drop_first=True)

print("\nFeature Scaling...")
X = df_encoded.drop('y_yes', axis=1)
y = df_encoded['y_yes']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

# 4. Feature Importance using RandomForest
from sklearn.ensemble import RandomForestClassifier
print("\nComputing feature importances...")
rf = RandomForestClassifier()
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(10)

# Save feature importance plot
plt.figure(figsize=(8, 6))
top_features.plot(kind='barh')
plt.title("Top 10 Feature Importances")
plt.tight_layout()
plt.savefig("feature_importance.png")

# 5. Model Building - Train/Test Split
print("\nSplitting dataset: 80% train / 20% test")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Logistic Regression
print("\nTraining Logistic Regression model...")
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# 7. Decision Tree with Hyperparameter Tuning
print("\nTraining Decision Tree model with GridSearchCV...")
param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}
tree = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(tree, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 8. Model Evaluation
print("\nEvaluating models...")

# Logistic Regression
y_pred_log = log_reg.predict(X_test)
log_roc_auc = roc_auc_score(y_test, log_reg.predict_proba(X_test)[:, 1])
print("\nLogistic Regression Report:")
print(classification_report(y_test, y_pred_log))
print(f"ROC AUC: {log_roc_auc:.4f}")

# Decision Tree
y_pred_tree = grid_search.predict(X_test)
tree_roc_auc = roc_auc_score(y_test, grid_search.predict_proba(X_test)[:, 1])
print("\nDecision Tree Report:")
print(classification_report(y_test, y_pred_tree))
print(f"ROC AUC: {tree_roc_auc:.4f}")

# 9. Final Summary
if log_roc_auc > tree_roc_auc:
    print("\nLogistic Regression performs better based on ROC AUC.")
else:
    print("\nDecision Tree performs better based on ROC AUC.")

print("\nAll plots saved as PNG files.")
