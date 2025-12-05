# %% [markdown]
# # AD699 Assignment 3: Classification Trees & Random Forests
# 
# **Student:** Saurabh Sharma  
# 
# ---

# %% [markdown]
# ## Setup

# %%
# Core libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# OS utils
import os

# Settings
import warnings
warnings.filterwarnings('ignore')  # Silence noisy warnings for cleaner output
sns.set_style('whitegrid')  # Consistent plot aesthetics

# Save the plots in this directory
os.makedirs('outputs', exist_ok=True)

# Random seed - CHANGE TO YOUR BU ID LAST 3 DIGITS
SEED = 750

# %% [markdown]
# ---
# ## Part 1: Classification Tree

# %% [markdown]
# ### Q1: Load the Dataset

# %%
# Load data with proper encoding; skip the second header row; use first column as index
df = pd.read_csv('data/Colleges.csv', encoding='latin-1', index_col=0, skiprows=[1])

# Basic shape check for sanity
rows, columns = df.shape
print(f"Dataset shape: Rows({rows})XColumns({columns})")

# Peek at a random sample to verify loading worked
df.sample(5)

# %% [markdown]
# ### Q2: Describe the Dataset & Summary Statistics

# %%
print("Dataset Information:")
print(f"  - {len(df)} colleges")
print(f"  - {len(df.columns)} variables")
print(f"\nVariables: {list(df.columns)}")

# %%
# Summary statistics
df.describe()

# %%
# Check data types
df.dtypes

# %%
# Check for missing values
df.isnull().sum()

# %% [markdown]
# From above it can be observed that there are no null columns.

# %% [markdown]
# ### Q3: Create Yield Variable

# %%
# Create yield = (Enroll / Accept) * 100
df['yield'] = (df['Enroll'] / df['Accept']) * 100

# %%
df['yield'].sample(5, random_state=SEED)

# %%
print("Yield Statistics:")
print(df['yield'].describe())

# %%
# Visualize yield distribution
plt.figure(figsize=(10, 5))
plt.hist(df['yield'], bins=20, edgecolor='black')
plt.axvline(df['yield'].median(), color='red', linestyle='--', label=f'Median = {df["yield"].median():.2f}%')
plt.xlabel('Yield (%)')
plt.ylabel('Number of Colleges')
plt.title('Distribution of College Yield')
plt.legend()
plt.savefig('outputs/yield_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# The yield distribution shows a roughly bell-shaped pattern with a slight right skew, ranging from approximately 10% to 100%. The median yield of 38.74% divides the dataset into equal halves for classification purposes. Most colleges cluster in the 20-50% yield range, with the peak around 30-35%, indicating typical conversion rates in higher education. The right tail extending to 100% represents highly selective institutions where nearly all accepted students enroll, while the left side represents less selective schools where students have multiple options. This distribution suggests yield is influenced by factors like selectivity, reputation, and financial aid, which our classification tree will attempt to model."

# %%
# Delete the original variables
df = df.drop(['Enroll', 'Accept'], axis=1)

print(f"New shape after dropping Enroll and Accept: {df.shape}")

# %% [markdown]
# ### Q4: Convert Yield to Factor (High/Low)

# %%
# Convert to high/low based on median
median_yield = df['yield'].median()

df['yield_category'] = df['yield'].apply(lambda x: 'high' if x >= median_yield else 'low')

print(f"Median yield: {median_yield:.2f}%\n")
print("Class distribution:")
print(df['yield_category'].value_counts())

# %%
# Drop the numeric yield column
df = df.drop('yield', axis=1)

# %% [markdown]
# ### Q5: Partition Data (60/40 Train/Validation)

# %%
# Prepare X and y
# X = Features (independent variables): All columns except the target variable
# y = Target (dependent variable): What we're trying to predict
X = df.drop('yield_category', axis=1)  # Remove target column to get features
y = df['yield_category']  # Keep only the target column (high/low)

# Convert Private to numeric (from categorical to binary)
# Machine learning models need numeric input, not text
# 'Yes' becomes 1, 'No' becomes 0
X['Private'] = X['Private'].map({'Yes': 1, 'No': 0})

# Fill any missing values with mean
# Some colleges might have missing data for certain features
# We replace NaN values with the average (mean) of that column
# This prevents errors during model training
X = X.fillna(X.mean())

# %%
# Split the data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.4, random_state=SEED, stratify=y
)

print(f"Training set: {len(X_train)} samples")
print(y_train.value_counts())
print(f"\nValidation set: {len(X_val)} samples")
print(y_val.value_counts())

# %% [markdown]
# ### Q6: Build Classification Tree

# %%
# Build the tree
dt_model = DecisionTreeClassifier(max_depth=5, random_state=SEED)
dt_model.fit(X_train, y_train)

print(f"Tree depth: {dt_model.get_depth()}")
print(f"Number of leaves: {dt_model.get_n_leaves()}")

# %% [markdown]
# ### Q7: Display the Tree

# %%
plt.figure(figsize=(20, 10))
plot_tree(dt_model, 
          feature_names=X.columns,
          class_names=['high', 'low'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title('Classification Tree for College Yield', fontsize=16)
plt.savefig('outputs/classification_tree.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### Q8: What Did You See?

# %%
# Feature importance
# Create a DataFrame to store and display feature importance scores
# Feature importance tells us which variables had the most influence on the tree's decisions
importance_df = pd.DataFrame({
    'Feature': X.columns,  # Column names from our dataset
    'Importance': dt_model.feature_importances_  # Importance scores from the trained model (0-1 scale)
}).sort_values('Importance', ascending=False)  # Sort from most to least important

print("Feature Importance:")
print(importance_df.head(10))  # Display top 10 most important features

# %% [markdown]
# **Observations:**
# 
# The model reveals that `Room.Board` cost is the primary predictor of yield, with lower-cost colleges generally achieving higher yield. The tree's complex structure shows that predicting yield requires considering multiple factors including `tuition`, `applications`, `graduation rates`, and `student quality metrics`, as no single variable perfectly separates `high` from `low` yield colleges.

# %% [markdown]
# ### Q9: Root Node Analysis

# %%
# Get root node information
# Access the internal tree structure to analyze the first split
tree = dt_model.tree_  # Get the underlying tree structure from the trained model
root_feature_idx = tree.feature[0]  # Index of the feature used at root (position 0 = root node)
root_threshold = tree.threshold[0]  # The cutoff value used to split at the root
root_feature = X.columns[root_feature_idx]  # Convert index to actual feature name

print("Root Node Split:")
print(f"  Variable: {root_feature}")  # Which feature splits the data first
print(f"  Threshold: {root_threshold:.2f}")  # At what value the split occurs
print(f"  Rule: If {root_feature} <= {root_threshold:.2f} go left, else go right")  # Decision rule

# %% [markdown]
# **Why is the root node significant?**
# 
# The root node is significant because it represents the single most discriminative feature in the dataset. Room.Board at the `$3,615.50` threshold provides the maximum information gain, meaning this split best separates high-yield from low-yield colleges. As the foundation of the tree, this decision determines the initial grouping from which all subsequent splits are made. The selection of Room.Board suggests that housing affordability is a primary factor influencing student enrollment decisions.

# %% [markdown]
# ### Q10: Which Variables Appeared in the Model?

# %%
# Find which features were used
# Filter the importance dataframe to separate used and unused features
features_used = importance_df[importance_df['Importance'] > 0]['Feature'].tolist()  # Features with importance > 0 were used in splits
features_not_used = importance_df[importance_df['Importance'] == 0]['Feature'].tolist()  # Features with 0 importance were never selected

# Display summary of feature usage
print(f"Features used: {len(features_used)} out of {len(X.columns)}")
print(f"\nUsed: {features_used}")  # List of features that appear in the tree
print(f"\nNot used: {features_not_used}")  # Features ignored by the algorithm

# %% [markdown]
# **Why not all variables?**
# 
# No, only 8 out of 16 features appear in the model. The decision tree algorithm selectively uses features that maximize information gain at each split. 
# Variables like `F.Undergrad`, `Private`, and `Expend` were excluded because: 
# 
# 1. They're redundant with features already used (e.g., Apps captures size), 
# 2. They have low predictive power for yield, 
# 3. The max_depth=5 constraint limits the number of splits, and 
# 4. The most important features (`Room.Board`, `Apps`, `Outstate`) already explain most of the variation in yield. 
# 
# This feature selection is actually beneficial—it creates a simpler, more interpretable model focused on the variables that truly matter for predicting enrollment decisions.

# %% [markdown]
# ### Q11: Confusion Matrices & Performance

# %%
# Make predictions
# Use the trained decision tree to predict yield categories for both datasets
y_train_pred = dt_model.predict(X_train)  # Predictions on training data (data the model has seen)
y_val_pred = dt_model.predict(X_val)  # Predictions on validation data (unseen data)

# Calculate accuracies
# Compare predictions to actual values to measure performance
train_acc = accuracy_score(y_train, y_train_pred)  # % of correct predictions on training set
val_acc = accuracy_score(y_val, y_val_pred)  # % of correct predictions on validation set

# Display results
print(f"Training Accuracy: {train_acc:.4f}")  # How well model performs on training data
print(f"Validation Accuracy: {val_acc:.4f}")  # How well model generalizes to new data
print(f"Overfitting Gap: {(train_acc - val_acc):.4f}")  # Difference indicates overfitting (larger = more overfitting)

# %%
# Training confusion matrix
# A confusion matrix shows how many predictions were correct vs incorrect for each class
print("Training Set:")
cm_train = confusion_matrix(y_train, y_train_pred)  # Compare actual labels to predicted labels

# Display in a readable DataFrame format with labels
print(pd.DataFrame(cm_train, 
                   index=['Actual: high', 'Actual: low'],  # Rows = actual values
                   columns=['Pred: high', 'Pred: low']))  # Columns = predicted values

# Classification report provides precision, recall, F1-score for each class
print(f"\n{classification_report(y_train, y_train_pred)}")

# %%
# Validation confusion matrix
# This is the MOST IMPORTANT evaluation - shows performance on unseen data
print("Validation Set:")
cm_val = confusion_matrix(y_val, y_val_pred)  # Compare actual vs predicted on validation set
# Display in DataFrame format for easy reading
print(pd.DataFrame(cm_val,
                   index=['Actual: high', 'Actual: low'],  # Rows = true labels
                   columns=['Pred: high', 'Pred: low']))  # Columns = model predictions
# Detailed metrics: precision (accuracy of positive predictions), recall (coverage), F1 (harmonic mean)
print(f"\n{classification_report(y_val, y_val_pred)}")

# %%
# Visualize confusion matrices
# Create side-by-side heatmaps to compare training vs validation performance
fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns for comparison

# Left plot: Training confusion matrix
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=axes[0])  # Blue colormap, show counts
axes[0].set_title(f'Decision Tree - Training\nAccuracy: {train_acc:.2%}')
axes[0].set_ylabel('Actual')  # Y-axis = true labels
axes[0].set_xlabel('Predicted')  # X-axis = predicted labels

# Right plot: Validation confusion matrix (more important!)
sns.heatmap(cm_val, annot=True, fmt='d', cmap='Oranges', ax=axes[1])  # Orange colormap for contrast
axes[1].set_title(f'Decision Tree - Validation\nAccuracy: {val_acc:.2%}')
axes[1].set_ylabel('Actual')
axes[1].set_xlabel('Predicted')

plt.tight_layout()  # Adjust spacing between plots
plt.savefig('outputs/decision_tree_confusion_matrices.png', dpi=300, bbox_inches='tight')  # Save to outputs
plt.show()

# %% [markdown]
# **Performance Assessment:**
# 
# The decision tree achieved `78.92%` training accuracy and `69.13%` validation accuracy, showing a `9.79%` overfitting gap—acceptable but indicating some memorization of training patterns. The model performs moderately well, significantly better than random guessing (50%), but reveals a critical weakness: it's much better at identifying low-yield colleges (82% recall) than high-yield colleges (56% recall). With 68 false negatives on validation, the model is overly conservative in predicting high yield. While the ~70% validation accuracy is decent, the class imbalance in performance and moderate overfitting suggest this single decision tree has limitations that ensemble methods might address.

# %% [markdown]
# ---
# ## Part 2: Random Forest

# %% [markdown]
# ### Q1-2: Same Dataset and Partition
# 
# Using the same X_train, X_val, y_train, y_val from above.

# %% [markdown]
# ### Q3: Build Random Forest Model

# %%
# Build Random Forest
# n_estimators: number of trees in the ensemble
# max_depth: limit tree depth to control overfitting
# random_state: reproducibility
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=SEED)
rf_model.fit(X_train, y_train)  # Train on the training split

# Summarize key hyperparameters of the trained model
print(f"Number of trees: {rf_model.n_estimators}")
print(f"Max depth per tree: {rf_model.max_depth}")

# %%
# Feature importance
# Build a DataFrame so we can sort and inspect which features the forest relies on most
rf_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

# Show the top-ranked features to see drivers of yield predictions
print("Random Forest Feature Importance:")
print(rf_importance.head(10))

# %% [markdown]
# ### Q4: Confusion Matrices & Performance

# %%
# Make predictions on both seen (train) and unseen (validation) data
y_train_pred_rf = rf_model.predict(X_train)
y_val_pred_rf = rf_model.predict(X_val)

# Calculate accuracy metrics to check fit quality and generalization
train_acc_rf = accuracy_score(y_train, y_train_pred_rf)
val_acc_rf = accuracy_score(y_val, y_val_pred_rf)

# Report headline metrics and the overfitting gap (train - val)
print(f"Training Accuracy: {train_acc_rf:.4f}")
print(f"Validation Accuracy: {val_acc_rf:.4f}")
print(f"Overfitting Gap: {(train_acc_rf - val_acc_rf):.4f}")

# %%
# Training confusion matrix helps us see class-wise correctness on data the model saw during training
print("Training Set:")
cm_train_rf = confusion_matrix(y_train, y_train_pred_rf)

# Display counts with readable row/column labels
print(pd.DataFrame(cm_train_rf,
                   index=['Actual: high', 'Actual: low'],
                   columns=['Pred: high', 'Pred: low']))

# Precision/recall/F1 give more detail than accuracy alone
print(f"\n{classification_report(y_train, y_train_pred_rf)}")

# %%
# Validation confusion matrix
print("Validation Set:")
cm_val_rf = confusion_matrix(y_val, y_val_pred_rf)
print(pd.DataFrame(cm_val_rf,
                   index=['Actual: high', 'Actual: low'],
                   columns=['Pred: high', 'Pred: low']))
print(f"\n{classification_report(y_val, y_val_pred_rf)}")

# %%
# Visualize confusion matrices for train vs validation side-by-side to spot over/under-fitting
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Training heatmap: should be high on the diagonal if the model fits training data well
sns.heatmap(cm_train_rf, annot=True, fmt='d', cmap='Greens', ax=axes[0])
axes[0].set_title(f'Random Forest - Training\nAccuracy: {train_acc_rf:.2%}')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# Validation heatmap: key view for generalization; compare diagonals vs off-diagonals
sns.heatmap(cm_val_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title(f'Random Forest - Validation\nAccuracy: {val_acc_rf:.2%}')
axes[1].set_ylabel('Actual')
axes[1].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig('outputs/random_forest_confusion_matrices.png', dpi=300, bbox_inches='tight')  # Save figure for reuse
plt.show()

# %% [markdown]
# **Performance Assessment:**
# 
# Random Forest achieved `90.11%` training accuracy and `73.95%` validation accuracy, outperforming the decision tree by `4.82` percentage points on validation data. While the model shows a larger overfitting gap (`16.15%` vs `9.79%`), the superior validation performance is what matters for real-world predictions. Critically, Random Forest dramatically improves high-yield detection—raising recall from `56%` to `72%` and achieves balanced performance across both classes (72-76% recall for both). The model reduces false negatives from 68 to 43, a 37% improvement in the most costly error type. Overall, Random Forest provides meaningfully better predictions with more balanced class performance, justifying its recommendation despite the slightly larger training-validation gap.

# %% [markdown]
# ---
# ## Part 3: Model Comparison

# %% [markdown]
# ### Q1: Compare the Models

# %%
# Comparison table summarizing key metrics for both models
comparison = pd.DataFrame({
    'Model': ['Decision Tree', 'Random Forest'],
    'Training Acc': [train_acc, train_acc_rf],
    'Validation Acc': [val_acc, val_acc_rf],
    'Overfitting Gap': [train_acc - val_acc, train_acc_rf - val_acc_rf]
})

# Display the table without row indices for readability
print("Model Comparison:")
print(comparison.to_string(index=False))

# %%
# Visualization comparing accuracy and overfitting for both models
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy comparison: bars for training vs validation to check generalization
x = np.arange(len(comparison))
width = 0.35
axes[0].bar(x - width/2, comparison['Training Acc'], width, label='Training', alpha=0.8)
axes[0].bar(x + width/2, comparison['Validation Acc'], width, label='Validation', alpha=0.8)
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Model Accuracy Comparison')
axes[0].set_xticks(x)
axes[0].set_xticklabels(comparison['Model'])
axes[0].legend()
axes[0].set_ylim([0, 1.05])

# Overfitting comparison: visualize the train-val gap for each model
axes[1].bar(comparison['Model'], comparison['Overfitting Gap'], alpha=0.8)
axes[1].set_ylabel('Training - Validation Accuracy')
axes[1].set_title('Overfitting Analysis')
axes[1].axhline(0, color='black', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/model_comparison.png', dpi=300, bbox_inches='tight')  # Save plot for reuse
plt.show()

# %% [markdown]
# ### Q2: Model Recommendation

# %%
print("RECOMMENDATION: Random Forest\n")

print("Reasons:")
print(f"1. Higher validation accuracy: {val_acc_rf:.2%} vs {val_acc:.2%}")
print(f"2. Better generalization: {(train_acc_rf - val_acc_rf)*100:.1f}% gap vs {(train_acc - val_acc)*100:.1f}%")
print(f"3. Ensemble approach reduces variance and improves robustness")
print(f"4. More stable predictions across different data samples")
print(f"\nTrade-off: Less interpretable than single tree, but performance gain justifies this.")

# %% [markdown]
# ## Recommendation
# 
# Random Forest is the superior model, achieving 73.95% validation accuracy versus Decision Tree's 69.13% a meaningful 4.82-point improvement that translates to 15 fewer errors. 
# 
# While Random Forest shows a larger overfitting gap (16.16% vs 9.79%), this reflects better training data fit rather than worse generalization; validation performance proves Random Forest generalizes better. 
# 
# Critically, Random Forest reduces false negatives from 68 to 43 (37% improvement) and achieves balanced class performance (72-76% recall) versus Decision Tree's imbalance (56-82%). The ensemble approach provides robust, stable predictions through averaging 100 trees. Unless interpretability is absolutely critical, Random Forest's superior validation accuracy and balanced performance make it the clear choice for practical yield prediction.


