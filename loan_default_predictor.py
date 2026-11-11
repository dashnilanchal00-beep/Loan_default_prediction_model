# ALL IMPORTS 

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score,
    matthews_corrcoef,
    average_precision_score
)

warnings.filterwarnings('ignore', category=FutureWarning)

# Using threshold as 0.3 instead of default 0.5

threshold = 0.30

# Reading Dataset 

df = pd.read_csv('loan_dataset.csv')

pd.set_option('display.max_columns', None)

print(df.shape)
print(df.head())
print(df.columns)
print(df['loan_status'].value_counts())

drop_cols = [
    'id', 'member_id', 'url', 'desc',
    'emp_title', 'title', 'zip_code',
    'issue_d',
    'out_prncp', 'out_prncp_inv',
    'total_pymnt', 'total_pymnt_inv',
    'total_rec_prncp', 'total_rec_int',
    'total_rec_late_fee', 'recoveries',
    'collection_recovery_fee',
    'last_pymnt_d', 'last_pymnt_amnt',
    'last_credit_pull_d', 'grade', 'addr_state',
    'funded_amnt', 'funded_amnt_inv'
]

# Dropping unneceseery or Data leakage columns 

df = df.drop(columns=drop_cols)

print(df.columns)
print(df.describe())

missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

print(df['emp_length'].value_counts(dropna=False))

df['emp_length'] = df['emp_length'].astype('object')
df['emp_length'] = df['emp_length'].fillna('Unknown')

df = df.drop(columns=[ 'mths_since_last_record', 'mths_since_last_delinq' ])

print(df['pub_rec_bankruptcies'].value_counts(dropna=False).sort_index())
print(df['sub_grade'].value_counts(dropna=False).sort_index())

df['pub_rec_bankruptcies'] = df['pub_rec_bankruptcies'].fillna(0)

# Converting columns into proper formats 

df['earliest_cr_line'] = pd.to_datetime(
    df['earliest_cr_line'], 
    format='%b-%y',
    errors='coerce'
)

missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])
print(df.dtypes)
print(df['earliest_cr_line'].head(10))

df['int_rate'] = df['int_rate'].str.replace('%', '').astype(float)
df['revol_util'] = df['revol_util'].str.replace('%', '').astype(float)

categorical_cols = [
    'term',
    'sub_grade',
    'emp_length',
    'home_ownership',
    'verification_status',
    'purpose'
]

df[categorical_cols] = df[categorical_cols].astype('category')

print(df.dtypes)
print(df.describe())

# Making X and y 

X = df.drop(columns=['loan_status'])
y = df['loan_status']

scaling_cols = [
    'loan_amnt',
    'int_rate',
    'installment',
    'annual_inc',
    'dti',
    'delinq_2yrs',
    'inq_last_6mths',
    'open_acc',
    'pub_rec',
    'revol_bal',
    'revol_util',
    'total_acc',
    'pub_rec_bankruptcies'
]

onehot_cols = [
    'term',
    'home_ownership',
    'verification_status',
    'purpose'
]

ordinal_cols = [
    'sub_grade',
    'emp_length'
]

ordinal_categories = [
    [
        'A1', 'A2', 'A3', 'A4', 'A5',
        'B1', 'B2', 'B3', 'B4', 'B5',
        'C1', 'C2', 'C3', 'C4', 'C5',
        'D1', 'D2', 'D3', 'D4', 'D5',
        'E1', 'E2', 'E3', 'E4', 'E5',
        'F1', 'F2', 'F3', 'F4', 'F5',
        'G1', 'G2', 'G3', 'G4', 'G5'
    ],
    [
        '< 1 year', '1 year', '2 years', '3 years', '4 years',
        '5 years', '6 years', '7 years', '8 years', '9 years',
        '10+ years', 'Unknown'
    ]
]

model_names = [
    'Logistic Regression',
    'Naive Bayes',
    'Decision Tree',
    'Random Forest',
    'Gradient Boosting'
]

metrics = [
    'Train Accuracy',
    'Test Accuracy',
    'AUC',
    'Recall',
    'Precision',
    'F1',
    'Balanced Accuracy',
    'Specificity',
    'MCC',
    'Average Precision'
]

results = {}
conf_matrices = {}

for name in model_names:
    results[name] = {}
    for metric in metrics:
        results[name][metric] = []
    conf_matrices[name] = []

seeds_used = []

for state in range(0, 50, 5):

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=state,
        stratify=y
    )
    
    # Coping to avoid splitting error 
    
    X_train = X_train.copy()
    X_test = X_test.copy()

    # Filling missing revol_util values using the training data median
    revol_util_median = X_train['revol_util'].median()

    X_train['revol_util'] = X_train['revol_util'].fillna(revol_util_median)
    X_test['revol_util'] = X_test['revol_util'].fillna(revol_util_median)

    X_train_scale = X_train[scaling_cols]
    X_train_onehot = X_train[onehot_cols]
    X_train_ordinal = X_train[ordinal_cols]

    X_test_scale = X_test[scaling_cols]
    X_test_onehot = X_test[onehot_cols]
    X_test_ordinal = X_test[ordinal_cols]

    # scaling number columns
    scaler = StandardScaler()

    X_train_scale = scaler.fit_transform(X_train_scale)
    X_test_scale = scaler.transform(X_test_scale)

    # one hot encoding to catagory columns
    onehot_encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown='ignore'
    )

    X_train_onehot = onehot_encoder.fit_transform(X_train_onehot)
    X_test_onehot = onehot_encoder.transform(X_test_onehot)

    # ordinal encoding ordered to catagory columns
    ordinal_encoder = OrdinalEncoder(
        categories=ordinal_categories,
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )

    X_train_ordinal = ordinal_encoder.fit_transform(X_train_ordinal)
    X_test_ordinal = ordinal_encoder.transform(X_test_ordinal)

    # joining the all processed columns
    X_train_final = np.hstack((
        X_train_scale,
        X_train_onehot,
        X_train_ordinal
    ))

    X_test_final = np.hstack((
        X_test_scale,
        X_test_onehot,
        X_test_ordinal
    ))

    # creating models and setting their parameters
    models = {
        'Logistic Regression': LogisticRegression(
            C=1.0,
            solver='lbfgs',
            class_weight='balanced',
            max_iter=10000,
            random_state=42
        ),

        'Naive Bayes': GaussianNB(
            var_smoothing=1e-8
        ),

        'Decision Tree': DecisionTreeClassifier(
            criterion='gini',
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=42
        ),

        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),

        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42
        )
    }

    seeds_used.append(state)

    print(f'\n================ Seed {state} ================')

    for name, model in models.items():

        model.fit(X_train_final, y_train)

        # getting probability for class 1
        train_probabilities = model.predict_proba(X_train_final)[:, 1]
        test_probabilities = model.predict_proba(X_test_final)[:, 1]

        # appling 0.3 instead of default 0.5 threshold
        train_predictions = (train_probabilities >= threshold).astype(int)
        test_predictions = (test_probabilities >= threshold).astype(int)

        # calculating the accuracies
        train_accuracy = np.mean(train_predictions == y_train.to_numpy())
        test_accuracy = np.mean(test_predictions == y_test.to_numpy())

        # AUC and average precision use probability scores
        auc_score = roc_auc_score(y_test, test_probabilities)

        # calculating the confusion matrix in [0,1] order
        cm = confusion_matrix(
            y_test,
            test_predictions,
            labels=[0, 1]
        )

        tn, fp, fn, tp = cm.ravel()

        if (tn + fp) > 0:
            specificity = tn / (tn + fp)
        else:
            specificity = 0.0

        recall = recall_score(
            y_test,
            test_predictions,
            pos_label=1,
            zero_division=0
        )

        precision = precision_score(
            y_test,
            test_predictions,
            pos_label=1,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            test_predictions,
            pos_label=1,
            zero_division=0
        )

        balanced_accuracy = balanced_accuracy_score(
            y_test,
            test_predictions
        )

        mcc = matthews_corrcoef(y_test, test_predictions)

        average_precision = average_precision_score(
            y_test,
            test_probabilities
        )

        # saving the results
        
        results[name]['Train Accuracy'].append(train_accuracy)
        results[name]['Test Accuracy'].append(test_accuracy)
        results[name]['AUC'].append(auc_score)
        results[name]['Recall'].append(recall)
        results[name]['Precision'].append(precision)
        results[name]['F1'].append(f1)
        results[name]['Balanced Accuracy'].append(balanced_accuracy)
        results[name]['Specificity'].append(specificity)
        results[name]['MCC'].append(mcc)
        results[name]['Average Precision'].append(average_precision)

        conf_matrices[name].append(cm)

        print(f'\n{name}')
        print(f'  Train Accuracy: {train_accuracy:.4f}')
        print(f'  Test Accuracy: {test_accuracy:.4f}')
        print(f'  AUC: {auc_score:.4f}')
        print(f'  Recall: {recall:.4f}')
        print(f'  Precision: {precision:.4f}')
        print(f'  F1 Score: {f1:.4f}')
        print(f'  Balanced Accuracy: {balanced_accuracy:.4f}')
        print(f'  Specificity: {specificity:.4f}')
        print(f'  MCC: {mcc:.4f}')
        print(f'  Average Precision: {average_precision:.4f}')
        print(f'  Confusion Matrix [[TN FP] [FN TP]]:\n{cm}')

    print(f'The comparison for seed {state} is completed.')

# printing avarages

print('\nAverage scores across seeds')

for model_name in model_names:
    print(f'\n{model_name}')

    for metric in metrics:
        average_score = np.mean(results[model_name][metric])
        print(f'  Average {metric}: {average_score:.4f}')

# printing highest sdcores

print('\nWinner by average score')

for metric in metrics:
    best_model = model_names[0]
    best_score = np.mean(results[best_model][metric])

    for model_name in model_names[1:]:
        average_score = np.mean(results[model_name][metric])
        if average_score > best_score:
            best_score = average_score
            best_model = model_name

    print(f'{metric}: {best_model} ({best_score:.4f})')

# counting wins separately

print('\nNumber of seed wins for each metric')

for metric in metrics:
    win_counts = {}
    for model_name in model_names:
        win_counts[model_name] = 0

    for seed_index in range(len(seeds_used)):
        best_score = results[model_names[0]][metric][seed_index]

        for model_name in model_names[1:]:
            score = results[model_name][metric][seed_index]
            if score > best_score:
                best_score = score

        for model_name in model_names:
            score = results[model_name][metric][seed_index]
            if np.isclose(score, best_score):
                win_counts[model_name] += 1

    print(f'\n{metric} seed wins (out of {len(seeds_used)})')

    for model_name in model_names:
        print(f'  {model_name}: {win_counts[model_name]}')

# printing confusion metrics

print('\nConfusion matrix summed across all seeds [[TN FP] [FN TP]]')

for model_name in model_names:
    print(f'\n{model_name}')
    print(np.sum(conf_matrices[model_name], axis=0))

# printing avarage confusion metric

print('\nConfusion matrix averaged across seeds')

for model_name in model_names:
    print(f'\n{model_name}')
    print(np.mean(conf_matrices[model_name], axis=0).round(1))

# calculating avg AUC and recall

avg_auc = []
avg_recall = []

for model_name in model_names:
    avg_auc.append(np.mean(results[model_name]['AUC']))
    avg_recall.append(np.mean(results[model_name]['Recall']))

# Avarage AUC chart

plt.figure(figsize=(10, 5))
bars = plt.bar(model_names, avg_auc, color='steelblue')
plt.title('Average AUC by Model')
plt.xlabel('Model')
plt.ylabel('Average AUC')
plt.ylim(0, 1)
plt.xticks(rotation=30, ha='right')

for bar, score in zip(bars, avg_auc):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.015,
        f'{score:.3f}',
        ha='center'
    )

plt.tight_layout()
plt.show()

# Avarage recall chart

plt.figure(figsize=(10, 5))
bars = plt.bar(model_names, avg_recall, color='darkorange')
plt.title('Average Recall by Model')
plt.xlabel('Model')
plt.ylabel('Average Recall')
plt.ylim(0, 1)
plt.xticks(rotation=30, ha='right')

for bar, score in zip(bars, avg_recall):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.015,
        f'{score:.3f}',
        ha='center'
    )

plt.tight_layout()
plt.show()
