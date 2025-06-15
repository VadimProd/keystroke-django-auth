import pandas as pd
import numpy as np
import subprocess
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, \
    confusion_matrix, roc_curve, ConfusionMatrixDisplay
from KeystrokeAnomalyDetector import KeystrokeAnomalyDetector
from typing import Literal
from tqdm import tqdm

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(DIR_PATH, 'datasets/DSL-StrongPasswordData.csv')

def create_csv(data: np.array, filename: str, mode: Literal['train', 'test']) -> None:
    if filename[-3:] != 'csv':
        print("Error: incorrect file format")
        exit(0)

    column_names = [f"event_{i}" for i in range(data.shape[1])]
    df = pd.DataFrame(data, columns=column_names)

    # Add column for training csv-file
    if mode == 'train':
        df.insert(0, "repetition", range(0, len(df)))
    
    full_path = os.path.join(DIR_PATH, "datasets", filename)
    df.to_csv(full_path, index=False)

def evaluate_another_project():
    #
    # Get data (training vector for the model)
    #  

    X_train = detector.get_train_data('datasets/DSL-StrongPasswordData.csv', target_subject='s002')

    #
    # Save it for R-script in the required format
    #

    create_csv(data=X_train, filename='training_data.csv', mode='train')

    #
    # Run trainer.R script
    #

    result = subprocess.run(
        ["Rscript", os.path.join(DIR_PATH, 'R-scripts/trainer.R')],
        stdout=subprocess.DEVNULL,
    )

    if result.returncode == 0:
        print("Successfully trained")
    else:
        print("Error:", result.returncode)

    os.remove('/Users/vadimnaumov/Desktop/learning/8_sem/NIR/keystroke-django-auth/analysis/datasets/training_data.csv')

    #
    # Get test data
    #

    X_test, y_true = detector.get_test_data(
        extractor_path=DATASET_PATH,
        target_subject='s002',
        impostors=impostors,
        n_test_legit=50,
        n_test_impostors_each=1
    )
    create_csv(data=X_test, filename='current_attempt.csv', mode='test')

    #
    # Run auth script
    #

    result = subprocess.run(
        ["Rscript", os.path.join(DIR_PATH, 'R-scripts/authenticator.R')],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True
    )

    #
    # Evaluate output
    #

    output_lines = result.stdout.strip().splitlines()

    # According to the PHP (another project)
    cutoff = 800.0
    raw_scores = [float(prob) for prob in output_lines]
    scores = [min(s / cutoff, 1.0) for s in raw_scores]
    scores = [1.0 - s for s in scores]

    # ROC and EER
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2
    cutoff = thresholds[eer_index]

    # Predictions: >= as higher is better
    y_pred = np.array([int(s >= cutoff) for s in scores])

    res = {
        'roc_auc': roc_auc_score(y_true, scores),
        'accuracy': accuracy_score(y_true, y_pred),
        'eer': eer,
        'fpr_list': fpr,
        'tpr_list': tpr,
        'fpr': fpr[eer_index],
        'tpr': tpr[eer_index],
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    return res

if __name__ == '__main__':
    mode = ['OneClassSVM', 'IsolationForest', 'LOF', 'KNN']
    impostors = [f's00{i}' for i in range (3, 10, 1)]
    impostors.extend(
        [f's0{i}' for i in range (10, 58, 1)]
    )

    #
    # My model
    #

    detector = KeystrokeAnomalyDetector(
        model_name='IsolationForest',
        model_params={'n_estimators': 90},
        scaler_enabled=False
    )
    detector.run_pipeline(
        extractor_path=DATASET_PATH,
        target_subject='s002',
        impostors=impostors,
        n_test_legit=50,
        n_test_impostors_each=1
    )

    res_my_model = detector.get_report()
    
    #
    # Another project
    #
    
    res_another = evaluate_another_project()

    #
    # Comparasion
    #

    # Print report as table
    method_list = ["My solution", "Another solution"]
    accuracy_list = [res_my_model['accuracy'], res_another['accuracy']]
    auc_list = np.array([res_my_model['roc_auc'], res_another['roc_auc']])
    eer_list = np.array([res_my_model['eer'], res_another['eer']])
    tpr_list = np.array([res_my_model['tpr'], res_another['tpr']])
    fpr_list = np.array([res_my_model['fpr'], res_another['fpr']])


    table_data = {
        "Метод": method_list,
        "Accuracy": accuracy_list,
        "AUC, %": np.round(auc_list * 100),
        "EER, %": np.round(eer_list * 100),
        "TPR, %": np.round(tpr_list * 100),
        "FPR, %": np.round(fpr_list * 100)
    }

    df = pd.DataFrame(table_data)
    print(df)
    print(res_my_model['confusion_matrix'])


    # Build confusion matrix
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # My
    cmd_my = ConfusionMatrixDisplay(confusion_matrix=res_my_model['confusion_matrix'], display_labels=[0, 1])
    cmd_my.plot(cmap=plt.cm.Blues, ax=ax[0])
    ax[0].set_title('Матрица ошибок разработанного решения')

    # Another
    cmd_another = ConfusionMatrixDisplay(confusion_matrix=res_another['confusion_matrix'], display_labels=[0, 1])
    cmd_another.plot(cmap=plt.cm.Reds, ax=ax[1])
    ax[1].set_title('Матрица ошибок другого решения')

    plt.tight_layout()
    plt.show()

    # Build ROC-curve
    print(df.to_string(index=False))

    plt.figure(figsize=(7, 5))
    plt.plot(
        res_my_model['fpr_list'], res_my_model['tpr_list'], 
        label=f'Мое решение (AUC = {(res_my_model['roc_auc']):.2f})'
    )
    plt.plot(
        res_another['fpr_list'], res_another['tpr_list'], 
        label=f'Другое решение (AUC = {(res_another["roc_auc"]):.2f})'
    )
    plt.plot([0, 1], [0, 1], 'k--', label='Случайный классификатор')

    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'ROC-кривые')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
