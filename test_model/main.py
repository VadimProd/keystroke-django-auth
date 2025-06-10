import pandas as pd
import numpy as np
import subprocess
import os
import pandas as pd


from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from KeystrokeAnomalyDetector import KeystrokeAnomalyDetector
from typing import Literal
from tqdm import tqdm


def create_csv(data: np.array, filename: str, mode: Literal['train', 'test']) -> None:
    if filename[-3:] != 'csv':
        print("Error: incorrect file format")
        exit(0)

    column_names = [f"event_{i}" for i in range(data.shape[1])]
    df = pd.DataFrame(data, columns=column_names)

    # Add column for training csv-file
    if mode == 'train':
        df.insert(0, "repetition", range(0, len(df)))

    df.to_csv(f"datasets/{filename}", index=False)

if __name__ == '__main__':
    mode = ['OneClassSVM', 'IsolationForest', 'LOF', 'KNN']
    impostors = [f's00{i}' for i in range (3, 10, 1)]
    impostors.extend(
        [f's0{i}' for i in range (10, 58, 1)]
    )

    #
    # Init model
    #

    detector = KeystrokeAnomalyDetector(model_name='LOF', scaler_enabled=False)
    detector.run_pipeline(
        extractor_path='~/Desktop/learning/8_sem/NIR/keystroke_django_auth/test_model/datasets/DSL-StrongPasswordData.csv',
        target_subject='s002',
        impostors=impostors,
        n_test_legit=100,
        n_test_impostors_each=2
    )

    exit(0)

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
        ["Rscript", "R-scripts/trainer.R"],
        stdout=subprocess.DEVNULL,
    )

    if result.returncode == 0:
        print("Successfully trained")
    else:
        print("Error:", result.returncode)

    os.remove('datasets/training_data.csv')

    #
    # Get test data
    #

    X_test, y_true = detector.get_test_data(
        extractor_path='datasets/DSL-StrongPasswordData.csv',
        target_subject='s002',
        impostors=impostors,
        n_test_legit=50,
        n_test_impostors_each=5
    )
    create_csv(data=X_test, filename='current_attempt.csv', mode='test')

    #
    # Run auth script
    #
    
    result = subprocess.run(
        ["Rscript", "R-scripts/authenticator.R"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True
    )

    #
    # Evaluate output
    #

    output_lines = result.stdout.strip().splitlines()
    cutoff = 800.0

    y_pred = []
    for prob in tqdm(output_lines, desc="Processing predictions"):
        score = float(prob)
        percent_score = min(score / cutoff, 1.0)
        is_real_user = percent_score < 1.0
        y_pred.append(1 if is_real_user else 0)
    
    print(f"{classification_report(y_true, y_pred)}")
    print(f"ROC-AUC: {roc_auc_score(y_pred, y_true):.2f}")
