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

    #
    # Init model
    #

    detector = KeystrokeAnomalyDetector(model_name='KNN', scaler_enabled=False)
    # detector.run_pipeline('DSL-StrongPasswordData.csv')

    #
    # Get data (training vector for the model)
    #  

    X_train = detector.get_train_data('datasets/DSL-StrongPasswordData.csv')

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

    X_test, y_true = detector.get_test_data('datasets/DSL-StrongPasswordData.csv')

    y_pred = []
    for i in tqdm(range(len(X_test)), desc="Predict..."):
        create_csv(data=X_test[i].reshape(1, -1), filename='current_attempt.csv', mode='test')

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
        last_line = output_lines[-1] if output_lines else None

        score = float(last_line)
        cutoff = 800.0
        percent_score = min(score / cutoff, 1.0)
        is_real_user = percent_score < 1.0
        y_pred.append(1 if is_real_user else -1)
        
        # print(f"{percent_score:.3f}")
    
    accuracy = accuracy_score(y_pred, y_true)
    roc_auc = roc_auc_score(y_pred, y_true)
    print(f"{classification_report(y_true, y_pred)}")

