import psycopg2
import numpy as np
import matplotlib.pyplot as plt

# Models
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors, KNeighborsRegressor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.special import expit

# Some tools
from sklearn.metrics import classification_report, confusion_matrix, \
    roc_auc_score, accuracy_score, roc_curve

from KeystrokeDataExtractor import KeystrokeDataExtractor

class KeystrokeAnomalyDetector:
    def __init__(self, model_name='KNN', scaler_enabled=True):
        self.model_name = model_name
        self.scaler_enabled = scaler_enabled
        self.model = None
        self.scaler = StandardScaler() if scaler_enabled else None
        self.X_train = None
        self.X_test = None
        self.y_true = None
        self.eval_results = {}

    def extract_features(self, timing_str, mode='full'):
        parts = timing_str.split('\t')
        features = []
        for i in range(1, len(parts), 2):
            try:
                value = float(parts[i])
            except (ValueError, IndexError):
                value = 0.0
            features.append(value)

        hold = [features[i] for i in range(0, len(features), 3)]
        kd_kd = [features[i] for i in range(1, len(features), 3)]
        kd_ku = [features[i] for i in range(2, len(features), 3)]

        if mode != 'full':
            return [np.mean(hold), np.mean(kd_kd), np.mean(kd_ku)]
        return features

    def load_data_from_extractor(
        self, 
        extractor_path, 
        target_subject='s002', 
        impostors=['s003', 's004'],
        n_test_legit=50,
        n_test_impostors_each=50
    ):
        # Get legit data
        extractor = KeystrokeDataExtractor(extractor_path)
        data_target = extractor.get_subject_data(target_subject)

        # Get legit training data
        X_train = data_target['first_15'].to_numpy()

        # Get legit testing data
        X_test = data_target['remaining'][:n_test_legit].to_numpy()
        y_true = [1] * len(X_test)

        for imp in impostors:
            # Get impostor testing data (first n_test_impostors_each pieces)
            imp_data = extractor.get_subject_data(imp)
            if len(imp_data['remaining']) == 0: continue
            
            X_imp = imp_data['remaining'][:n_test_impostors_each].to_numpy()
            X_test = np.vstack((X_test, X_imp))
            y_true += [-1] * len(X_imp)

        self.X_train = X_train
        self.X_test = X_test
        self.y_true = np.array(y_true)
        print(y_true.count(1), y_true.count(-1))

    def get_train_data(
        self, 
        extractor_path, 
        target_subject
    ):
        if self.X_train is None: 
            self.load_data_from_extractor(extractor_path, target_subject)
        return self.X_train
    
    def get_test_data(
        self, 
        extractor_path, 
        target_subject, impostors, 
        n_test_legit, n_test_impostors_each
    ):
        if self.X_test is None:
            self.load_data_from_extractor(
                extractor_path, target_subject, impostors, n_test_legit, n_test_impostors_each
            )
        return self.X_test, self.y_true

    def scale_data(self):
        if self.scaler_enabled:
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)

    def fit_model(self):
        if self.model_name == 'OneClassSVM':
            self.model = OneClassSVM(gamma='auto')
            self.model.fit(self.X_train)

        elif self.model_name == 'IsolationForest':
            self.model = IsolationForest(random_state=42, n_estimators=100)
            self.model.fit(self.X_train)

        elif self.model_name == 'LOF':
            self.model = LocalOutlierFactor(
                n_neighbors=4,
                contamination=0.05,
                novelty=True,
                metric='minkowski',
                leaf_size=30,
                p=2
            )
            self.model.fit(self.X_train)

        elif self.model_name == 'KNN':
            self.model = NearestNeighbors(n_neighbors=4, metric='manhattan')
            self.model.fit(self.X_train)

        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def predict(self):
        if self.model_name == 'KNN':
            val_distances, _ = self.model.kneighbors(self.X_test)
            anomaly_scores = val_distances.mean(axis=1)
            train_distances, _ = self.model.kneighbors(self.X_train)
            threshold = np.percentile(train_distances.mean(axis=1), 95)
            y_pred = np.array([1 if s < threshold else -1 for s in anomaly_scores])
            return y_pred, anomaly_scores

        else:
            y_pred = self.model.predict(self.X_test)
            decision_scores = self.model.decision_function(self.X_test)
            return y_pred, decision_scores

    def evaluate(self, y_pred, scores=None):
        self.y_true = (self.y_true == 1).astype(int)
        y_pred = (y_pred == 1).astype(int)
        
        if scores is not None:
            if self.model_name == 'KNN':
                scores = -scores

            roc_auc = roc_auc_score(self.y_true, scores)
            print(f"ROC-AUC {roc_auc}")
            print(f"Accuracy {accuracy_score(self.y_true, y_pred)}")
            print(f"Confusion matrix \n{confusion_matrix(self.y_true, y_pred)}")
            print(f"Classification report \n{classification_report(self.y_true, y_pred)}")

            fpr, tpr, thresholds = roc_curve(self.y_true, scores)
            fnr = 1 - tpr

            # EER — point where FPR = FNR (or nearest to this)
            eer_threshold_index = np.nanargmin(np.absolute(fnr - fpr))
            eer = (fpr[eer_threshold_index] + fnr[eer_threshold_index]) / 2
            print(f"EER: {eer:.4f} at threshold {thresholds[eer_threshold_index]:.4f}")

            # Build ROC curve
            plt.figure(figsize=(6, 4))
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
            plt.scatter(fpr[eer_threshold_index], tpr[eer_threshold_index], color='red', label=f'EER ≈ {eer:.2f}')
            plt.xlabel('False Positive Rate (FPR)')
            plt.ylabel('True Positive Rate (TPR)')
            plt.title(f'ROC-кривая модели {self.model_name}')
            plt.legend(loc='lower right')
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            self.eval_results['roc_auc'] = roc_auc
            self.eval_results['eer'] = eer
            self.eval_results['fpr'] = fpr
            self.eval_results['tpr'] = tpr
            self.eval_results['y_pred'] = y_pred
        else:
            print("EER cannot be calculated without probability scores.")
        # if scores is not None:
        #     probabilities = expit(scores)
        #     for i, (score, prob) in enumerate(zip(scores, probabilities)):
        #         status = "Принято (настоящий пользователь)" if y_pred[i] == 1 else "Отклонено (возможный злоумышленник)"
        #         print(f"Образец {i}: оценка = {score:.4f}, вероятность = {prob:.2%} — {status}")

    def run_pipeline(
        self, 
        extractor_path, 
        target_subject, impostors, 
        n_test_legit, n_test_impostors_each
    ):
        self.load_data_from_extractor(
            extractor_path, target_subject, impostors, n_test_legit, n_test_impostors_each
        )
        self.scale_data()
        self.fit_model()
        y_pred, scores = self.predict()
        self.evaluate(y_pred, scores)