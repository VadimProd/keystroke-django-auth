import psycopg2
import numpy as np

# Models
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.special import expit

# Some tools
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

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

    def load_data_from_extractor(self, extractor_path, target_subject='s002', impostors=['s003', 's004']):
        extractor = KeystrokeDataExtractor(extractor_path)
        data_target = extractor.get_subject_data(target_subject)

        X_train = data_target['first_15'].to_numpy()
        X_test = data_target['remaining'][:50].to_numpy()
        y_true = [1] * len(X_test)

        for imp in impostors:
            imp_data = extractor.get_subject_data(imp)
            
            X_imp = imp_data['remaining'][:50].to_numpy()
            X_test = np.vstack((X_test, X_imp))
            y_true += [-1] * len(X_imp)

        self.X_train = X_train
        self.X_test = X_test
        self.y_true = np.array(y_true)

    def get_train_data(self, extractor_path):
        if self.X_train is None: 
            self.load_data_from_extractor(extractor_path)
        return self.X_train
    
    def get_test_data(self, extractor_path):
        if self.X_test is None:
            self.load_data_from_extractor(extractor_path)
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
            self.model = IsolationForest(random_state=42)
            self.model.fit(self.X_train)

        elif self.model_name == 'LOF':
            self.model = LocalOutlierFactor(
                n_neighbors=15,
                contamination=0.05,
                novelty=True,
                metric='minkowski',
                leaf_size=30,
                p=2
            )
            self.model.fit(self.X_train)

        elif self.model_name == 'KNN':
            self.model = NearestNeighbors(n_neighbors=4)
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
        print(f"-> {self.model_name}")
        print(confusion_matrix(self.y_true, y_pred))
        # print(classification_report(self.y_true, y_pred))
        print(roc_auc_score(self.y_true, y_pred))
        # if scores is not None:
        #     probabilities = expit(scores)
        #     for i, (score, prob) in enumerate(zip(scores, probabilities)):
        #         status = "Принято (настоящий пользователь)" if y_pred[i] == 1 else "Отклонено (возможный злоумышленник)"
        #         print(f"Образец {i}: оценка = {score:.4f}, вероятность = {prob:.2%} — {status}")

    def run_pipeline(self, extractor_path):
        self.load_data_from_extractor(extractor_path)
        self.scale_data()
        self.fit_model()
        y_pred, scores = self.predict()
        self.evaluate(y_pred, scores)