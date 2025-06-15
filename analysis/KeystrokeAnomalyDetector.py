import numpy as np

# Models
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# Some tools
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, roc_curve

from KeystrokeDataExtractor import KeystrokeDataExtractor

class KeystrokeAnomalyDetector:
    def __init__(self, model_name='KNN', model_params=None, scaler_enabled=True):
        self.model_name = model_name
        self.model_params = model_params or {}
        self.scaler_enabled = scaler_enabled
        self.model = None
        self.scaler = StandardScaler() if scaler_enabled else None
        self.X_train = None
        self.X_test = None
        self.y_true = None
        self.eval_results = {}
    
    def _create_model(self):
        if self.model_name == 'OneClassSVM':
            params = {'gamma': 'auto'}
            params.update(self.model_params)
            return OneClassSVM(**params)

        elif self.model_name == 'IsolationForest':
            params = {'random_state': 42, 'n_estimators': 100}
            params.update(self.model_params)
            return IsolationForest(**params)

        elif self.model_name == 'LOF':
            params = {
                'n_neighbors': 4,
                'metric': 'minkowski',
                'novelty': True,
                'p': 1
            }
            params.update(self.model_params)
            return LocalOutlierFactor(**params)

        elif self.model_name == 'KNN':
            params = {'n_neighbors': 6, 'metric': 'minkowski'}
            params.update(self.model_params)
            # KNN для NearestNeighbors
            return NearestNeighbors(**params)

        else:
            raise ValueError(f"Unknown model: {self.model_name}")

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

    def get_train_data(
        self, 
        extractor_path, 
        target_subject
    ):
        if self.X_train is None: 
            self.load_data_from_extractor(extractor_path, target_subject)
        return self.X_train
    
    def _generate_anomalous_data(self, X, n_samples):
        n_features = X.shape[1]

        return np.array([
            [np.random.choice(X[:, j]) for j in range(n_features)]
            for _ in range(n_samples)
        ], dtype=X.dtype)

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
        self.model = self._create_model()
        self.model.fit(self.X_train)

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

    def get_report(self):
        return self.eval_results.copy()

    def evaluate(self, y_pred, scores=None):
        self.y_true = (self.y_true == 1).astype(int)
        y_pred = (y_pred == 1).astype(int)
        
        if scores is not None:
            if self.model_name == 'KNN':
                scores = -scores

            roc_auc = roc_auc_score(self.y_true, scores)
            accuracy = accuracy_score(self.y_true, y_pred)
            conf_matrix = confusion_matrix(self.y_true, y_pred)

            fpr, tpr, thresholds = roc_curve(self.y_true, scores)
            fnr = 1 - tpr

            # EER — point where FPR = FNR (or nearest to this)
            eer_threshold_index = np.nanargmin(np.absolute(fnr - fpr))
            eer = (fpr[eer_threshold_index] + fnr[eer_threshold_index]) / 2

            # ZM-FAR — FAR при FRR = 0 (то есть TPR = 1)
            zmfar_index = np.where(tpr == 1.0)[0]
            zmfar = fpr[zmfar_index[0]] if len(zmfar_index) > 0 else None

            # Save evaluation
            self.eval_results['roc_auc'] = roc_auc
            self.eval_results['accuracy'] = accuracy
            self.eval_results['eer'] = eer
            self.eval_results['fpr_list'] = fpr
            self.eval_results['tpr_list'] = tpr
            self.eval_results['fpr'] = fpr[eer_threshold_index]
            self.eval_results['tpr'] = tpr[eer_threshold_index]
            self.eval_results['y_pred'] = y_pred
            self.eval_results['far'] = fpr[eer_threshold_index]
            self.eval_results['frr'] = fnr[eer_threshold_index]
            self.eval_results['zmfar'] = zmfar
            self.eval_results['confusion_matrix'] = conf_matrix
        else:
            print("EER cannot be calculated without probability scores.")

    def run_pipeline(
        self, 
        extractor_path, 
        target_subject, impostors, 
        n_test_legit, n_test_impostors_each
    ):
        self.load_data_from_extractor(
            extractor_path, target_subject, impostors, 
            n_test_legit, n_test_impostors_each
        )
        self.scale_data()
        self.fit_model()
        y_pred, scores = self.predict()
        self.evaluate(y_pred, scores)

    def run_validation(self, extractor_path, target_subject, n_test_legit, n_test_anomaly, n_test_impostors_each, impostors):
        # Get legit data
        extractor = KeystrokeDataExtractor(extractor_path)
        data_target = extractor.get_subject_data(target_subject)

        # Get legit training data
        self.X_train = data_target['first_15'].to_numpy()

        # Get legit (for testing)
        X_test_legit = data_target['remaining'][:n_test_legit].to_numpy()
        y_true_legit = [1] * len(X_test_legit)

        # Get anomaly (for testing)
        X_test_anomaly = self._generate_anomalous_data(data_target['remaining'][n_test_legit:].to_numpy(), n_test_anomaly)
        y_true_anomaly = [0] * len(X_test_anomaly)

        # Get impostors (for testing)
        extractor = KeystrokeDataExtractor(extractor_path)
        X_test_impostors = []
        y_true_impostors = []

        for impostor in impostors:
            data_subject = extractor.get_subject_data(impostor)

            if len(data_subject['first_15']) == 0: 
                continue

            X = data_subject['first_15'][:n_test_impostors_each].to_numpy()
            X_test_impostors.append(X)
            y_true_impostors.extend([0] * len(X))

        X_test_impostors = np.vstack(X_test_impostors)
        y_true_impostors = np.array(y_true_impostors)
        
        # Combining data
        self.X_test = np.vstack([X_test_legit, X_test_anomaly, X_test_impostors])
        self.y_true = np.concatenate([y_true_legit, y_true_anomaly, y_true_impostors])

        self.fit_model()
        y_pred, scores = self.predict()
        self.evaluate(y_pred, scores)