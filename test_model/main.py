import psycopg2
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Модели
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.special import expit 
from sklearn.metrics import classification_report, confusion_matrix

DB_HOST = "localhost"
DB_PORT = 5431
DB_NAME = "keystroke_db"
DB_USER = "keystroke_user"
DB_PASS = "keystroke_pass"

TARGET_USER_ID = 1

def generate_artificial_anomalies(X_legitimate, n_samples=200, random_state=42):
    """
    Генерирует искусственные аномалии путем случайной выборки значений признаков 
    из легитимных данных пользователя (как в статье).
    
    Параметры:
        X_legitimate (np.array): Легитимные данные пользователя (форма [n_samples, n_features]).
        n_samples (int): Количество генерируемых аномальных образцов.
        random_state (int): Для воспроизводимости.
        
    Возвращает:
        np.array: Искусственные аномалии формы [n_samples, n_features].
    """
    np.random.seed(random_state)
    n_features = X_legitimate.shape[1]
    anomalies = np.zeros((n_samples, n_features))
    
    for feature_idx in range(n_features):
        # Берем случайные значения из легитимных данных для каждого признака
        anomalies[:, feature_idx] = np.random.choice(
            X_legitimate[:, feature_idx], 
            size=n_samples,
            replace=True
        )
    
    return anomalies

def extract_features_from_timing(timing_str, mode='full'):
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
        # np.mean(hold), np.std(hold),
        # np.mean(kd_kd), np.std(kd_kd),
        # np.mean(kd_ku), np.std(kd_ku)

        # print(features_mean)
        return [
            np.mean(hold), #np.std(hold), 
            np.mean(kd_kd),# np.std(kd_kd),
            np.mean(kd_ku),# np.std(kd_ku)
        ]
    else:
        return features

def load_user_data(user_id):
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )
    cursor = conn.cursor()
    cursor.execute("SELECT timing_data FROM keystroke_keystrokesample WHERE user_id = %s", (user_id,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [row[0] for row in rows]

def main():
    user_data = load_user_data(TARGET_USER_ID)
    if not user_data:
        print(f"Данные для пользователя с ID={TARGET_USER_ID} не найдены.")
        return

    X = [extract_features_from_timing(timing, 'full') for timing in user_data]
    X = np.array(X)

    if len(X) < 2:
        print("Недостаточно данных для обучения модели (нужно минимум 2 записи).")
        return

    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Генерация искусственных аномалий (имитация атак)
    X_artificial = scaler.transform(generate_artificial_anomalies(X_train_scaled, n_samples=10))
    # print(X_train_scaled[:5])
    # print(X_artificial)
    X_test_combined = np.vstack([X_test_scaled, X_artificial])
    y_true = np.array(
        [1] * len(X_test_scaled) + 
        [-1] * len(X_artificial)
    )

    # X_train_scaled = X_train
    # X_test_scaled = X_test

    # print(X_train_scaled)

    model = 2

    if model == 3:
        model = OneClassSVM(gamma='auto', nu=0.05)
        model.fit(X_train_scaled)
    elif model == 2:
        # model = IsolationForest(contamination=0.05, random_state=42, n_estimators=6)
        model = IsolationForest(contamination=0.05, random_state=42)
        model.fit(X_train_scaled)

    elif model == 1:
        for n in range(1, 15):
            # Обучаем LOF с возможностью оценки новых данных
            model = LocalOutlierFactor(
                n_neighbors=15,
                contamination=0.05,
                novelty=True,
                metric='minkowski',  # обычно работает хорошо
                leaf_size=30,        # оптимален для большинства случаев
                p=2                  # Евклидово расстояние
            )
            model.fit(X_train_scaled)

            y_pred = model.predict(X_test_scaled)  # 1 - inlier, -1 - outlier
            decision_scores = model.decision_function(X_test_scaled)
            probabilities = expit(decision_scores)

            print(f"[n_neighbors = {n}]")
            for i, (score, prob) in enumerate(zip(decision_scores, probabilities)):
                status = "Принято (настоящий пользователь)" if y_pred[i] == 1 else "Отклонено (возможный злоумышленник)"
                print(f"Образец {i}: оценка аномалии = {score:.4f}, вероятность легитимности = {prob:.2%} — {status}")
            print("\n")

    # y_pred = model.predict(X_test_scaled)
    # decision_scores = model.decision_function(X_test_scaled)

    vika = scaler.transform(np.array([[
        88.0, 233.0, 145.0,  
        72.0, 236.0, 164.0,  
        79.0, 308.0, 229.0,  
        77.0, 223.0, 146.0,  
        75.0, 242.0, 167.0,  
        86.0, 266.0, 180.0,  
        72.0, 218.0, 146.0,  
        74.0, 208.0, 134.0,  
        70.0, 272.0, 202.0,  
        96.0, 260.0, 164.0,  
        91.0
    ]]))
    print(vika)
    y_pred = model.predict(vika)
    decision_scores = model.decision_function(vika)

    # y_pred = model.predict(X_test_combined)
    # decision_scores = model.decision_function(X_test_combined)

    # print("\nРезультаты на тестовых данных + искусственных аномалиях:")
    # print(confusion_matrix(y_true, y_pred))
    # print(classification_report(y_true, y_pred))
    # print(y_true)


    # Переводим оценки в вероятности
    probabilities = expit(decision_scores)
    
    for i, (score, prob) in enumerate(zip(decision_scores, probabilities)):
        status = "Принято (настоящий пользователь)" if y_pred[i] == 1 else "Отклонено (возможный злоумышленник)"
        print(f"Образец {i}: оценка аномалии = {score:.4f}, вероятность легитимности = {prob:.2%} — {status}")

    
    # y_true = [1]*len(X_test) + [-1]*len(attacker_data)
    # y_pred = list(model.predict(X_test_scaled)) + list(model.predict(attacker_scaled))

    # print(confusion_matrix(y_true, y_pred))
    # print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    main()


'''
hold[q] 88.0    keydown[a]-keydown[q]   233.0   keydown[a]-keyup[q]     145.0   
hold[a] 72.0    keydown[z]-keydown[a]   236.0   keydown[z]-keyup[a]     164.0
hold[z] 79.0    keydown[w]-keydown[z]   308.0   keydown[w]-keyup[z]     229.0   
hold[w] 77.0    keydown[s]-keydown[w]   223.0   keydown[s]-keyup[w]     146.0   
hold[s] 75.0    keydown[x]-keydown[s]   242.0   keydown[x]-keyup[s]     167.0   
hold[x] 86.0    keydown[e]-keydown[x]   266.0   keydown[e]-keyup[x]     180.0   
hold[e] 72.0    keydown[d]-keydown[e]   218.0   keydown[d]-keyup[e]     146.0   
hold[d] 74.0    keydown[c]-keydown[d]   208.0   keydown[c]-keyup[d]     134.0   
hold[c] 70.0    keydown[1]-keydown[c]   272.0   keydown[1]-keyup[c]     202.0   
hold[1] 96.0    keydown[2]-keydown[1]   260.0   keydown[2]-keyup[1]     164.0   
hold[2] 91.0
'''
