import matplotlib.pyplot as plt

from KeystrokeAnomalyDetector import KeystrokeAnomalyDetector
from tqdm import tqdm

DATASET_PATH = 'C://learning//NIR//keystroke-django-auth//test_model//datasets//DSL-StrongPasswordData.csv'
# DATASET_PATH = '~/Desktop/learning/8_sem/NIR/keystroke_django_auth/test_model/datasets/DSL-StrongPasswordData.csv'

def validate_model(model_name: str, k: int, valid_range, params=None):

    res = {}

    for n in tqdm(valid_range, desc="Testing"):
        eer_all, roc_auc_all, far_all, fpr_all, tpr_all = 0, 0, 0, 0, 0

        if model_name == "KNN":
            param = 'n_neighbors'
            model_params={param: n, 'p': params['p']}
        elif model_name == 'IsolationForest':
            param = 'n_estimators'
            model_params={param: n, 'max_features': params['max_features']}
        elif model_name == 'OneClassSVM':
            param = 'nu'
            model_params={param: n}
        elif model_name == 'LOF':
            param = 'n_neighbors'
            model_params={param: n, 'p': params['p']}

        for _ in range(k):
            detector = KeystrokeAnomalyDetector(
                model_name=model_name, 
                model_params=model_params,
                scaler_enabled=False
            )
            # detector.run_pipeline(
            #     extractor_path=DATASET_PATH,
            #     target_subject='s002',
            #     impostors=impostors,
            #     n_test_legit=100,
            #     n_test_impostors_each=2
            # )
            detector.run_validation(
                extractor_path=DATASET_PATH,
                target_subject='s002',
                n_test_legit=50,
                n_test_anomaly=30,
                n_test_impostors_each=2,
                impostors=impostors
            )
            eer_all += detector.get_report()['eer'] * 100
            roc_auc_all += detector.get_report()['roc_auc']
            far_all += detector.get_report()['far']
            fpr_all = detector.get_report()['fpr']
            tpr_all = detector.get_report()['tpr']
            
        eer_mean = float(eer_all/k)
        roc_auc_mean = float(roc_auc_all/k)
        far_mean = float(far_all/k)
        # fpr_mean = float(fpr_all/k)
        # tpr_mean = float(tpr_all/k)
        res[str(n)] = [eer_mean, roc_auc_mean, far_mean]
    
    for key in res:
        print(f'n_neighbors = {key}, EER = {(res[key][0]):.2f}, AUC = {(res[key][1]):.2f}, FAR = {(res[key][2]):.2f}')

    best_key = min(res, key=res.get)
    print(f"The best result: {param} = {best_key}, EER = {(res[best_key][0]):.2f}, AUC = {(res[best_key][1]):.2f}, FAR = {(res[best_key][2]):.2f}")

    return res

    # # Build graph of EER
    # plt.figure(figsize=(8, 5))
    # plt.plot(list(res.keys()), [v[0] for v in res.values()], marker='o', linestyle='-', color='blue')
    # plt.xlabel(param)
    # plt.ylabel('EER (%)')
    # plt.title(f'Зависимость EER от параметра {param} с метрикой Минковского')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

def validate_KNN(k: int, valid_range, p: int):
    return validate_model(
        model_name='KNN', 
        k=k, 
        valid_range = valid_range, 
        params={'p': p}
    )

def validate_Isolation_Forest(k: int, valid_range, max_features: float):
    return validate_model(
        model_name='IsolationForest', 
        k=k, 
        valid_range = valid_range,
        params={'max_features': max_features}
    )

def validate_SVM(k: int, valid_range):
    return validate_model(
        model_name='OneClassSVM', 
        k=k, 
        valid_range = valid_range
    )

def validate_LOF(k: int, valid_range, p):
    return validate_model(
        model_name='LOF', 
        k=k, 
        valid_range = valid_range,
        params={'p': p}
    )

if __name__ == '__main__':
    mode = ['OneClassSVM', 'IsolationForest', 'LOF', 'KNN']
    impostors = [f's00{i}' for i in range (3, 10, 1)]
    impostors.extend(
        [f's0{i}' for i in range (10, 58, 1)]
    )

    #
    # KNN
    #

    res1 = validate_KNN(k=10, valid_range=range(2, 16), p=1)
    res2 = validate_KNN(k=10, valid_range=range(2, 16), p=2)

    plt.figure(figsize=(8, 5))
    plt.plot(list(res1.keys()), [v[0] for v in res1.values()], marker='o', linestyle='-', color='blue', label='Метрика Манхэттена')
    plt.plot(list(res2.keys()), [v[0] for v in res2.values()], marker='o', linestyle='-', color='green', label='Метрика Евклида')

    plt.xlabel("Количество соседей")
    plt.ylabel('Значение EER (%)')

    plt.title(f'Зависимость EER от количества соседей и метрики')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(list(res1.keys()), [v[1] for v in res1.values()], marker='o', linestyle='-', color='blue', label='Метрика Манхэттена')
    plt.plot(list(res2.keys()), [v[1] for v in res2.values()], marker='o', linestyle='-', color='green', label='Метрика Евклида')

    plt.xlabel("Количество соседей")
    plt.ylabel('Значение AUC')

    plt.title(f'Зависимость AUC от количества соседей и метрики')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    #
    # One Class SVM
    #

    res1 = validate_SVM(k=10, valid_range=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])# np.arange(0.01, 0.5, 0.01))

    plt.figure(figsize=(8, 5))
    plt.plot(list(res1.keys()), [v[0] for v in res1.values()], marker='o', linestyle='-', color='blue')

    plt.xlabel("Количество соседей")
    plt.ylabel('Значение EER (%)')

    plt.title(f'Зависимость EER от количества соседей')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(list(res1.keys()), [v[1] for v in res1.values()], marker='o', linestyle='-', color='blue')

    plt.xlabel("Количество соседей")
    plt.ylabel('Значение AUC')

    plt.title(f'Зависимость AUC от количества соседей')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    #
    # Isolation Forest
    #
    
    range_from = 10
    range_to = 200
    range_step = 10

    res1 = validate_Isolation_Forest(k=2, valid_range=range(range_from, range_to, range_step), max_features=0.3)
    res2 = validate_Isolation_Forest(k=2, valid_range=range(range_from, range_to, range_step), max_features=0.5)
    res3 = validate_Isolation_Forest(k=2, valid_range=range(range_from, range_to, range_step), max_features=0.7)
    res4 = validate_Isolation_Forest(k=2, valid_range=range(range_from, range_to, range_step), max_features=0.9)
    res5 = validate_Isolation_Forest(k=2, valid_range=range(range_from, range_to, range_step), max_features=1.0)

    plt.figure(figsize=(8, 5))
    plt.plot(list(res1.keys()), [v[0] for v in res1.values()], marker='o', linestyle='-', color='blue', label=f'max_features={0.3}')
    plt.plot(list(res2.keys()), [v[0] for v in res2.values()], marker='o', linestyle='-', color='green', label=f'max_features={0.5}')
    plt.plot(list(res3.keys()), [v[0] for v in res3.values()], marker='o', linestyle='-', color='red', label=f'max_features={0.7}')
    plt.plot(list(res4.keys()), [v[0] for v in res4.values()], marker='o', linestyle='-', color='purple', label=f'max_features={0.9}')
    plt.plot(list(res5.keys()), [v[0] for v in res5.values()], marker='o', linestyle='-', color='orange', label=f'max_features={1.0}')

    plt.xlabel("Количество соседей")
    plt.ylabel('Значение EER (%)')

    plt.title(f'Зависимость EER от количества соседей')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(list(res1.keys()), [v[1] for v in res1.values()], marker='o', linestyle='-', color='blue')

    plt.xlabel("Количество соседей")
    plt.ylabel('Значение AUC')

    plt.title(f'Зависимость AUC от количества соседей')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    #
    # LOF
    #
    
    res1 = validate_LOF(k=10, valid_range=range(2, 16, 1), p=1)
    res2 = validate_LOF(k=10, valid_range=range(2, 16, 1), p=2)


    plt.figure(figsize=(8, 5))
    plt.plot(list(res1.keys()), [v[0] for v in res1.values()], marker='o', linestyle='-', color='blue', label='Метрика Манхэттена (p=1)')
    plt.plot(list(res2.keys()), [v[0] for v in res2.values()], marker='o', linestyle='-', color='green', label='Метрика Евклида (p=2)')

    plt.xlabel("Количество соседей")
    plt.ylabel('Значение EER (%)')

    plt.title(f'Зависимость EER от количества соседей и метрики')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(list(res1.keys()), [v[1] for v in res1.values()], marker='o', linestyle='-', color='blue', label='Метрика Манхэттена (p=1)')
    plt.plot(list(res2.keys()), [v[1] for v in res2.values()], marker='o', linestyle='-', color='green', label='Метрика Евклида (p=2)')


    plt.xlabel("Количество соседей")
    plt.ylabel('Значение AUC')

    plt.title(f'Зависимость AUC от количества соседей и метрики')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()