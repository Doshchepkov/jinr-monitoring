#!/usr/bin/env python
"""
Обучение модели XGBoost для прогнозирования скачков температуры.

Запуск:
    python train_model.py --data-path datasets/merged_dataset2.csv
"""

import argparse
import pandas as pd
import numpy as np
import joblib
import gc
import time
import os
import json
import sys
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.metrics import fbeta_score
import warnings
warnings.filterwarnings('ignore')

from src.features_main import add_time_features, compute_normalization, split_into_folds
from src.episodes import make_episodes, print_stats


def setup_logging(save_dir: str = "logs") -> tuple:
    """
    Настраивает логирование: создаёт папку, сохраняет вывод консоли в файл.
    
    Parameters
    ----------
    save_dir : str
        Папка для сохранения логов
    
    Returns
    -------
    tuple
        (log_dir, timestamp, log_file_path)
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(save_dir, f"training_{timestamp}.txt")
    
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    
    original_stdout = sys.stdout
    log_f = open(log_file, 'w', encoding='utf-8')
    sys.stdout = Tee(original_stdout, log_f)
    
    print(f"Логи сохраняются в: {log_file}")
    print(f"Дата запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return log_f, log_file


def train_model(
    data_path: str,
    L: int = 60,
    H: int = 30,
    n: float = 0.1,
    save_path: str = "models/final_xgb.pkl",
    logs_dir: str = "logs"
) -> tuple:
    """
    Основная функция обучения модели XGBoost с сохранением логов.
    """
    
    log_f, log_file = setup_logging(logs_dir)
    
    print("=" * 50)
    print("Обучение модели XGBoost")
    print("=" * 50)
    
    params = {
        "data_path": data_path,
        "L": L,
        "H": H,
        "n": n,
        "save_path": save_path,
        "timestamp": datetime.now().isoformat()
    }
    print("\nПараметры обучения:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    
    # 1. Загрузка данных
    print("\n1. Загрузка данных...")
    df = pd.read_csv(data_path)
    print(f"   Загружено {len(df):,} строк")
    print(f"   Колонки: {df.columns.tolist()[:10]}...")
    
    # 2. Добавление временных признаков
    print("\n2. Добавление временных признаков...")
    df = add_time_features(df, "_time")
    
    # 3. Список признаков (исключаем время)
    feature_cols = [c for c in df.columns if c not in ["_time"]]
    print(f"   Признаков: {len(feature_cols)}")
    
    # 4. Разбиение на фолды
    print("\n3. Разбиение на фолды...")
    folds = split_into_folds(df, n_folds=10)
    print(f"   Создано {len(folds)} фолдов")
    for i, f in enumerate(folds):
        print(f"     Фолд {i}: {len(f)} строк")
    
    # 5. Walk-forward валидация
    print("\n4. Walk-forward валидация...")
    
    results = []
    best_model = None
    best_mean = None
    best_std = None
    best_thr_final = None
    best_f_macro = -1
    
    for k in range(5):
        print(f"\n   === Fold {k+1}/5 ===")
        
        # Разбиение
        train_df = pd.concat(folds[k:k+4]).reset_index(drop=True)
        val_df = folds[k+4].reset_index(drop=True)
        test_df = folds[k+5].reset_index(drop=True)
        
        # Нормализация
        mean, std = compute_normalization(train_df, feature_cols)
        
        # Формирование эпизодов
        X_train, y_train = make_episodes(train_df, feature_cols, mean, std, L, H, n, aug_k=1)
        X_val, y_val = make_episodes(val_df, feature_cols, mean, std, L, H, n, aug_k=1)
        X_test, y_test = make_episodes(test_df, feature_cols, mean, std, L, H, n, aug_k=1)
        
        # Преобразование в 2D
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_val = X_val.reshape(X_val.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        
        print(f"   Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        print_stats("   Train", y_train)
        print_stats("   Val", y_val)
        print_stats("   Test", y_test)
        
        # Обучение
        print("   Обучение XGBoost...")
        start_time = time.time()
        
        model = XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric="logloss"
        )
        
        model.fit(X_train, y_train)
        elapsed = time.time() - start_time
        
        # Предсказания
        prob_val = model.predict_proba(X_val)[:, 1]
        
        # Поиск лучшего порога
        best_thr = 0.5
        best_f = 0
        
        for thr in np.linspace(0.1, 0.9, 9):
            pred_val = (prob_val >= thr).astype(int)
            f2 = fbeta_score(y_val, pred_val, beta=2, pos_label=1, zero_division=0)
            f1 = fbeta_score(y_val, pred_val, beta=1, pos_label=0, zero_division=0)
            f_macro = (f2 + f1) / 2
            
            if f_macro > best_f:
                best_f = f_macro
                best_thr = thr
        
        # Оценка на тесте
        prob_test = model.predict_proba(X_test)[:, 1]
        pred_test = (prob_test >= best_thr).astype(int)
        
        f2_test = fbeta_score(y_test, pred_test, beta=2, pos_label=1, zero_division=0)
        f1_test = fbeta_score(y_test, pred_test, beta=1, pos_label=0, zero_division=0)
        f_macro_test = (f2_test + f1_test) / 2
        
        print(f"   Лучший порог: {best_thr:.2f}")
        print(f"   Val F-macro: {best_f:.3f}")
        print(f"   Test F-macro: {f_macro_test:.3f}")
        print(f"   Время: {elapsed:.1f} сек")
        
        # Сохраняем лучшую модель по F-macro на тесте
        if f_macro_test > best_f_macro:
            best_f_macro = f_macro_test
            best_model = model
            best_mean = mean
            best_std = std
            best_thr_final = best_thr
        
        results.append({
            "fold": k+1,
            "thr": best_thr,
            "val_f_macro": best_f,
            "test_f_macro": f_macro_test,
            "train_size": len(y_train),
            "val_size": len(y_val),
            "test_size": len(y_test),
            "train_pos_pct": (y_train.sum() / len(y_train) * 100),
            "time_sec": elapsed
        })
        
        # Очистка памяти
        del X_train, y_train, X_val, y_val, X_test, y_test
        gc.collect()
    
    # 6. Сохраняем результаты в CSV
    print("\n" + "=" * 50)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ ВАЛИДАЦИИ")
    print("=" * 50)
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print(f"\nСредний Test F-macro: {results_df['test_f_macro'].mean():.3f}")
    print(f"Лучший Test F-macro: {best_f_macro:.3f} (Fold {results_df[results_df['test_f_macro'] == best_f_macro]['fold'].values[0]})")
    
    # Сохраняем результаты
    results_csv = os.path.join(logs_dir, f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\n✅ Результаты сохранены в: {results_csv}")
    
    # 7. Сохраняем лучшую модель (из лучшего фолда)
    print("\n" + "=" * 50)
    print("Сохранение лучшей модели...")
    print("=" * 50)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    model_data = {
        "model": best_model,
        "mean": best_mean,
        "std": best_std,
        "feature_cols": feature_cols,
        "L": L,
        "H": H,
        "n": n,
        "threshold": best_thr_final,
        "best_test_f_macro": best_f_macro,
        "mean_test_f_macro": results_df['test_f_macro'].mean(),
        "timestamp": datetime.now().isoformat()
    }
    
    joblib.dump(model_data, save_path)
    
    # Сохраняем информацию о модели в JSON
    model_info = {
        "model_path": save_path,
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "n_estimators": best_model.n_estimators,
        "max_depth": best_model.max_depth,
        "learning_rate": best_model.learning_rate,
        "L": L,
        "H": H,
        "n": n,
        "threshold": best_thr_final,
        "best_test_f_macro": best_f_macro,
        "mean_test_f_macro": results_df['test_f_macro'].mean(),
        "timestamp": datetime.now().isoformat()
    }
    
    model_info_path = os.path.join(logs_dir, f"model_info_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(model_info_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Модель сохранена в {save_path}")
    print(f"✅ Информация о модели сохранена в {model_info_path}")
    print(f"   Использована лучшая модель с Test F-macro = {best_f_macro:.3f}")
    print(f"   Рекомендуемый порог: {best_thr_final:.2f}")
    
    # Закрываем файл лога и восстанавливаем stdout
    sys.stdout = sys.__stdout__
    log_f.close()
    
    print(f"\n✅ Полный лог сохранён в: {log_file}")
    
    return best_model, results_df


# ==================== ТОЧКА ВХОДА ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение модели XGBoost")
    parser.add_argument("--data-path", required=True, help="Путь к CSV файлу с данными")
    parser.add_argument("--L", type=int, default=60, help="Длина исторического окна (минут)")
    parser.add_argument("--H", type=int, default=30, help="Горизонт прогноза (минут)")
    parser.add_argument("--n", type=float, default=0.1, help="Порог скачка (10% = 0.1)")
    parser.add_argument("--save-path", default="models/final_xgb.pkl", help="Путь для сохранения модели")
    parser.add_argument("--logs-dir", default="logs", help="Папка для сохранения логов")
    
    args = parser.parse_args()
    
    train_model(
        data_path=args.data_path,
        L=args.L,
        H=args.H,
        n=args.n,
        save_path=args.save_path,
        logs_dir=args.logs_dir
    )