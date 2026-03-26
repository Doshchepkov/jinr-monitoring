"""
Буфер для хранения положительных примеров (скачков)
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime


class PositiveBuffer:
    """
    Буфер для сохранения положительных примеров (скачков температуры)
    для последующего дообучения модели.
    """
    
    def __init__(self, buffer_path: str = "buffer/positive_events.csv"):
        """
        Parameters
        ----------
        buffer_path : str
            Путь к файлу для хранения буфера
        """
        self.buffer_path = buffer_path
        os.makedirs(os.path.dirname(buffer_path), exist_ok=True)
        
        # Загружаем существующие примеры
        self.data = self._load()
    
    def _load(self) -> pd.DataFrame:
        """Загружает буфер из файла"""
        if os.path.exists(self.buffer_path):
            try:
                return pd.read_csv(self.buffer_path)
            except:
                return pd.DataFrame()
        return pd.DataFrame()
    
    def _save(self):
        """Сохраняет буфер в файл"""
        self.data.to_csv(self.buffer_path, index=False)
    
    def add_episode(self, df_window: pd.DataFrame, timestamp: str, probability: float):
        """
        Добавляет эпизод в буфер.
        
        Parameters
        ----------
        df_window : pd.DataFrame
            Окно данных (L минут), которое привело к предсказанию
        timestamp : str
            Временная метка момента предсказания
        probability : float
            Вероятность скачка (от модели)
        """
        # Преобразуем окно в строку (можно сохранять как JSON)
        window_json = df_window.to_json(orient="records", date_format="iso")
        
        new_row = pd.DataFrame([{
            "timestamp": timestamp,
            "probability": probability,
            "window_data": window_json,
            "added_at": datetime.now().isoformat(),
            "used_for_retraining": False
        }])
        
        self.data = pd.concat([self.data, new_row], ignore_index=True)
        self._save()
        print(f"✅ Положительный эпизод добавлен в буфер. Всего в буфере: {len(self.data)}")
    
    def get_unused(self) -> pd.DataFrame:
        """Возвращает все эпизоды, которые ещё не использовались для дообучения"""
        return self.data[self.data["used_for_retraining"] == False]
    
    def mark_as_used(self, indices):
        """Отмечает эпизоды как использованные для дообучения"""
        self.data.loc[indices, "used_for_retraining"] = True
        self._save()
    
    def get_for_retraining(self, max_samples: int = 1000) -> list:
        """
        Возвращает список эпизодов для дообучения.
        
        Parameters
        ----------
        max_samples : int
            Максимальное количество эпизодов для извлечения
        
        Returns
        -------
        list
            Список DataFrame'ов с эпизодами
        """
        unused = self.get_unused()
        if len(unused) == 0:
            return []
        
        # Берём последние max_samples
        to_use = unused.tail(max_samples)
        
        episodes = []
        for _, row in to_use.iterrows():
            # Восстанавливаем DataFrame из JSON
            df_window = pd.read_json(row["window_data"], orient="records")
            episodes.append(df_window)
        
        # Отмечаем как использованные
        self.mark_as_used(to_use.index)
        
        return episodes
    
    def size(self) -> int:
        """Количество эпизодов в буфере"""
        return len(self.data)
    
    def unused_size(self) -> int:
        """Количество неиспользованных эпизодов"""
        return len(self.get_unused())