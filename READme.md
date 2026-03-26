# Система мониторинга температуры МИВК ОИЯИ

> Прогнозирование скачков температуры в системе охлаждения с помощью XGBoost

##  О проекте

Разработан прототип системы раннего предупреждения о скачках температуры.  
Система анализирует временные ряды показаний датчиков и предсказывает, произойдёт ли критическое повышение температуры в ближайшие 20 минут.

**Ключевые характеристики:**

- Бинарная классификация (скачок / норма)
- Горизонт прогноза: 20 минут
- Порог скачка: +10% относительно текущего значения
- Метрика качества: F-macro = 0.822

##  Структура проекта

jinr-monitoring/
├── datasets/ # Исходные данные
│ └── merged_dataset2.csv # Объединённый датасет
├── models/ # Сохранённые модели
│ └── final_xgb.pkl # Обученная модель XGBoost
├── logs/ # Логи обучения
│ ├── training_.txt # Полный вывод консоли
│ ├── training_results_.csv # Метрики по фолдам
│ └── model_info_*.json # Параметры модели
├── screenshots/ # Графики для отчёта
│ ├── correlation_matrix.png
│ ├── walk_forward_validation.png
│ └── positive_episode.png
├── src/ # Исходный код
│ ├── features_main.py # Временные признаки, нормализация
│ ├── episodes.py # Формирование эпизодов, разметка
│ ├── augmentation.py # Аугментации (jitter, scaling, time_warp)
│ ├── visualization.py # Визуализация
│ └── buffer.py # Буфер для положительных примеров (для будущего API)
├── train_model.py # Скрипт обучения модели
├── visualize_data.py # Скрипт визуализации данных
├── build_dataset.py # Сборка датасета из исходных CSV
└── requirements.txt # Зависимости



##  Быстрый старт

### 1. Клонирование репозитория

```bash
git clone https://github.com/your-username/jinr-monitoring.git
cd jinr-monitoring

### 2. Установка зависимостей

pip install -r requirements.txt

### 3. Подготовка данных

Поместите исходные CSV-файлы в папку source_datasets/ и выполните:
python build_dataset.py

Или просто возьмите готовый для работы датасет из datasets/merged_dataset2.csv

### 4. Визуализация данных

python visualize_data.py

### 5. Обучение модели

python train_model.py --data-path datasets/merged_dataset2.csv

После обучения:

Модель сохранится в models/final_xgb.pkl

Логи сохранятся в logs/

### Автор

Ощепков Дмитрий Владимирович
РУДН, факультет физико-математических и естественных наук
Группа НФИбд01-22

