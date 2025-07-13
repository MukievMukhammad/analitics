import pandas as pd
import re
import glob
import streamlit as st
import numpy as np
from scipy.stats import pearsonr
from statsmodels.tsa.seasonal import seasonal_decompose


@st.cache_data
def load_data():
    all_files = glob.glob("data/*.csv")
    df_list = []

    for file in all_files:
        # Читаем файл с разделителем ";"
        temp = pd.read_csv(file, sep=";")
        # Получаем название последнего столбца
        last_col = temp.columns[-1]
        # Извлекаем словосочетание между кавычками « »
        match = re.search(r"«([^»]+)»", str(last_col))
        if match:
            source_phrase = match.group(1)
        else:
            source_phrase = file  # если кавычки не найдены, используем имя файла
        # Исключаем первую строку из таблицы
        temp = temp.iloc[1:]
        # Удаляем последний столбец из таблицы
        temp = temp.drop(columns=[last_col])
        # Добавляем столбец source с извлечённым словосочетанием
        temp["source"] = source_phrase
        df_list.append(temp)

    return df_list

    # # Словарь для преобразования русских названий месяцев
    # month_map = {
    #     "январь": "01",
    #     "февраль": "02",
    #     "март": "03",
    #     "апрель": "04",
    #     "май": "05",
    #     "июнь": "06",
    #     "июль": "07",
    #     "август": "08",
    #     "сентябрь": "09",
    #     "октябрь": "10",
    #     "ноябрь": "11",
    #     "декабрь": "12",
    # }

    # def convert_russian_month_year(date_str):
    #     """Преобразует 'август 2023' в '2023-08-01'"""
    #     parts = date_str.split()
    #     if len(parts) == 2:
    #         month_rus, year = parts
    #         month_num = month_map.get(month_rus.lower())
    #         if month_num:
    #             return f"{year}-{month_num}-01"
    #     return None

    # for df in df_list:
    #     try:
    #         df["Число запросов"] = df["Число запросов"].str.replace(" ", "").astype(int)
    #     except:
    #         pass

    # for df in df_list:
    #     df["Период"] = df["Период"].apply(convert_russian_month_year)
    #     df["Период"] = pd.to_datetime(df["Период"], errors="coerce")

    # results = []
    # for source, group in df.groupby("source"):
    #     group = group.sort_values("Период")
    #     result = seasonal_decompose(
    #         group["Число запросов"],
    #         period=12,
    #         model="additive",
    #         extrapolate_trend="freq",
    #     )
    #     results.append(
    #         {
    #             "source": source,
    #             "trend": result.trend,
    #             "seasonal": result.seasonal,
    #             "observed": group["Число запросов"],
    #         }
    #     )

    # return results


# Функция преобразования дат
@st.cache_data
def prepare_data(df):
    month_map = {
        "январь": "01",
        "февраль": "02",
        "март": "03",
        "апрель": "04",
        "май": "05",
        "июнь": "06",
        "июль": "07",
        "август": "08",
        "сентябрь": "09",
        "октябрь": "10",
        "ноябрь": "11",
        "декабрь": "12",
    }

    def convert_russian_month_year(date_str):
        parts = str(date_str).split()
        if len(parts) == 2:
            month_rus, year = parts
            month_num = month_map.get(month_rus.lower())
            if month_num:
                return f"{year}-{month_num}-01"
        return None

    df_copy = df.copy()
    df_copy["Период"] = df_copy["Период"].apply(convert_russian_month_year)
    df_copy["Период"] = pd.to_datetime(df_copy["Период"], errors="coerce")
    df_copy = df_copy.set_index("Период")

    # Очистка данных
    try:
        df_copy["Число запросов"] = (
            df_copy["Число запросов"].str.replace(" ", "").astype(float)
        )
    except Exception as e:
        pass
        # st.error(f"Ошибка при перобразовании типа 'Число запросов': {e}")

    return df_copy


def sine(x, shift=0):
    coeff = 1.1
    return np.sin(2 * np.pi * coeff * (x + shift) / 12)


def similarity_to_sine(seasonal_component):
    """Вычисляет коэффициент корреляции Пирсона между сезонностью и синусоидой"""
    if seasonal_component is None or len(seasonal_component) == 0:
        return -1
    
    # Удаляем NaN значения
    seasonal_clean = seasonal_component.dropna()
    if len(seasonal_clean) < 3:  # Минимум точек для корреляции
        return -1
    
    n = len(seasonal_clean)
    # Создаем синусоиду с тем же количеством точек
    x = np.arange(n)
    sine_wave = sine(x)  # 12-месячный период
    
    try:
        # Вычисляем корреляцию Пирсона
        corr, _ = pearsonr(seasonal_clean.values, sine_wave)
        return abs(corr)  # Берем абсолютное значение (инвертированная синусоида тоже похожа)
    except:
        return -1

import numpy as np
from scipy.stats import pearsonr

def find_best_shift_and_correlation(seasonal_component, compression_factor=1.0):
    """Находит оптимальный сдвиг синусоиды для максимального наложения"""
    if seasonal_component is None or len(seasonal_component) == 0:
        return 0, -1
    
    seasonal_clean = seasonal_component.dropna()
    if len(seasonal_clean) < 3:
        return 0, -1
    
    n = len(seasonal_clean)
    x = np.arange(n)
    
    best_corr = -1
    best_shift = 0
    
    # Тестируем сдвиги от 0 до 12 месяцев
    for shift in np.arange(0, 12, 0.5):  # Шаг 0.5 для более точного поиска
        # Создаем сдвинутую синусоиду
        sine_wave = sine(x, shift)
        
        try:
            corr, _ = pearsonr(seasonal_clean.values, sine_wave)
            if abs(corr) > best_corr:
                best_corr = abs(corr)
                best_shift = shift
        except:
            continue
    
    return best_shift, best_corr


def analyze_all_tables(df_list):
    """Анализирует все таблицы и возвращает результаты с сортировкой по схожести"""
    results = []
    
    for i, df in enumerate(df_list):
        try:
            # Подготавливаем данные
            df_prepared = prepare_data(df)
            if len(df_prepared) < 8:
                print("df_prepared < 8")
                continue
                
            # Выполняем декомпозицию
            period = min(11, len(df_prepared) // 2)
            result = seasonal_decompose(df_prepared['Число запросов'], 
                                      period=period, 
                                      model='additive')
            
            # Вычисляем схожесть с синусоидой
            best_shift = 0
            # best_shift, sine_similarity = find_best_shift_and_correlation(result.seasonal)
            sine_similarity = similarity_to_sine(result.seasonal)
            
            results.append({
                'index': i,
                'source': df['source'].iloc[0] if 'source' in df.columns else f"Таблица {i}",
                'similarity': sine_similarity,
                'result': result,
                'best_shift': best_shift,
                'data': df_prepared,
                'observations': len(df_prepared)
            })
            
        except Exception as e:
            print(e)
            continue
    
    # Сортируем по схожести (от большей к меньшей)
    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results


def seasonality_peak_score(seasonal_component, peak_months=[8, 9, 10]):
    """Вычисляет оценку активности сезонности в летне-осенний период"""
    if seasonal_component is None or len(seasonal_component) == 0:
        return -1
    
    seasonal_clean = seasonal_component.dropna()
    if len(seasonal_clean) == 0:
        return -1
    
    # Получаем месяцы из индекса
    if hasattr(seasonal_clean.index, 'month'):
        months = seasonal_clean.index.month
    else:
        # Если индекс не даты, создаем циклические месяцы
        months = (np.arange(len(seasonal_clean)) % 12) + 1
    
    # Выбираем значения сезонности для июня-сентября
    peak_mask = np.isin(months, peak_months)
    peak_values = seasonal_clean[peak_mask]
    
    if len(peak_values) == 0:
        return -1
    
    # Оценка - среднее значение сезонности в эти месяцы
    # Положительные значения означают пик, отрицательные - спад
    score = peak_values.mean()
    return score


def analyze_all_tables_with_dual_sorting(df_list):
    """Анализирует все таблицы с двумя параметрами сортировки"""
    results = []
    
    for i, df in enumerate(df_list):
        try:
            df_prepared = prepare_data(df)
            if len(df_prepared) < 8:
                continue
                
            period = min(11, len(df_prepared) // 2)
            result = seasonal_decompose(df_prepared['Число запросов'], 
                                      period=period, 
                                      model='additive')
            
            # Первый параметр - схожесть с синусоидой
            best_shift, sine_similarity = find_best_shift_and_correlation(result.seasonal, 1.1)
            
            # Второй параметр - активность в июне-сентябре
            summer_peak_score = seasonality_peak_score(result.seasonal)
            
            results.append({
                'index': i,
                'source': df['source'].iloc[0] if 'source' in df.columns else f"Таблица {i}",
                'similarity': sine_similarity,
                'summer_peak': summer_peak_score,
                'best_shift': best_shift,
                'result': result,
                'data': df_prepared,
                'observations': len(df_prepared)
            })
            
        except Exception as e:
            # st.error(e)
            continue
    
    # Двойная сортировка: сначала по схожести с синусоидой, затем по летнему пику
    results.sort(key=lambda x: (x['summer_peak'], x['similarity']), reverse=True)
    return results
