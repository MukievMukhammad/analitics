import pandas as pd
import re
import glob
import streamlit as st


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
