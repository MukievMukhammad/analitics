import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose

from data import load_data, prepare_data


# Настройка страницы
st.set_page_config(page_title="Анализ временных рядов", layout="wide")
st.title("Анализ тренда и сезонности по запросам из Яндекса")

df_list = load_data()

# Создаем список источников для выбора
sources = []
for i, df in enumerate(df_list):
    if "source" in df.columns and not df["source"].empty:
        source_name = df["source"].iloc[0]
        sources.append(f"{i}: {source_name}")

# Выбор таблицы
selected_option = st.selectbox("Выберите таблицу для анализа:", sources)

# Извлекаем индекс выбранной таблицы
selected_index = int(selected_option.split(":")[0])
df_selected = df_list[selected_index]

df_prepared = prepare_data(df_selected)

# Параметры анализа
col1, col2 = st.columns(2)
with col1:
    period = st.slider("Период сезонности", min_value=4, max_value=12, value=12)
with col2:
    model_type = st.selectbox("Тип модели", ["additive", "multiplicative"])

# Выполнение декомпозиции
try:
    result = seasonal_decompose(
        df_prepared["Число запросов"], period=period, model=model_type
    )

    # Отображение основной информации
    st.subheader(f"Анализ для: {df_selected['source'].iloc[0]}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Количество наблюдений", len(df_prepared))
    with col2:
        st.metric(
            "Период анализа",
            f"{df_prepared.index.min().strftime('%b %Y')} - {df_prepared.index.max().strftime('%b %Y')}",
        )
    with col3:
        st.metric("Среднее значение", f"{df_prepared['Число запросов'].mean():.0f}")

    # Переключатель для выбора типа графиков
    chart_type = st.radio("Тип графиков:", ["Plotly (интерактивные)", "Matplotlib"])

    if chart_type == "Plotly (интерактивные)":
        # Создаем интерактивные графики
        fig_plotly = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=("Исходные данные", "Тренд", "Сезонность", "Остатки"),
            shared_xaxes=True,
            vertical_spacing=0.1,
        )

        fig_plotly.add_trace(
            go.Scatter(
                x=result.observed.index,
                y=result.observed.values,
                name="Observed",
                line=dict(color="blue"),
            ),
            row=1,
            col=1,
        )

        fig_plotly.add_trace(
            go.Scatter(
                x=result.trend.index,
                y=result.trend.values,
                name="Trend",
                line=dict(color="red"),
            ),
            row=2,
            col=1,
        )

        fig_plotly.add_trace(
            go.Scatter(
                x=result.seasonal.index,
                y=result.seasonal.values,
                name="Seasonal",
                line=dict(color="green"),
            ),
            row=3,
            col=1,
        )

        # fig_plotly.add_trace(go.Scatter(
        #     x=result.resid.index,
        #     y=result.resid.values,
        #     name='Residual',
        #     line=dict(color='orange')
        # ), row=4, col=1)

        # Форматирование оси X для всех подграфиков
        for i in range(1, 4):  # Для всех 3 подграфиков
            fig_plotly.update_xaxes(
                tickformat="%b %Y",  # Формат: Янв 2023
                showticklabels=True,
                row=i, col=1
            )

        fig_plotly.update_layout(height=800, showlegend=False)
        st.plotly_chart(fig_plotly, use_container_width=True)

    if chart_type == "Matplotlib":
        # Создание графиков
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

        # Исходные данные
        axes[0].plot(result.observed, color="blue", linewidth=2)
        axes[0].set_title("Исходные данные", fontsize=12, fontweight="bold")
        axes[0].grid(True, alpha=0.3)

        # Тренд
        axes[1].plot(result.trend, color="red", linewidth=2)
        axes[1].set_title("Тренд", fontsize=12, fontweight="bold")
        axes[1].grid(True, alpha=0.3)

        # Сезонность
        axes[2].plot(result.seasonal, color="green", linewidth=2)
        axes[2].set_title("Сезонность", fontsize=12, fontweight="bold")
        axes[2].grid(True, alpha=0.3)

        # Остатки
        axes[3].plot(result.resid, color="orange", linewidth=2)
        axes[3].set_title("Остатки", fontsize=12, fontweight="bold")
        axes[3].grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        plt.xlabel("Период")
        plt.tight_layout()

        # Отображение графика в Streamlit
        st.pyplot(fig)

except Exception as e:
    st.error(f"Ошибка при выполнении анализа: {e}")


# Статистики по компонентам
st.subheader("Статистический анализ")

col1, col2 = st.columns(2)

with col1:
    st.write("**Тренд**")
    trend_stats = {
        "Среднее": result.trend.mean(),
        "Мин": result.trend.min(),
        "Макс": result.trend.max(),
        "Стд. отклонение": result.trend.std(),
    }
    st.json(trend_stats)

with col2:
    st.write("**Сезонность**")
    seasonal_stats = {
        "Среднее": result.seasonal.mean(),
        "Мин": result.seasonal.min(),
        "Макс": result.seasonal.max(),
        "Амплитуда": result.seasonal.max() - result.seasonal.min(),
    }
    st.json(seasonal_stats)

# Таблица с данными
if st.checkbox("Показать исходные данные"):
    st.subheader("Исходные данные")
    st.dataframe(df_prepared)


# Кнопка для анализа всех таблиц
if st.button("Анализировать все таблицы"):
    progress_bar = st.progress(0)
    results_summary = []

    for i, df in enumerate(df_list):
        try:
            df_prep = prepare_data(df)
            if len(df_prep) >= 22:  # Минимум для анализа
                result = seasonal_decompose(
                    df_prep["Число запросов"],
                    period=min(11, len(df_prep) // 2),
                    model="additive",
                )

                results_summary.append(
                    {
                        "Источник": df["source"].iloc[0],
                        "Наблюдений": len(df_prep),
                        "Средний тренд": result.trend.mean(),
                        "Амплитуда сезонности": result.seasonal.max()
                        - result.seasonal.min(),
                        "Среднее значение": df_prep["Число запросов"].mean(),
                    }
                )
        except:
            pass

        progress_bar.progress((i + 1) / len(df_list))

    # Отображение сводной таблицы
    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        st.subheader("Сводные результаты анализа")
        st.dataframe(summary_df)

        # Возможность скачать результаты
        csv = summary_df.to_csv(index=False)
        st.download_button(
            label="Скачать результаты CSV",
            data=csv,
            file_name="analysis_results.csv",
            mime="text/csv",
        )
