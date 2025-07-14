import streamlit as st
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from single_table import render_detail_page

from utils import (
    analyze_all_tables,
    load_data,
    sine,
    analyze_all_tables_with_dual_sorting,
    analyze_all_tables_with_triple_sorting,
)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose

from utils import load_data, prepare_data


# Основная структура приложения с вкладками
tab1, tab2 = st.tabs(["Все таблицы по схожести", "Индивидуальный анализ"])

with tab2:
    # Ваш существующий код для индивидуального анализа
    render_detail_page()

with tab1:
    st.header("Аналитика по запросам в Яндекс")

    df_list = load_data()

    # Кнопка для запуска анализа
    # if st.button("Анализировать все таблицы", key="analyze_all"):
    with st.spinner("Анализируем все таблицы..."):
        # Анализируем все таблицы
        all_results_ = analyze_all_tables_with_triple_sorting(df_list)

        if not all_results_:
            st.error("Не удалось проанализировать ни одну таблицу")
            st.stop()

        # Добавьте в интерфейс Streamlit:
        col2, col3 = st.columns(2)

        with col2:
            season_filter = st.selectbox(
                "Фильтр по сезонности",
                ["Летне-осенний пик", "Летне-осенний спад", "Нейтральная", "Все"],
            )

        with col3:
            min_similarity = st.slider(
                "Минимальная схожесть с синусоидой", 0.0, 1.0, 0.0, 0.1
            )

        # Применяем фильтры
        all_results = all_results_
        if season_filter != "Все":
            all_results = [
                r
                for r in all_results
                if (season_filter == "Летне-осенний пик" and r["summer_peak"] > 0)
                or (season_filter == "Летне-осенний спад" and r["summer_peak"] < -0.1)
                or (season_filter == "Нейтральная" and -0.1 <= r["summer_peak"] <= 0)
            ]

        all_results = [r for r in all_results if r["similarity"] >= min_similarity]

        # Отображаем сводную таблицу
        st.subheader("Рейтинг таблиц по трем параметрам")

        summary_data = []
        for result in all_results:
            # Определяем тип сезонности
            if result["summer_peak"] > 0:
                season_type = "Летне-осенний пик"
            elif result["summer_peak"] < -0.1:
                season_type = "Летне-осенний спад"
            else:
                season_type = "Нейтральная"

            # Форматируем максимальное значение
            max_val_formatted = (
                f"{result['max_value']:,.0f}"
                if result["max_value"] >= 1000
                else f"{result['max_value']:.1f}"
            )

            summary_data.append(
                {
                    "Ранг": len(summary_data) + 1,
                    "Источник": result["source"],
                    "Схожесть с синусоидой": f"{result['similarity']:.3f}",
                    "Летне-осенняя активность": f"{result['summer_peak']:.3f}",
                    "Максимальное значение": max_val_formatted,
                    "Тип сезонности": season_type,
                    "Оптимальный сдвиг (мес.)": f"{result['best_shift']:.1f}",
                    "Наблюдений": result["observations"],
                }
            )

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)

        # Настройки отображения
        col1, col2 = st.columns(2)
        with col1:
            show_top_n = st.slider("Показать топ N таблиц", 1, len(all_results), 100)
        with col2:
            chart_type = st.selectbox("Тип графиков", ["Plotly", "Matplotlib"])

        # Отображаем графики для топ-N таблиц
        st.subheader(f"Топ-{show_top_n} таблиц с наибольшей схожестью")

        for idx, result in enumerate(all_results[:show_top_n]):
            st.markdown(f"### {idx + 1}. {result['source']}")
            st.markdown(f"**Схожесть с синусоидой:** {result['similarity']:.3f}")

            if chart_type == "Matplotlib":
                # Matplotlib графики
                fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

                # Исходные данные
                axes[0].plot(result["result"].observed, color="blue", linewidth=2)
                axes[0].set_title("Исходные данные")
                axes[0].grid(True, alpha=0.3)

                # Тренд
                axes[1].plot(result["result"].trend, color="red", linewidth=2)
                axes[1].set_title("Тренд")
                axes[1].grid(True, alpha=0.3)

                # Сезонность
                axes[2].plot(
                    result["result"].seasonal,
                    color="green",
                    linewidth=2,
                    label="Сезонность",
                )

                # Добавляем эталонную синусоиду для сравнения
                n = len(result["result"].seasonal.dropna())
                x = np.arange(n)
                sine_reference = sine(x, result["best_shift"])
                # Масштабируем синусоиду под амплитуду сезонности
                seasonal_clean = result["result"].seasonal.dropna()
                sine_scaled = sine_reference * (
                    seasonal_clean.std() / np.std(sine_reference)
                )

                axes[2].plot(
                    seasonal_clean.index[: len(sine_scaled)],
                    sine_scaled,
                    color="orange",
                    linestyle="--",
                    alpha=0.7,
                    label="Эталонная синусоида",
                )
                axes[2].set_title(f'Сезонность (схожесть: {result["similarity"]:.3f})')
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)

                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

            else:
                # Plotly графики
                fig_plotly = make_subplots(
                    rows=3,
                    cols=1,
                    subplot_titles=("Исходные данные", "Тренд", "Сезонность"),
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                )

                # Исходные данные
                fig_plotly.add_trace(
                    go.Scatter(
                        x=result["result"].observed.index,
                        y=result["result"].observed.values,
                        line=dict(color="blue"),
                        name="Observed",
                    ),
                    row=1,
                    col=1,
                )

                if hasattr(result["result"].observed.index, "month"):
                    summer_mask = result["result"].observed.index.month.isin([7, 8, 9])
                    summer_data = result["result"].observed[summer_mask]

                    if len(summer_data) > 0:
                        fig_plotly.add_trace(
                            go.Scatter(
                                x=summer_data.index,
                                y=summer_data.values,
                                mode="markers",
                                marker=dict(
                                    color="red",
                                    size=10,
                                    symbol="circle",
                                    line=dict(width=2, color="darkred"),
                                ),
                                name="Авг-Окт",
                                showlegend=False,
                            ),
                            row=1,
                            col=1,
                        )

                # Тренд
                fig_plotly.add_trace(
                    go.Scatter(
                        x=result["result"].trend.index,
                        y=result["result"].trend.values,
                        line=dict(color="red"),
                        name="Trend",
                    ),
                    row=2,
                    col=1,
                )

                # Сезонность
                seasonal_clean = result["result"].seasonal.dropna()
                fig_plotly.add_trace(
                    go.Scatter(
                        x=seasonal_clean.index,
                        y=seasonal_clean.values,
                        line=dict(color="green"),
                        name="Seasonal",
                    ),
                    row=3,
                    col=1,
                )

                # Эталонная синусоида
                n = len(seasonal_clean)
                x = np.arange(n)
                sine_reference = np.sin(2 * np.pi * x / 12)
                sine_scaled = sine_reference * (
                    seasonal_clean.std() / np.std(sine_reference)
                )

                fig_plotly.add_trace(
                    go.Scatter(
                        x=seasonal_clean.index[: len(sine_scaled)],
                        y=sine_scaled,
                        line=dict(color="orange", dash="dash"),
                        name="Эталонная синусоида",
                    ),
                    row=3,
                    col=1,
                )

                if hasattr(seasonal_clean.index, "month"):
                    summer_mask = seasonal_clean.index.month.isin([8, 9, 10])
                    summer_data = seasonal_clean[summer_mask]

                    if len(summer_data) > 0:
                        fig_plotly.add_trace(
                            go.Scatter(
                                x=summer_data.index,
                                y=summer_data.values,
                                mode="markers",
                                marker=dict(
                                    color="red",
                                    size=10,
                                    symbol="circle",
                                    line=dict(width=2, color="darkred"),
                                ),
                                name="Авг-Окт",
                                showlegend=False,
                            ),
                            row=3,
                            col=1,
                        )

                # Форматирование осей
                for i in range(1, 4):
                    fig_plotly.update_xaxes(
                        tickformat="%b %Y", tickangle=45, row=i, col=1
                    )

                fig_plotly.update_layout(height=600, showlegend=False)
                st.plotly_chart(
                    fig_plotly, use_container_width=True, key=f"chart_{idx}"
                )

            st.markdown("---")  # Разделитель между таблицами
