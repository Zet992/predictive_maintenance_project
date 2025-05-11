import streamlit as st
import reveal_slides as rs


def presentation_page():
    st.title("Презентация проекта")
    presentation_markdown = """
    # Прогнозирование отказов оборудования
    ---
    ## Введение
    - Описание задачи и датасета.
    - Цель: предсказать отказ оборудования (Target = 1) или его отсутствие (Target = 0).
    ---
    ## Этапы работы
    1. Загрузка данных.
    2. Предобработка данных.
    3. Обучение модели.
    4. Оценка модели.
    5. Визуализация результатов.
    ---
    ## Streamlit-приложение
    - Основная страница: анализ данных и предсказания.
    - Страница с презентацией: описание проекта.
    ---
    ## Заключение
    - Удалось сделать модель с удовлетворительной точностью
    - Для улучшения проекта можно:
    - Научить модель предсказывать тип отказов
    - Улучшить предобработку
    - Выбрать более мощную модель
    - Оптимизировать обучение модели
    - Внедрить ClearML в проект
    """

    with st.sidebar:
        st.header("Настройки презентации")
        theme = st.selectbox("Тема", ["black", "white", "league", "beige", "sky",
                                      "night", "serif", "simple", "solarized"])
        height = st.number_input("Высота слайдов", value=500)
        transition = st.selectbox("Переход", ["slide", "convex", "concave",
                                              "zoom", "none"])
        plugins = st.multiselect("Плагины", ["highlight", "katex", "mathjax2",
                                             "mathjax3", "notes", "search", "zoom"], [])
    rs.slides(
        presentation_markdown,
        height=height,
        theme=theme,
        config={
            "transition": transition,
            "plugins": plugins,
        },
        markdown_props={"data-separator-vertical": "^--$"},
    )


presentation_page()
