# Airbnb Dashboard — CABA

Dashboard interactivo hecho con **Streamlit** para explorar datos de Airbnb en la Ciudad de Buenos Aires.

## Demo
 [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://airbnb-dashboard-dvkswvc9wyr8nguxfgd2bn.streamlit.app/)


## Funcionalidades
- Filtros por barrio, tipo de habitación, rango de precios y noches mínimas
- KPIs con métricas rápidas
- Histogramas, rankings y boxplots interactivos con Plotly
- Mapa de densidad (hexágonos) con Pydeck
- Matriz de correlación
- Tabla y descarga de los listings filtrados (CSV/Parquet)

## Stack
- Python
- Streamlit
- Pandas
- Plotly
- Pydeck

## Estructura
- `app.py`: app principal en Streamlit
- `src/data_prep.py`: script de limpieza y normalización
- `data/`: dataset (CSV original y parquet procesado)
