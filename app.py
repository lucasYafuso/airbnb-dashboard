import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk

# 1) Configuración de página: ¡primero siempre!
st.set_page_config(page_title="Airbnb Dashboard — CABA", layout="wide")

# 2) Tema Plotly según tema de Streamlit
def _apply_plotly_theme():
    base = st.get_option("theme.base") or "light"
    template = "plotly_dark" if base == "dark" else "plotly_white"
    px.defaults.template = template  

_apply_plotly_theme()

# 3) Carga de datos cacheada
@st.cache_data
def load_data(parquet_path: str) -> pd.DataFrame:
    return pd.read_parquet(parquet_path)

DATA_PATH = "data/processed/airbnb_sample.parquet"
df = load_data(DATA_PATH)

# === Defaults calculados a partir de df ===
# (hacelo DESPUÉS de cargar df y ANTES de dibujar los widgets)
if "price" in df.columns and len(df):
    min_price = int(df["price"].min())
    max_price = int(df["price"].clip(upper=df["price"].quantile(0.99)).max())
    p05 = int(df["price"].quantile(0.05))
    p95 = int(df["price"].quantile(0.95))
    # Rango "normal" por defecto: 5–95 percentil, acotado por max_price calculado
    default_price_range = (max(min_price, p05), min(p95, max_price))
else:
    min_price, max_price = 0, 1000
    default_price_range = (0, 300)

default_min_nights = 7

# === Inicialización de session_state (solo primera vez) ===
DEFAULTS = {
    "neighs": [],                 # sin barrios seleccionados
    "roomtypes": [],              # sin tipos seleccionados
    "price_range": default_price_range,
    "min_nights": default_min_nights,
    "search_name": "",
    "only_active": False,
}

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

# === Sidebar ===
st.sidebar.header("Filtros")

# Botón reset: setea valores y rerun
if st.sidebar.button("🔄 Resetear filtros"):
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.rerun()

# Widgets usan key=... (no uses default=; el valor viene de session_state)
# Barrio
if "neighbourhood" in df.columns:
    all_neighs = sorted(df["neighbourhood"].dropna().unique().tolist())
    sel_neighs = st.sidebar.multiselect(
        "Barrios (neighbourhood)",
        options=all_neighs,
        key="neighs"
    )
else:
    sel_neighs = []

# Tipo de habitación
if "room_type" in df.columns:
    all_room_types = sorted(df["room_type"].dropna().unique().tolist())
    sel_room_types = st.sidebar.multiselect(
        "Tipo de habitación (room_type)",
        options=all_room_types,
        key="roomtypes"
    )
else:
    sel_room_types = []

# Precio
if "price" in df.columns:
    sel_price = st.sidebar.slider(
        "Rango de precio (USD)",
        min_value=min_price,
        max_value=max_price,
        value=st.session_state["price_range"],
        key="price_range"
    )
else:
    sel_price = None

# Noches mínimas
if "minimum_nights" in df.columns:
    max_min_nights = int(min(30, df["minimum_nights"].max()))
    sel_min_nights = st.sidebar.slider(
        "Noches mínimas máximas permitidas (≤)",
        min_value=1,
        max_value=max_min_nights,
        value=st.session_state["min_nights"],
        key="min_nights"
    )
else:
    sel_min_nights = None

# Extras
sidebar_query = st.sidebar.text_input(
    "Buscar por nombre (contiene):",
    key="search_name"
) if "name" in df.columns else ""

only_active = st.sidebar.checkbox(
    "Solo listings activos (availability_365 > 0)",
    key="only_active"
) if "availability_365" in df.columns else False


# 6) Aplicar filtros sobre una copia para no mutar 'df' original
fdf = df.copy()

if sel_neighs:
    fdf = fdf[fdf["neighbourhood"].isin(sel_neighs)]

if sel_room_types:
    fdf = fdf[fdf["room_type"].isin(sel_room_types)]

if sel_price is not None:
    lo, hi = sel_price
    fdf = fdf[(fdf["price"] >= lo) & (fdf["price"] <= hi)]

if sel_min_nights is not None:
    fdf = fdf[fdf["minimum_nights"] <= sel_min_nights]

# Filtros extra aplicados sobre 'fdf'
if sidebar_query:
    fdf = fdf[fdf["name"].astype(str).str.contains(sidebar_query, case=False, na=False)]

if only_active:
    fdf = fdf[fdf["availability_365"] > 0]

# 7) KPIs: números rápidos arriba del dashboard
st.title("Airbnb — Buenos Aires (CABA)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Listings", f"{len(fdf):,}")

if "price" in fdf.columns and len(fdf):
    col2.metric("Precio promedio", f"${fdf['price'].mean():.0f}")
    col3.metric("Precio mediano", f"${fdf['price'].median():.0f}")
else:
    col2.metric("Precio promedio", "—")
    col3.metric("Precio mediano", "—")

if "minimum_nights" in fdf.columns and len(fdf):
    col4.metric("Noches mín. prom.", f"{fdf['minimum_nights'].mean():.1f}")
else:
    col4.metric("Noches mín. prom.", "—")


# Mapa simple (fallback rápido)
if {"latitude", "longitude"}.issubset(fdf.columns) and len(fdf):
    st.subheader("Mapa de listings")
    st.map(fdf[["latitude", "longitude"]].rename(columns={"latitude": "lat", "longitude": "lon"}))
else:
    st.info("No hay columnas de latitude/longitude o no hay resultados para mostrar en el mapa.")


# 8) Visualizaciones principales (tabs)
st.header("Panel")
tab_precio, tab_barrios, tab_dispersion, tab_mapa, tab_corr, tab_tabla = st.tabs([
    "💵 Precios",
    "🏙️ Barrios",
    "📦 Dispersión",
    "🗺️ Mapas",
    "📊 Correlación",
    "📄 Tabla & Descarga"
])

# --- Tab 1: Histograma de precios (distribución + manejo de cola)
with tab_precio:
    if "price" in fdf.columns and len(fdf):
        st.subheader("Distribución de precios")
        colA, colB, colC = st.columns([1,1,1])
        with colA:
            nbins = st.slider("Bins", 10, 200, 60, step=5)
        with colB:
            use_log_y = st.checkbox("Eje Y logarítmico", value=True, key="hist_logy")
        with colC:
            include_outliers = st.checkbox("Incluir outliers (hasta máx real)", value=False, key="hist_outliers")

        data_price = fdf[["price", "room_type"]].dropna(subset=["price"]).copy()
        if not include_outliers:
            p99 = data_price["price"].quantile(0.99)
            data_price = data_price[data_price["price"] <= p99]

        color_opt = "room_type" if "room_type" in data_price.columns else None

        fig_hist = px.histogram(
            data_price,
            x="price",
            nbins=nbins,
            color=color_opt,
            opacity=0.85,
            labels={"price": "Precio (USD)", "count": "Frecuencia"},
            title="Histograma de precios (filtrado)"
        )
        fig_hist.update_layout(bargap=0.02, height=450)
        if use_log_y:
            fig_hist.update_yaxes(type="log")

        # Línea de mediana como referencia (ayuda a lectura)
        if len(data_price):
            fig_hist.add_vline(x=float(data_price["price"].median()), line_dash="dash", opacity=0.6)

        st.plotly_chart(fig_hist, use_container_width=True)
        st.caption("Tip: activá eje Y log para ver la cola; desactivá outliers (cap 99º) para una lectura más limpia.")
    else:
        st.info("No hay columna `price` o no hay registros filtrados para visualizar.")

# --- Tab 2: Ranking de barrios (cantidad / precio medio / mediano)
with tab_barrios:
    if "neighbourhood" in fdf.columns and len(fdf):
        st.subheader("Ranking de barrios")
        col1, col2, col3 = st.columns([1.2, 1, 1])
        with col1:
            metric = st.selectbox(
                "Métrica",
                ["Cantidad de listings", "Precio medio (USD)", "Precio mediano (USD)"],
                index=0
            )
        with col2:
            top_n = st.slider("Top N", 5, 50, 15, step=5)
        with col3:
            mostrar_valor = st.checkbox("Mostrar valores sobre barras", value=True, key="rank_showvals")

        # Agregación: preparamos todas las métricas y luego elegimos la vista
        df_rank = (
            fdf.groupby("neighbourhood")
               .agg(
                   listings=("neighbourhood", "size"),
                   price_mean=("price", "mean"),
                   price_median=("price", "median")
               )
               .reset_index()
               .dropna(subset=["neighbourhood"])
        )

        # Selección de métrica
        if metric == "Cantidad de listings":
            df_rank["valor"] = df_rank["listings"]
            x_label = "Cantidad de listings"
        elif metric == "Precio medio (USD)":
            df_rank["valor"] = df_rank["price_mean"]
            x_label = "Precio medio (USD)"
        else:
            df_rank["valor"] = df_rank["price_median"]
            x_label = "Precio mediano (USD)"

        # Orden y top N
        df_rank = df_rank.sort_values("valor", ascending=False).head(top_n)

        # Barras horizontales (mejor para etiquetas largas)
        fig_rank = px.bar(
            df_rank.sort_values("valor", ascending=True),
            x="valor",
            y="neighbourhood",
            orientation="h",
            text="valor" if mostrar_valor else None,
            labels={"neighbourhood": "Barrio", "valor": x_label},
            title=f"Top {top_n} barrios por {metric.lower()}"
        )
        # Formato de etiquetas numéricas
        if "Precio" in metric:
            fig_rank.update_traces(texttemplate="%{text:.0f}" if mostrar_valor else None)
            fig_rank.update_xaxes(tickprefix="$", separatethousands=True)
        else:
            fig_rank.update_traces(texttemplate="%{text:,d}" if mostrar_valor else None)
            fig_rank.update_xaxes(separatethousands=True)

        fig_rank.update_layout(height=600, yaxis=dict(title=None))
        st.plotly_chart(fig_rank, use_container_width=True)
        st.caption("Usá 'Cantidad' para oferta y 'Mediana' para comparar precios típicos sin sesgo por outliers.")
    else:
        st.info("No hay columna `neighbourhood` o no hay registros filtrados para visualizar.")

# --- Tab 3: Boxplots (dispersión por grupo)
with tab_dispersion:
    st.subheader("Distribución de Precios (Boxplots)")
    if {"price", "neighbourhood", "room_type"}.issubset(fdf.columns) and len(fdf):
        col1, col2 = st.columns([2, 1])
        with col1:
            agrupar_por = st.selectbox(
                "Agrupar por",
                ["neighbourhood", "room_type"],
                index=0
            )
        with col2:
            incluir_outliers = st.checkbox("Incluir outliers (hasta máx real)", value=False, key="box_outliers")

        df_box = fdf[["price", "neighbourhood", "room_type"]].dropna(subset=["price"]).copy()
        if not incluir_outliers:
            p99 = df_box["price"].quantile(0.99)
            df_box = df_box[df_box["price"] <= p99]

        # Coloreamos por la otra dimensión para ver contraste
        color = "room_type" if agrupar_por == "neighbourhood" else "neighbourhood"

        fig_box = px.box(
            df_box,
            x=agrupar_por,
            y="price",
            color=color,
            points="suspectedoutliers",
            labels={"price": "Precio (USD)", agrupar_por: agrupar_por.title()},
            title=f"Boxplot de precio por {agrupar_por}"
        )
        fig_box.update_yaxes(tickprefix="$", separatethousands=True)
        fig_box.update_layout(height=600, xaxis_tickangle=-30)
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("Faltan `price`, `neighbourhood` o `room_type`, o no hay registros.")

# --- Tab 4: Mapa avanzado (solo Hexágonos / densidad)
with tab_mapa:
    st.subheader("Mapa de densidad (hexágonos)")
    if {"latitude", "longitude"}.issubset(fdf.columns) and len(fdf):
        colA, colB = st.columns([1,1])
        with colA:
            radio_hex = st.slider("Radio hex (m)", 50, 500, 150, step=25, key="hex_radius")
        with colB:
            elev_scale = st.slider("Escala de elevación", 1, 20, 6, step=1, key="hex_elev")

        midpoint = {
            "lat": float(fdf["latitude"].mean()),
            "lon": float(fdf["longitude"].mean())
        }

        layer = pdk.Layer(
            "HexagonLayer",
            data=fdf,
            get_position=["longitude", "latitude"],
            radius=radio_hex,
            elevation_scale=elev_scale,
            elevation_range=[0, 1000],
            extruded=True,
            pickable=True,
        )

        view_state = pdk.ViewState(
            latitude=midpoint["lat"],
            longitude=midpoint["lon"],
            zoom=11,
            pitch=40
        )

        r = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={"text": "{position}"}
        )
        st.pydeck_chart(r, use_container_width=True)
    else:
        st.info("No hay columnas de latitude/longitude o no hay resultados.")

# --- Tab 5: Correlación ---
with tab_corr:
    st.subheader("Matriz de correlación")
    num_cols = fdf.select_dtypes(include="number").columns.tolist()
    num_cols = [c for c in num_cols if c not in ("latitude", "longitude")]
    if len(num_cols) >= 2 and len(fdf):
        corr = fdf[num_cols].corr(numeric_only=True)
        fig_corr = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu", zmin=-1, zmax=1,
            labels={"color": "ρ"}, title="Correlación (Pearson)"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("No hay suficientes columnas numéricas para correlacionar.")

# --- Tab 6: Tabla & Descarga ---
from io import BytesIO
with tab_tabla:
    st.subheader("Muestra de listings filtrados")
    st.dataframe(fdf.head(50))
    st.subheader("Descargar resultados filtrados")

    csv_bytes = fdf.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ CSV", data=csv_bytes, file_name="listings_filtrados.csv", mime="text/csv")

    parquet_buf = BytesIO()
    fdf.to_parquet(parquet_buf, index=False)
    st.download_button("⬇️ Parquet", data=parquet_buf.getvalue(), file_name="listings_filtrados.parquet", mime="application/octet-stream")


