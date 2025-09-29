# src/data_prep.py
import pandas as pd
from pathlib import Path

# Mapeo de "nombres candidatos" → para estandarizar las columnas a un set fijo.
# Se toma el PRIMER nombre que exista en el CSV para cada columna estándar.
CANDIDATES = {
    "id": ["id", "listing_id"],
    "name": ["name", "listing_name"],
    "neighbourhood": [
        "neighbourhood_cleansed",
        "neighbourhood",
        "neighbourhood_group_cleansed",
        "neighbourhood_group",
        "neighborhood",  # US spelling
    ],
    "room_type": ["room_type"],
    "price": ["price", "price_usd"],
    "minimum_nights": ["minimum_nights", "min_nights"],
    "availability_365": ["availability_365", "availability"],
    "number_of_reviews": ["number_of_reviews", "reviews_count"],
    "last_review": ["last_review"],
    "latitude": ["latitude", "lat"],
    "longitude": ["longitude", "lon", "lng"],
}

STANDARD_ORDER = list(CANDIDATES.keys())

def _pick_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve un DataFrame nuevo con columnas estándar:
    para cada clave en CANDIDATES, toma la primera columna candidata que exista en df.
    """
    out = {}
    for std_col, candidates in CANDIDATES.items():
        for c in candidates:
            if c in df.columns:
                out[std_col] = df[c]
                break
    return pd.DataFrame(out)

def prepare_parquet(csv_path: str, out_parquet: str):
    """
    Lee el CSV (o CSV.GZ), selecciona UNA columna por campo estándar (sin duplicados),
    limpia 'price', asegura lat/lon numéricos y guarda en Parquet.
    """
    # 1) Lectura robusta (detecta compresión por extensión)
    df = pd.read_csv(csv_path, low_memory=False, compression="infer")

    # 2) Selección y estandarización de columnas según CANDIDATES
    df = _pick_columns(df)

    # 3) Limpieza de 'price': quita símbolos y separadores de miles, convierte a numérico y filtra <=0
    if "price" in df.columns:
        s = df["price"].astype(str)
        # Quita $ y espacios/comas (formato tipo $1,234.56). Para formato europeo ver comentario abajo.
        s = s.str.replace(r"\$", "", regex=True)
        s = s.str.replace(r"[,\s]", "", regex=True)
        # Si viene como "1.234,56" (europeo), usar:
        # s = df["price"].astype(str)
        # s = s.str.replace(r"[\s\$]", "", regex=True)
        # s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        df["price"] = pd.to_numeric(s, errors="coerce")
        df = df[df["price"] > 0]

    # 4) Lat/Lon: forzar a numérico (NaN si falla) y descartar filas sin coordenadas
    for col in ("latitude", "longitude"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if {"latitude", "longitude"}.issubset(df.columns):
        df = df.dropna(subset=["latitude", "longitude"])

    # 4.1) (Opcional) Deduplicar por id si viene disponible
    if "id" in df.columns:
        df = df.drop_duplicates(subset="id")

    # 5) Guardado en parquet (crea directorios si no existen)
    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    return out_parquet
