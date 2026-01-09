
# main.py — Telco Tenure App (flujo completo)

import os
import pandas as pd
import matplotlib.pyplot as plt

from telco_app.data_loader import DataLoader
from telco_app.preprocessing import Preprocessor
from telco_app.eda import EDAAnalyzer
from telco_app.modeling import RegressorTrainer
from telco_app.clustering import Clusterer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_CSV_LOCAL = os.path.join(BASE_DIR, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
DATA_CSV_DRIVE = "/content/drive/MyDrive/WA_Fn-UseC_-Telco-Customer-Churn.csv"

OUT_TABLES = os.path.join(BASE_DIR, "exportaciones", "tablas")
OUT_FIGS = os.path.join(BASE_DIR, "exportaciones", "figuras")
OUT_MODELS = os.path.join(BASE_DIR, "exportaciones", "modelos")

os.makedirs(OUT_TABLES, exist_ok=True)
os.makedirs(OUT_FIGS, exist_ok=True)
os.makedirs(OUT_MODELS, exist_ok=True)


def save_table(df: pd.DataFrame, name: str) -> str:
    path = os.path.join(OUT_TABLES, name)
    df.to_csv(path, index=False)
    return path


def save_fig(name: str) -> str:
    path = os.path.join(OUT_FIGS, name)
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path


def resolve_csv_path() -> str:
    if os.path.exists(DATA_CSV_LOCAL):
        return DATA_CSV_LOCAL
    if os.path.exists(DATA_CSV_DRIVE):
        return DATA_CSV_DRIVE
    raise FileNotFoundError(
        "No encontré el CSV. Coloca el archivo en ./data/ o monta Drive en Colab."
    )


def main():
    csv_path = resolve_csv_path()
    print("CSV:", csv_path)

    # 1) Cargar + validar
    df = DataLoader(csv_path).load()

    # 2) Preprocesamiento (ANTI-LEAKAGE: quitar TotalCharges al predecir tenure)
    prep = Preprocessor(drop_cols=["TotalCharges"])
    X_raw, y = prep.split_xy(df)
    prep.fit(X_raw)

    # 3) EDA (tablas de resumen, exportables)
    eda = EDAAnalyzer()
    eda_tables = {
        "tenure_por_genero.csv": eda.resumen_tenure_por(df, "gender").reset_index(),
        "tenure_por_contrato.csv": eda.resumen_tenure_por(df, "Contract").reset_index(),
        "tenure_por_internet.csv": eda.resumen_tenure_por(df, "InternetService").reset_index(),
        "tenure_por_pago.csv": eda.resumen_tenure_por(df, "PaymentMethod").reset_index(),
    }
    for fname, tdf in eda_tables.items():
        save_table(tdf, fname)

    # Figura: boxplot tenure por contrato
    plt.figure()
    df.boxplot(column="tenure", by="Contract")
    plt.title("Tenure según tipo de contrato")
    plt.suptitle("")
    plt.xlabel("Contract")
    plt.ylabel("tenure")
    plt.xticks(rotation=15, ha="right")
    fig1 = save_fig("tenure_por_contrato.png")

    # 4) Modelado supervisado (CV + comparación + GridSearch KNN)
    trainer = RegressorTrainer(prep)
    metrics_df = trainer.compare(X_raw, y)
    save_table(metrics_df, "metricas_modelos_cv.csv")

    best_params, best_mae = trainer.knn_gridsearch(X_raw, y)
    grid_df = pd.DataFrame([{"best_params": str(best_params), "best_mae_cv": best_mae}])
    save_table(grid_df, "knn_gridsearch_result.csv")

    # 5) No supervisado (KMeans SIN cargos)
    df2 = prep.clean_df(df)
    X_base = df2.drop(columns=["tenure", "customerID"], errors="ignore")
    X_clust_nocost = X_base.drop(columns=["MonthlyCharges", "TotalCharges"], errors="ignore")

    # Preprocesador temporal para clustering
    prep_c = Preprocessor(drop_cols=[])
    prep_c.fit(X_clust_nocost)
    Xc = prep_c.transform(X_clust_nocost)

    best_k, best_sil = Clusterer(k_min=2, k_max=8).best_k_silhouette(Xc)

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    labels = km.fit_predict(Xc)

    df2["Cluster_NoCost"] = labels
    prof = (
        df2.groupby("Cluster_NoCost")[["tenure", "MonthlyCharges", "TotalCharges"]]
           .agg(["count", "mean", "median"])
           .reset_index()
    )
    save_table(prof, "perfil_clusters_nocost.csv")

    # Figura: boxplot tenure por cluster
    plt.figure()
    df2.boxplot(column="tenure", by="Cluster_NoCost")
    plt.title(f"Tenure según cluster (KMeans sin cargos) | k={best_k} | sil={best_sil:.3f}")
    plt.suptitle("")
    plt.xlabel("Cluster_NoCost")
    plt.ylabel("tenure")
    fig2 = save_fig("tenure_por_cluster_nocost.png")

    # 6) Resumen final
    print("\\nExportaciones:")
    print(" -", os.path.join(OUT_TABLES, "metricas_modelos_cv.csv"))
    print(" -", os.path.join(OUT_TABLES, "knn_gridsearch_result.csv"))
    print(" -", os.path.join(OUT_TABLES, "perfil_clusters_nocost.csv"))
    print(" -", fig1)
    print(" -", fig2)
    print("\\nMejor KNN (GridSearch):", best_params, "| MAE:", best_mae)
    print("Métricas CV (comparación):")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
