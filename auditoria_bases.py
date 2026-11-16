import os
import pandas as pd
import soundfile as sf
import soundata

# 1. Dicionário de Rerotulagem (Três Classes Alvo + Ruído de Fundo/Neutro)
MAPA_CLASSES = {
    # Classes Alvo (Detectores Modulares)
    "car_horn": "1_alvo_buzina",
    "siren": "2_alvo_sirene",
    "crying_baby": "3_alvo_choro_bebe",
    # Opcional para testar balanceamento (descomente se quiser avaliar):
    # "dog": "4_alvo_latido_cao",
    # "dog_bark": "4_alvo_latido_cao",
    
    # Classes Negativas / Ruído Ambiente Contínuo ("Classe X / Outros")
    "rain": "0_fundo_neutro",
    "wind": "0_fundo_neutro",
    "footsteps": "0_fundo_neutro",
    "engine": "0_fundo_neutro",
    "engine_idling": "0_fundo_neutro",
    "air_conditioner": "0_fundo_neutro",
    "street_music": "0_fundo_neutro",
    "children_playing": "0_fundo_neutro",
}

# Mapa oficial de IDs do UrbanSound8K
US8K_ID_MAP = {
    0: "air_conditioner",
    1: "car_horn",
    2: "children_playing",
    3: "dog_bark",
    4: "drilling",
    5: "engine_idling",
    6: "gun_shot",
    7: "jackhammer",
    8: "siren",
    9: "street_music",
}

def ler_cabecalho_wav(caminho):
    if not caminho or not os.path.exists(caminho):
        return pd.Series([None, None, None, False])
    try:
        info = sf.info(caminho)
        return pd.Series([info.samplerate, info.channels, round(info.duration, 2), True])
    except Exception:
        return pd.Series([None, None, None, False])

# --- ETAPA A: CARREGAR ESC-50 ---
print("Processando ESC-50...")
df_esc = pd.read_csv("ESC-50/meta/esc50.csv")
df_esc["classe_original"] = df_esc["category"]
df_esc["nova_classe"] = df_esc["category"].map(MAPA_CLASSES)
df_esc = df_esc.dropna(subset=["nova_classe"]).copy()
df_esc["dataset"] = "ESC-50"
df_esc["filepath"] = df_esc["filename"].apply(lambda x: os.path.join("ESC-50/audio", x))

# --- ETAPA B: CARREGAR URBANSOUND8K (VIA ÍNDICE INTERNO DO SOUNDATA) ---
print("Processando índice oficial do UrbanSound8K...")
ds_us8k = soundata.initialize("urbansound8k", data_home="UrbanSound8K_data")
_ = ds_us8k.clip_ids  # Inicializa o índice interno
index_clips = ds_us8k._index["clips"]

registros_us8k = []
for clip_id, meta in index_clips.items():
    rel_path = meta["audio"][0]  # Ex: "audio/fold1/100032-3-0-0.wav"
    filename = os.path.basename(rel_path)
    fold = int(rel_path.split("fold")[1].split("/")[0])
    class_id = int(filename.replace(".wav", "").split("-")[1])
    class_name = US8K_ID_MAP.get(class_id, "unknown")
    
    registros_us8k.append({
        "dataset": "UrbanSound8K",
        "filepath": os.path.join("UrbanSound8K_data", rel_path),
        "classe_original": class_name,
        "nova_classe": MAPA_CLASSES.get(class_name, None),
        "fold": fold
    })

df_us8k = pd.DataFrame(registros_us8k).dropna(subset=["nova_classe"]).copy()

# --- ETAPA C: CONSOLIDAÇÃO E RELATÓRIO ---
colunas = ["dataset", "filepath", "classe_original", "nova_classe", "fold"]
df_total = pd.concat([df_esc[colunas], df_us8k[colunas]], ignore_index=True)

print("\n=======================================================")
print("1. VOLUMETRIA APÓS REROTULAGEM (PARA OS SLIDES):")
print("=======================================================")
print(pd.crosstab(df_total["nova_classe"], df_total["dataset"], margins=True))

print("\n=======================================================")
print("2. DETALHAMENTO POR CLASSE ORIGINAL:")
print("=======================================================")
print(pd.crosstab([df_total["nova_classe"], df_total["classe_original"]], df_total["dataset"], margins=True))

print("\n=======================================================")
print("3. AUDITORIA FÍSICA DOS ÁUDIOS BAIXADOS (ESC-50):")
print("=======================================================")
df_total[["sample_rate", "channels", "duration_wav", "valido"]] = df_total["filepath"].apply(ler_cabecalho_wav)
df_validos = df_total[df_total["valido"]]
if not df_validos.empty:
    print(df_validos.groupby(["dataset", "sample_rate", "channels", "duration_wav"]).size())

df_total.to_csv("inventario_rerotulado_sdcsa.csv", index=False)
print("\nSucesso! Arquivo 'inventario_rerotulado_sdcsa.csv' gerado.")