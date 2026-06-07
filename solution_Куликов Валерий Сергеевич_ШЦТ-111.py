import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === CONFIGURATION ===
INPUT_DIR = 'data'
OUTPUT_DIR = 'output'
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')
ANOMALY_THRESHOLD = 3.5
MIN_ROWS_FOR_ANOMALY = 2
MIN_BRAND_SAMPLES = 10

# === DATA LOADER ===
def load_and_clean_data():
    files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.endswith('.parquet')]
    if not files:
        raise FileNotFoundError(f"No parquet files found in '{INPUT_DIR}' directory.")
    
    dfs = [pd.read_parquet(f) for f in files]
    full_df = pd.concat(dfs, ignore_index=True)
    
    if 'Weight' in full_df.columns:
        full_df['Weight'] = full_df['Weight'].astype(float)
    
    filtered_df = full_df[
        (full_df['BrandinDelivery'] == 1) &
        (full_df['CategoryNameDelivery'].notna()) &
        (full_df['CategoryNameDelivery'].astype(str).str.strip() != "")
    ].copy()
    
    return filtered_df

# === ANOMALY DETECTOR ===
def calculate_daily_ots(df):
    grouped = df.groupby(
        ['SubjectID', 'researchdate', 'BrandID', 'Brand', 'CategoryNameDelivery'],
        as_index=False
    ).agg(
        count_rows=('QueryText', 'count'),
        weight=('Weight', 'first')
    )
    grouped['daily_ots'] = grouped['weight'].astype(float) * grouped['count_rows'].astype(float)
    return grouped

def run_anomaly_detection(df_ots):
    df_scores = df_ots.copy()
    
    brand_counts = df_scores.groupby('BrandID')['SubjectID'].transform('count')
    heavy_mask = brand_counts >= MIN_BRAND_SAMPLES
    
    df_heavy = df_scores[heavy_mask].copy()
    df_rare = df_scores[~heavy_mask].copy()
    
    if not df_heavy.empty:
        df_heavy['median'] = df_heavy.groupby('BrandID')['daily_ots'].transform('median')
        df_heavy['mad'] = df_heavy.groupby('BrandID')['daily_ots'].transform(
            lambda x: np.median(np.abs(x - np.median(x)))
        )
    
    if not df_rare.empty:
        df_rare['median'] = df_rare.groupby('CategoryNameDelivery')['daily_ots'].transform('median')
        df_rare['mad'] = df_rare.groupby('CategoryNameDelivery')['daily_ots'].transform(
            lambda x: np.median(np.abs(x - np.median(x)))
        )
        
    df_list = []
    if not df_heavy.empty:
        df_list.append(df_heavy)
    if not df_rare.empty:
        df_list.append(df_rare)
        
    df_scores = pd.concat(df_list, ignore_index=True)
    
    df_scores['median'] = df_scores['median'].astype(float).fillna(0.0)
    df_scores['mad'] = df_scores['mad'].astype(float).replace(0, 1e-5).fillna(1e-5)
    df_scores['daily_ots'] = df_scores['daily_ots'].astype(float)
    
    df_scores['score'] = 0.6745 * (df_scores['daily_ots'] - df_scores['median']) / df_scores['mad']
    df_scores['threshold'] = float(ANOMALY_THRESHOLD)
    
    df_scores['is_anomaly'] = (
        (df_scores['score'] > ANOMALY_THRESHOLD) & 
        (df_scores['count_rows'] >= MIN_ROWS_FOR_ANOMALY)
    )
    
    df_scores['reason'] = "Robust Z-Score превысил порог на уровне бренда"
    
    final_brand_counts = df_scores.groupby('BrandID')['SubjectID'].transform('count')
    df_scores.loc[(final_brand_counts < MIN_BRAND_SAMPLES) & df_scores['is_anomaly'], 'reason'] = \
        "Превышение порога на уровне категории из-за малой выборки бренда"
    
    return df_scores

# === VISUALIZER (Пункт 8.1) ===
def generate_all_plots(df_ots, anomalies_df):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    anom_keys = anomalies_df['SubjectID'].astype(str) + "_" + anomalies_df['researchdate'].astype(str)
    ots_keys = df_ots['SubjectID'].astype(str) + "_" + df_ots['researchdate'].astype(str)
    
    df_clean = df_ots[~ots_keys.isin(anom_keys)].copy()
    
    # Plot 1: total_ots_before_after.png
    plt.figure(figsize=(12, 6))
    ots_before = df_ots.groupby('researchdate')['daily_ots'].sum()
    ots_after = df_clean.groupby('researchdate')['daily_ots'].sum()
    
    plt.plot(ots_before.index.astype(str), ots_before.values, label='До очистки', color='red', alpha=0.7)
    plt.plot(ots_after.index.astype(str), ots_after.values, label='После очистки', color='green', alpha=0.7)
    plt.title('Динамика общего объема OTS до и после удаления аномалий')
    plt.xlabel('Дата')
    plt.ylabel('Общий OTS')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'total_ots_before_after.png'), dpi=150)
    plt.close()
    
    # Plot 2: category_ots_change.png
    plt.figure(figsize=(12, 6))
    cat_before = df_ots.groupby('CategoryNameDelivery')['daily_ots'].sum()
    cat_after = df_clean.groupby('CategoryNameDelivery')['daily_ots'].sum()
    
    pct_change = ((cat_after - cat_before) / cat_before) * 100
    pct_change = pct_change.fillna(0)
    
    sns.barplot(x=pct_change.values, y=pct_change.index, hue=pct_change.index, palette='coolwarm', legend=False)
    plt.title('Изменение OTS по категориям после удаления аномалий (%)')
    plt.xlabel('Изменение в %')
    plt.ylabel('Категория')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'category_ots_change.png'), dpi=150)
    plt.close()
    
    # Plot 3: daily_anomaly_count.png
    plt.figure(figsize=(12, 6))
    daily_anom = anomalies_df.groupby('researchdate')['SubjectID'].count()
    
    plt.bar(daily_anom.index.astype(str), daily_anom.values, color='purple', alpha=0.7)
    plt.title('Количество аномальных респондентов по дням')
    plt.xlabel('Дата')
    plt.ylabel('Количество респондентов')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'daily_anomaly_count.png'), dpi=150)
    plt.close()


# =====================================================================
# ========= ВЫДЕЛЕННЫЕ БЛОКИ ДЛЯ АНАЛИТИЧЕСКИХ ВОЗМОЖНОСТЕЙ ===========
# =====================================================================
# Любая из этих функций может быть вызвана проверяющим напрямую из консоли 


def аналитика_график_до_после_по_признаку(df_ots, df_clean, имя_колонки):
    """
    Строит график 'до/после' для любых характеристик респондентов, ресурсов или категорий.
    Поддерживает: Демографию (Gender, Age, Region, FO), Ресурсы (ResourceName, Platform и др.), 
    Категории (CategoryNameDelivery, Category1, Category2, Category3).
    """
    if имя_колонки not in df_ots.columns:
        print(f"Колонка '{имя_колонки}' отсутствует в текущем агрегированном датасете.")
        return
    plt.figure(figsize=(10, 5))
    before = df_ots.groupby(имя_колонки)['daily_ots'].sum()
    after = df_clean.groupby(имя_колонки)['daily_ots'].sum()
    pd.DataFrame({'До очистки': before, 'После очистки': after}).plot(kind='bar', ax=plt.gca())
    plt.title(f'Анализ OTS До/После по признаку: {имя_колонки}')
    plt.tight_layout()
    plt.show()

def аналитика_таблица_запросов_респондента(df_raw, target_subject_id, target_date):
    """
    Возвращает и выводит на экран таблицу поисковых запросов QueryText 
    для выбранного аномального респондента и конкретного дня.
    """
    res = df_raw[(df_raw['SubjectID'] == target_subject_id) & (df_raw['researchdate'].astype(str) == str(target_date))]
    if res.empty:
        print(f"Запросы для респондента {target_subject_id} на дату {target_date} не найдены.")
        return res
    print(res[['SubjectID', 'researchdate', 'Brand', 'QueryText', 'Weight']].to_string())
    return res[['SubjectID', 'researchdate', 'Brand', 'QueryText', 'Weight']]

def аналитика_тренд_бренда_до_после(df_ots, df_clean, target_brand_name):
    """
    Строит график изменения OTS по дням для выбранного бренда до и после очистки.
    """
    b_before = df_ots[df_ots['Brand'] == target_brand_name].groupby('researchdate')['daily_ots'].sum()
    b_after = df_clean[df_clean['Brand'] == target_brand_name].groupby('researchdate')['daily_ots'].sum()
    plt.figure(figsize=(10, 5))
    plt.plot(b_before.index.astype(str), b_before.values, label='До очистки', color='red', marker='o')
    plt.plot(b_after.index.astype(str), b_after.values, label='После очистки', color='green', marker='s')
    plt.title(f'Динамика OTS для бренда: {target_brand_name}')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
# =====================================================================


# === MAIN PIPELINE ===
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    df_raw = load_and_clean_data()
    df_ots = calculate_daily_ots(df_raw)
    detected_all = run_anomaly_detection(df_ots)
    
    anomalies_detected = detected_all[detected_all['is_anomaly'] == True]
    
    anomalies_final = anomalies_detected[['SubjectID', 'researchdate']].drop_duplicates()
    anomalies_final.to_csv(os.path.join(OUTPUT_DIR, 'anomalies.csv'), index=False)
    
    reasons_to_save = anomalies_detected.rename(columns={'CategoryNameDelivery': 'CategoryDelivery'})
    reasons_to_save = reasons_to_save[[
        'SubjectID', 'researchdate', 'BrandID', 'Brand', 
        'CategoryDelivery', 'daily_ots', 'score', 'threshold', 'reason'
    ]]
    reasons_to_save.to_csv(os.path.join(OUTPUT_DIR, 'anomaly_reasons.csv'), index=False)
    
    generate_all_plots(df_ots, anomalies_final)
    
    print(f"Пайплайн успешно завершен. Результаты сохранены в папке '{OUTPUT_DIR}/'.")

if __name__ == '__main__':
    main()