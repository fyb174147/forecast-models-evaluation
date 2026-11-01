"""
MASTER EVALUATION & REPORT PIPELINE (TỔNG HỢP TOÀN DIỆN THỰC NGHIỆM)
Script tự động thực thi toàn bộ quy trình:
1. Tổng hợp kết quả thực nghiệm thành các bảng CSV (Table A.4 -> A.9 & consolidated CSV)
2. Vẽ toàn bộ biểu đồ cột 4 phân vị so sánh Thực nghiệm (Exp) vs Bài báo gốc (Paper)
3. Tính toán toàn bộ ma trận kiểm định thống kê Diebold-Mariano (DM test) trên chuỗi sai số thực nghiệm
4. Xuất 24 bản đồ nhiệt (heatmaps) PDF/PNG và in bảng tóm tắt p-value ra màn hình
"""

import os
import sys
import glob
from pathlib import Path

# Ensure UTF-8 output on all consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent

# ==============================================================================
# BƯỚC 1: TỔNG HỢP KẾT QUẢ THỰC NGHIỆM THÀNH CÁC BẢNG CSV
# ==============================================================================
def step1_aggregate_tables():
    print("\n" + "="*80)
    print("BUOC 1: TONG HOP KET QUA THUC NGHIEM THANH CAC BANG CSV CHUAN BAI BAO")
    print("="*80)

    RESULTS_DIR = PROJECT_ROOT / "results"
    OUTPUT_DIR = PROJECT_ROOT / "tables" / "experiment_tables"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    NAME_MAP = {
        'BiTCN': 'BiTCN', 'Chronos': 'Chronos', 'DeepAR': 'DeepAR', 'DeepFactor': 'DeepFactor',
        'DeepNPTS': 'DeepNPTS', 'Informer': 'Informer', 'MOIRAI': 'MOIRAI', 'MQCNN': 'MQ-CNN',
        'MQRNN': 'MQ-RNN', 'Mamba': 'Mamba', 'NBEATS': 'N-BEATS', 'NHITS': 'NHITS',
        'NLinear': 'NLinear', 'PatchTST': 'PatchTST', 'TCN': 'TCN', 'TemporalFusionTransformer': 'TFT',
        'TiDE': 'TiDE', 'TimeGPT': 'TimeGPT', 'TimeXer': 'TimeXer', 'TimerXL': 'Timer-XL',
        'TimesFM': 'TimesFM', 'TimesNet': 'TimesNet', 'VanillaTransformer': 'Vanilla Transformer',
        'WaveNet': 'WaveNet', 'iTransformer': 'iTransformer', 'NaiveDrift': 'Naive Drift',
        'NaiveMovingAverage': 'Naive MA', 'LightGBM': 'LightGBM', 'KNNRegression': 'KNN Regression',
        'KNN': 'KNN Regression', 'ARIMA': 'AutoARIMA', 'AutoARIMA': 'AutoARIMA'
    }

    SERIES_MAP = {
        'pv_house_1': 'PV 1', 'pv_house_2': 'PV 2', 'pv_house_3': 'PV 3', 'pv_house_4': 'PV 4',
        'bess_house_1': 'BESS 1', 'bess_house_2': 'BESS 2', 'bess_house_3': 'BESS 3', 'bess_house_4': 'BESS 4',
        'germany_load': 'SFH 19', 'london_load': 'LCL 33',
        'zonnedael_customer_8': 'ZHC 8', 'zonnedael_customer_9': 'ZHC 9', 'zonnedael_customer_43': 'ZHC 43',
        'zonnedael_load_8': 'ZHC 8', 'zonnedael_load_9': 'ZHC 9', 'zonnedael_load_43': 'ZHC 43'
    }

    SAMPLING_MAP = {'Sampling_100': '100%', 'Sampling_50': '50%', 'Sampling_33': '33.33%', 'Sampling_25': '25%'}
    SERIES_ORDER = ['SFH 19', 'LCL 33', 'ZHC 8', 'ZHC 9', 'ZHC 43', 'PV 1', 'PV 2', 'PV 3', 'PV 4', 'BESS 1', 'BESS 2', 'BESS 3', 'BESS 4']
    SAMPLING_ORDER = ['100%', '50%', '33.33%', '25%']

    TABLE_GROUPS = {
        'table_a4_statistical_ml': ['KNN Regression', 'LightGBM', 'AutoARIMA', 'Naive Drift', 'Naive MA'],
        'table_a5_mlp': ['N-BEATS', 'NHITS', 'TiDE', 'NLinear'],
        'table_a6_recurrent_gluonts': ['DeepAR', 'DeepFactor', 'MQ-RNN', 'Mamba', 'DeepNPTS'],
        'table_a7_convolutional': ['TCN', 'BiTCN', 'MQ-CNN', 'WaveNet', 'TimesNet'],
        'table_a8_transformer': ['Vanilla Transformer', 'Informer', 'PatchTST', 'TFT', 'TimeXer', 'iTransformer'],
        'table_a9_foundation_models': ['TimeGPT', 'Chronos', 'TimesFM', 'MOIRAI', 'Timer-XL']
    }

    all_summary_files = glob.glob(str(RESULTS_DIR / '**' / 'model_metrics_summary.csv'), recursive=True)
    records = []
    for file_path in all_summary_files:
        if 'Epochs_1_' in file_path or 'Epochs_2_' in file_path:
            continue
        rel_parts = os.path.relpath(file_path, RESULTS_DIR).split(os.sep)
        if len(rel_parts) >= 4:
            dataset, raw_model, raw_sampling = rel_parts[0], rel_parts[1], rel_parts[2]
            clean_model = NAME_MAP.get(raw_model, raw_model)
            clean_sampling = SAMPLING_MAP.get(raw_sampling)
            if not clean_sampling:
                continue
            try:
                df_metric = pd.read_csv(file_path)
                for _, row in df_metric.iterrows():
                    m_val = str(row.get('model', row.get('Model', ''))).lower().strip()
                    clean_series = SERIES_MAP.get(m_val)
                    if clean_series:
                        mae_val = float(row.get('MAE', row.get('mae', 0.0)))
                        rmse_val = float(row.get('RMSE', row.get('rmse', 0.0)))
                        records.append({
                            'Model': clean_model, 'Sampling': clean_sampling, 'Series': clean_series,
                            'MAE': mae_val, 'RMSE': rmse_val
                        })
            except Exception:
                pass

    if not records:
        print("  [!] Khong tim thay du lieu records trong results/")
        return

    df_raw = pd.DataFrame(records)
    df_agg = df_raw.groupby(['Model', 'Sampling', 'Series'])[['MAE', 'RMSE']].mean().reset_index()

    all_table_dfs = []
    for tbl_key, models_in_tbl in TABLE_GROUPS.items():
        formatted_rows = []
        for model in models_in_tbl:
            for samp in SAMPLING_ORDER:
                sub = df_agg[(df_agg['Model'] == model) & (df_agg['Sampling'] == samp)]
                if sub.empty:
                    continue
                row_mae = {'Model': model, 'Sampling': samp, 'Metric': 'MAE'}
                for s in SERIES_ORDER:
                    val = sub[sub['Series'] == s]['MAE'].values
                    row_mae[s] = round(val[0], 3) if len(val) > 0 else np.nan
                formatted_rows.append(row_mae)
                row_rmse = {'Model': model, 'Sampling': samp, 'Metric': 'RMSE'}
                for s in SERIES_ORDER:
                    val = sub[sub['Series'] == s]['RMSE'].values
                    row_rmse[s] = round(val[0], 3) if len(val) > 0 else np.nan
                formatted_rows.append(row_rmse)
        df_tbl = pd.DataFrame(formatted_rows)
        df_tbl.to_csv(OUTPUT_DIR / f'{tbl_key}.csv', index=False)
        if not df_tbl.empty:
            all_table_dfs.append(df_tbl)
        print(f'  [+] Da xuat {tbl_key}.csv: {len(df_tbl)} dong')

    if all_table_dfs:
        df_consolidated = pd.concat(all_table_dfs, ignore_index=True)
        df_consolidated.to_csv(OUTPUT_DIR / 'all_tables_a4_to_a9_consolidated.csv', index=False)
        print(f'  [+] Da xuat all_tables_a4_to_a9_consolidated.csv ({len(df_consolidated)} dong)!')

# ==============================================================================
# BƯỚC 2: VẼ TOÀN BỘ BIỂU ĐỒ CỘT SO SÁNH EXP VS PAPER
# ==============================================================================
def step2_generate_comparison_plots():
    print("\n" + "="*80)
    print("BUOC 2: DANG TAO BIEU DO COT DOI SANH THUC NGHIEM VS BAI BAO GOC...")
    print("="*80)

    PAPER_CSV = PROJECT_ROOT / 'tables' / 'paper_tables' / 'all_tables_a4_to_a9_consolidated.csv'
    EXP_CSV = PROJECT_ROOT / 'tables' / 'experiment_tables' / 'all_tables_a4_to_a9_consolidated.csv'
    BASE_FIGURES_DIR = PROJECT_ROOT / 'figures' / 'comparisons'
    BASE_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not PAPER_CSV.exists() or not EXP_CSV.exists():
        print("  [!] Chua du file consolidated CSV de ve bieu do so sanh.")
        return

    df_p = pd.read_csv(PAPER_CSV)
    df_e = pd.read_csv(EXP_CSV)

    available_models = sorted(list(set(df_e['Model'].unique()).intersection(set(df_p['Model'].unique()))))
    print(f"  [+] Phat hien {len(available_models)} mo hinh san sang de ve bieu do: {available_models}")

    samplings = ['25%', '33.33%', '50%', '100%']
    display_samplings = ['25%', '33%', '50%', '100%']
    series_cols = ['SFH 19', 'LCL 33', 'ZHC 8', 'ZHC 9', 'ZHC 43', 'PV 1', 'PV 2', 'PV 3', 'PV 4', 'BESS 1', 'BESS 2', 'BESS 3', 'BESS 4']
    count = 0

    for model in available_models:
        model_dir = BASE_FIGURES_DIR / model.replace(" ", "_")
        model_dir.mkdir(parents=True, exist_ok=True)

        for ser in series_cols:
            if ser not in df_e.columns or ser not in df_p.columns:
                continue

            exp_mae, exp_rmse, pap_mae, pap_rmse = [], [], [], []
            has_data = False

            for s in samplings:
                e_m = df_e[(df_e['Model'] == model) & (df_e['Sampling'] == s) & (df_e['Metric'] == 'MAE')]
                e_r = df_e[(df_e['Model'] == model) & (df_e['Sampling'] == s) & (df_e['Metric'] == 'RMSE')]
                p_m = df_p[(df_p['Model'] == model) & (df_p['Sampling'] == s) & (df_p['Metric'] == 'MAE')]
                p_r = df_p[(df_p['Model'] == model) & (df_p['Sampling'] == s) & (df_p['Metric'] == 'RMSE')]

                v_em = e_m[ser].values[0] if not e_m.empty and not pd.isna(e_m[ser].values[0]) else np.nan
                v_er = e_r[ser].values[0] if not e_r.empty and not pd.isna(e_r[ser].values[0]) else np.nan
                v_pm = p_m[ser].values[0] if not p_m.empty and not pd.isna(p_m[ser].values[0]) else np.nan
                v_pr = p_r[ser].values[0] if not p_r.empty and not pd.isna(p_r[ser].values[0]) else np.nan

                if not np.isnan(v_em):
                    has_data = True

                exp_mae.append(v_em)
                exp_rmse.append(v_er)
                pap_mae.append(v_pm)
                pap_rmse.append(v_pr)

            if not has_data:
                continue

            x = np.arange(len(samplings))
            width = 0.18
            fig, ax = plt.subplots(figsize=(8, 4.5))

            ax.bar(x - 1.5*width, exp_mae, width, label='Exp MAE', color='#1f77b4')
            ax.bar(x - 0.5*width, pap_mae, width, label='Paper MAE', color='#aec7e8')
            ax.bar(x + 0.5*width, exp_rmse, width, label='Exp RMSE', color='#ff7f0e')
            ax.bar(x + 1.5*width, pap_rmse, width, label='Paper RMSE', color='#ffbb78')

            ax.set_ylabel('Error Value')
            ax.set_title(f'{model} - {ser} (Exp vs Paper across Samplings)')
            ax.set_xticks(x)
            ax.set_xticklabels(display_samplings)
            ax.legend(loc='upper left')
            ax.grid(axis='y', linestyle='--', alpha=0.5)

            out_path = model_dir / f"{ser.replace(' ', '_')}_comparison.png"
            plt.tight_layout()
            plt.savefig(out_path, dpi=200)
            plt.close(fig)
            count += 1

    print(f"  [+] Da ve va luu thanh cong {count} bieu do cot vao figures/comparisons/")

# ==============================================================================
# BƯỚC 3: TÍNH TOÁN VÀ VẼ TOÀN BỘ MA TRẬN DIEBOLD-MARIANO TEST (CẢ 4 DATASETS)
# ==============================================================================
def step3_run_diebold_mariano():
    print("\n" + "="*80)
    print("BUOC 3: DANG CHAY KIEM DINH Y NGHIA THONG KE DIEBOLD-MARIANO (DM TEST)...")
    print("="*80)

    from utils.dm_test import run_dm_for_dataset, DEFAULT_LEGEND_NAMES

    datasets = ["belgium", "germany", "london", "zonnedael"]
    samplings = ["Sampling_25", "Sampling_33", "Sampling_100"]
    total_heatmaps = 0

    for d in datasets:
        for s in samplings:
            outs = run_dm_for_dataset(
                dataset_name=d,
                base_dir=PROJECT_ROOT,
                sampling=s,
                font_size=8,
                legend_names=DEFAULT_LEGEND_NAMES
            )
            total_heatmaps += len(outs)

    print(f"  [+] Tong cong: Da xuat {total_heatmaps} ma tran nhiet (PDF & PNG) tren ca 4 datasets vao results/dm_test/")

# ==============================================================================
# BƯỚC 4: IN BẢNG TỔNG HỢP P-VALUE VÀ KẾT QUẢ ĐỐI SOÁT
# ==============================================================================
def step4_print_summary():
    print("\n" + "="*85)
    print("BUOC 4: BAO CAO TONG HOP KIEM DINH DIEBOLD-MARIANO TREN KET QUA TU THUC NGHIEM")
    print("="*85)

    from utils.dm_test import (
        find_model_dirs,
        read_target_series,
        _compute_common_timestamp_index,
        _align_losses_to_index,
        _build_losses_from_actual_forecast,
        build_dm_matrix
    )

    dataset_dir = PROJECT_ROOT / "results" / "results_belgium"
    model_dirs = find_model_dirs(dataset_dir)
    act_map, fc_map = {}, {}
    for md in model_dirs:
        a, f, _ = read_target_series(md, "PV_house_1_forecast_vs_actual.csv", "Sampling_25")
        if f is not None:
            act_map[md.name] = a
            fc_map[md.name] = f

    c_idx = _compute_common_timestamp_index(fc_map)
    act_map = _align_losses_to_index(act_map, c_idx)
    fc_map = _align_losses_to_index(fc_map, c_idx)
    mae_losses, rmse_losses = {}, {}
    models = list(fc_map.keys())
    for m in models:
        mae_losses[m], rmse_losses[m] = _build_losses_from_actual_forecast(act_map[m], fc_map[m])

    df_mae = pd.DataFrame(build_dm_matrix(mae_losses, models), index=models, columns=models)
    df_rmse = pd.DataFrame(build_dm_matrix(rmse_losses, models), index=models, columns=models)

    summary_rows = []
    for m in models:
        if m == "NaiveDrift": continue
        p_mae = df_mae.loc[m, "NaiveDrift"]
        p_rmse = df_rmse.loc[m, "NaiveDrift"]
        conclusion = "Vuot troi thuc su (p <= 0.05)" if p_mae <= 0.05 else "Tuong duong (p > 0.05)"
        summary_rows.append({
            "Model": m,
            "p-val (MAE)": f"{p_mae:.6f}",
            "p-val (RMSE)": f"{p_rmse:.6f}",
            "Ket luan thong ke vs NaiveDrift": conclusion
        })

    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_string(index=False))

    print("\n" + "="*85)
    print("HOAN TAT 100% QUY TRINH THUC NGHIEM & XUAT DU LIEU NOP BAI!")
    print("1. Cac bang CSV:       tables/experiment_tables/")
    print("2. Bieu do cot Exp/Pap: figures/comparisons/")
    print("3. Ban do nhiet DM:     results/dm_test/belgium/")
    print("4. Goi bao cao Overleaf: docs/overleaf_report.zip")
    print("="*85 + "\n")

if __name__ == "__main__":
    step1_aggregate_tables()
    step2_generate_comparison_plots()
    step3_run_diebold_mariano()
    step4_print_summary()
