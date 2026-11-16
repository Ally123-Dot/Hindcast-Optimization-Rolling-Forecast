# 文件名: Evaluate_Group1_1h1h_RMSE_CH_DI_LI.py
import os
import glob
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
import warnings

warnings.filterwarnings("ignore")

# 本文件保留高冰区评价实现；三海域的指标公式和评价流程相同。
# 以下仅备注区域适配项，不自动切换海域。
# 切换时须同时对应输入目录、文件匹配、浮标编号前缀、输出文件名及海域提示。
# Group2 边缘冰区：
# RESULT_DIR 末级目录：HyosPy_Uncertainty_Batch_Group2_MIZ_1h1h_1
# search_pattern 文件模式：HyosPy_Group2_1h1h_Forecast_*.csv
# buoy_id 的 replace 前缀：HyosPy_Group2_1h1h_Forecast_
# out_csv 文件名：Group2_MIZ_1h1h_DI_LI_CH_Metrics.csv
# Group3 开阔水域：
# RESULT_DIR 末级目录：HyosPy_Uncertainty_Batch_Group3_OpenWater_1h1h
# search_pattern 文件模式：HyosPy_Group3_1h1h_Forecast_*.csv
# buoy_id 的 replace 前缀：HyosPy_Group3_1h1h_Forecast_
# out_csv 文件名：Group3_OpenWater_1h1h_DI_LI_CH_Metrics.csv
# 输出目录均为原项目 output/Evaluation_Results。
# Group2 原目录末尾的 _1 保留，使用时须与实际目录一致。

# ================= 1. 配置路径 =================
BASE_DIR = r"D:\VScode BO\VScode BO\BO03\data"
RESULT_DIR = os.path.join(BASE_DIR, "..", "output", "HyosPy_Uncertainty_Batch_Group1_HighIce_1h1h")
EVAL_OUT_DIR = os.path.join(BASE_DIR, "..", "output", "Evaluation_Results")

os.makedirs(EVAL_OUT_DIR, exist_ok=True)

# ================= 2. 评价指标 =================

def haversine(lat1, lon1, lat2, lon2):
    """计算大圆球面距离 (单位：公里)"""
    r = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlo = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlo/2)**2
    return 2 * r * np.arcsin(np.sqrt(a))

def calculate_bearing(lat1, lon1, lat2, lon2):
    """计算从起点到终点的正北方位角 (Bearing/Azimuth)，返回弧度 (Radians)"""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.arctan2(x, y)

# 【公式 1】 RMSE (Kampouris et al., 2021)
def calculate_rmse(truth_lats, truth_lons, pred_lats, pred_lons):
    if len(truth_lats) < 1: return np.nan
    squared_errors = [haversine(truth_lats[i], truth_lons[i], pred_lats[i], pred_lons[i]) ** 2 for i in range(len(truth_lats))]
    return np.sqrt(np.mean(squared_errors))

# 【公式 2】 方向指标 DI (Song et al., 2025)
def calculate_di(lat_start, lon_start, t_lat_end, t_lon_end, p_lat_end, p_lon_end):
    """
    DI = cos(θ)，θ为观测向量与模拟向量的夹角
    最佳为 1，最差为 -1
    """
    # 真实轨迹向量方位角
    bearing_obs = calculate_bearing(lat_start, lon_start, t_lat_end, t_lon_end)
    # 模拟轨迹向量方位角
    bearing_sim = calculate_bearing(lat_start, lon_start, p_lat_end, p_lon_end)

    # 夹角 θ
    theta = bearing_sim - bearing_obs
    return np.cos(theta)

# 【公式 3】 长度指标 LI (Song et al., 2025)
def calculate_li(lat_start, lon_start, t_lat_end, t_lon_end, p_lat_end, p_lon_end):
    """
    LI = 1 / (1 + |(SL - OL) / OL|)
    最佳为 1，最差趋近于 0
    """
    OL = haversine(lat_start, lon_start, t_lat_end, t_lon_end) # 观测长度
    SL = haversine(lat_start, lon_start, p_lat_end, p_lon_end) # 模拟长度

    if OL == 0:
        return np.nan # 浮标完全静止，物理长度公式无意义

    return 1.0 / (1.0 + abs((SL - OL) / OL))

# 【公式 4】 凸包包络包含率 (Convex Hull Containment)
def calculate_convex_hull_containment(truth_df, ens_df):
    if truth_df.empty or ens_df.empty: return 0.0
    hit_count, valid_hours = 0, 0

    for _, row in truth_df.iterrows():
        h = row['Hour']
        t_lat, t_lon = row['Lat'], row['Lon']

        hour_data = ens_df[ens_df['Hour'] == h].dropna(subset=['Lat', 'Lon'])
        points = hour_data[['Lon', 'Lat']].values

        if len(points) >= 3:
            try:
                # 微小抖动防止奇异矩阵崩溃
                points_jittered = points + np.random.normal(0, 1e-7, points.shape)
                hull_delaunay = Delaunay(points_jittered)
                if hull_delaunay.find_simplex(np.array([t_lon, t_lat])) >= 0:
                    hit_count += 1
                valid_hours += 1
            except Exception:
                continue
    if valid_hours == 0: return 0.0
    return (hit_count / valid_hours) * 100.0

# ================= 3. 批量评估 =================
def evaluate_batch_results():
    print(f" 启动 [RMSE + DI + LI + 凸包] 指标评估 (Group 1 - 高冰区 1h+1h)...")

    search_pattern = os.path.join(RESULT_DIR, "HyosPy_Random_1h1h_Forecast_*.csv")
    csv_files = glob.glob(search_pattern)
    if not csv_files:
        print(f" 未找到符合模式的文件: {search_pattern}")
        return

    metrics_records = []
    print(f" 发现 {len(csv_files)} 个跑批文件，提取评价数据...")

    for idx, file_path in enumerate(csv_files, 1):
        basename = os.path.basename(file_path)
        # 提取浮标 ID (根据高冰区文件的命名规则)
        buoy_id = basename.replace("HyosPy_Random_1h1h_Forecast_", "").replace(".csv", "").split("_")[0]

        try:
            df = pd.read_csv(file_path)
        except Exception:
            continue

        # 获取 0h (起始点) 用于向量计算
        try:
            start_point = df[(df['Set'] == 'Truth') & (df['Hour'] == 0)].iloc[0]
            lat_start, lon_start = start_point['Lat'], start_point['Lon']
        except IndexError:
            continue # 缺失起始点无法计算向量

        # 获取预报评估期 (Hour >= 1)
        truth = df[(df['Set'] == 'Truth') & (df['Hour'] >= 1)].sort_values('Hour')
        if len(truth) < 1: continue

        truth_lats = truth['Lat'].values
        truth_lons = truth['Lon'].values

        # 观测向量终点 (用于计算 DI 和 LI)
        t_lat_end, t_lon_end = truth_lats[-1], truth_lons[-1]

        # 仅在观测时刻评价，避免次小时输出与整点观测错配。
        ens_df = df[(df['Set'].str.startswith('Set_')) &
                    (df['Hour'].isin(truth['Hour']))]
        ens_sets = ens_df['Set'].unique()
        if len(ens_sets) == 0: continue

        member_RMSE_list = []

        for s in ens_sets:
            member_data = ens_df[ens_df['Set'] == s].sort_values('Hour')
            if len(member_data) == len(truth):
                rmse_j = calculate_rmse(truth_lats, truth_lons, member_data['Lat'].values, member_data['Lon'].values)
                if not np.isnan(rmse_j): member_RMSE_list.append(rmse_j)

        if not member_RMSE_list: continue

        # 计算质心 (Centroid) 的各项指标
        ens_centroid = ens_df.groupby('Hour')[['Lat', 'Lon']].mean().reset_index().sort_values('Hour')
        c_lat_end, c_lon_end = ens_centroid['Lat'].values[-1], ens_centroid['Lon'].values[-1]

        centroid_RMSE = calculate_rmse(truth_lats, truth_lons, ens_centroid['Lat'].values, ens_centroid['Lon'].values)
        centroid_DI = calculate_di(lat_start, lon_start, t_lat_end, t_lon_end, c_lat_end, c_lon_end)
        centroid_LI = calculate_li(lat_start, lon_start, t_lat_end, t_lon_end, c_lat_end, c_lon_end)
        ch_containment = calculate_convex_hull_containment(truth, ens_df)

        metrics_records.append({
            'BuoyID': buoy_id,
            'File_Name': basename,
            'Centroid_RMSE_km': round(centroid_RMSE, 4),
            'Centroid_DI': round(centroid_DI, 4) if not np.isnan(centroid_DI) else np.nan,
            'Centroid_LI': round(centroid_LI, 4) if not np.isnan(centroid_LI) else np.nan,
            'Ensemble_Mean_RMSE_km': round(np.mean(member_RMSE_list), 4),
            'Ensemble_Best_RMSE_km': round(np.min(member_RMSE_list), 4),
            'Convex_Hull_Containment_%': round(ch_containment, 2)
        })

        if idx % 20 == 0: print(f"  已完成评估 {idx} / {len(csv_files)}...")

    # ================= 4. 输出报表 =================
    if not metrics_records:
        print(" 未能成功计算出任何评估记录，请检查输入数据格式。")
        return

    metrics_df = pd.DataFrame(metrics_records)

    # RMSE 分级。
    metrics_df['RMSE_Rating'] = pd.cut(
        metrics_df['Centroid_RMSE_km'],
        bins=[-np.inf, 0.5, 1.0, 2.0, np.inf],
        labels=['Excellent (<0.5km)', 'Good (0.5-1.0km)', 'Fair (1.0-2.0km)', 'Poor (>2.0km)'],
        include_lowest=True
    )

    out_csv = os.path.join(EVAL_OUT_DIR, "Group1_HighIce_1h1h_DI_LI_CH_Metrics.csv")
    metrics_df.to_csv(out_csv, index=False)

    print("-" * 60)
    print(" 评估完成。")
    print(f" 统计学特征概览 (Group 1 高冰区 - {len(metrics_df)}个靶点):")
    print(f"   ▶ 系综质心 RMSE       : {metrics_df['Centroid_RMSE_km'].mean():.4f} km")
    print(f"   ▶ 质心方向指标 DI     : {metrics_df['Centroid_DI'].mean():.4f} (越接近 1 越好)")
    print(f"   ▶ 质心长度指标 LI     : {metrics_df['Centroid_LI'].mean():.4f} (越接近 1 越好)")
    print(f"   ▶ 平均凸包包络率      : {metrics_df['Convex_Hull_Containment_%'].mean():.2f} %")
    print("-" * 60)

if __name__ == "__main__":
    evaluate_batch_results()
