# 文件名: Run_HyosPy_Batch_Group1_1h1h.py
import numpy as np
import pandas as pd
import os
import uuid
import glob
import xarray as xr
from datetime import timedelta
from opendrift.models.oceandrift import OceanDrift
from opendrift.readers import reader_netCDF_CF_generic
from opendrift.readers import reader_constant
from scipy.stats import qmc
from bayes_opt import BayesianOptimization
from bayes_opt.acquisition import UpperConfidenceBound
from sklearn.gaussian_process.kernels import Matern
import warnings
import gc
from Base_Forcing_Loader_2024 import get_base_readers

warnings.filterwarnings("ignore")

# ================= 1. 配置与批处理列表 =================
BASE_DIR = r"D:\VScode BO\VScode BO\BO03\data"
LEVEL1_FILE = os.path.join(BASE_DIR, "LEVEL1_2024.csv")

# 100 个伪预报成员。
ENSEMBLE_DIR = os.path.join(BASE_DIR, "ENSEMBLE_100")

# 高冰区批量预报输出目录。
OUT_DIR = os.path.join(BASE_DIR, "..", "output", "HyosPy_Uncertainty_Batch_Group1_HighIce_1h1h")
os.makedirs(OUT_DIR, exist_ok=True)

# 高冰区浮标样本列表。
INPUT_CSV = os.path.join(BASE_DIR, "Random_Sampled_500_Group1_HighIce.csv")

def load_batch_targets():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f" 找不到输入文件: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    time_col = 'Target_Time' if 'Target_Time' in df.columns else ('Start_Time' if 'Start_Time' in df.columns else df.columns[1])
    id_col = 'BuoyID' if 'BuoyID' in df.columns else df.columns[0]

    targets = []
    for _, row in df.iterrows():
        targets.append((float(row[id_col]), str(row[time_col])))
    return targets

BATCH_TARGETS = load_batch_targets()
# 每次模拟使用 10,000 个粒子。
NUM_PARTICLES = 10000

# ================= 2. 粒子输运 =================
class PolarOilDrift(OceanDrift):
    required_variables = {
        'x_sea_water_velocity': {'fallback': 0.0},
        'y_sea_water_velocity': {'fallback': 0.0},
        'x_wind': {'fallback': 0.0},
        'y_wind': {'fallback': 0.0},
        'sea_ice_area_fraction': {'fallback': 0.0},
        'eastward_sea_ice_velocity': {'fallback': 0.0},
        'northward_sea_ice_velocity': {'fallback': 0.0}
    }

    def __init__(self, *args, **kwargs):
        super(PolarOilDrift, self).__init__(*args, **kwargs)
        self.k_sic, self.base_fw, self.user_Kh = 1.0, 0.03, 10.0

    # 粒子推进。
    def update(self):
        N = self.num_elements_active()
        if N == 0: return

        def safe_get(v):
            val = getattr(self.environment, v, 0.0)
            if hasattr(val, 'mask'): val = np.ma.filled(val, 0.0)
            return np.nan_to_num(np.asarray(val), nan=0.0)

        u_oce, v_oce = safe_get('x_sea_water_velocity'), safe_get('y_sea_water_velocity')
        u_ice, v_ice = safe_get('eastward_sea_ice_velocity'), safe_get('northward_sea_ice_velocity')
        u_wind, v_wind = safe_get('x_wind'), safe_get('y_wind')
        sic = safe_get('sea_ice_area_fraction')

        # 论文公式: C'_ice = k_SIC * C_ice,obs
        c_p = np.clip(sic * self.k_sic, 0.0, 1.0)

        # 论文公式: 30/80 阈值分段 k_ice
        k_i = np.zeros_like(c_p)
        mask_miz = (c_p >= 0.3) & (c_p < 0.8)
        if mask_miz.any():
            k_i[mask_miz] = (c_p[mask_miz] - 0.3) / 0.5
        k_i[c_p >= 0.8] = 1.0

        # 论文公式: D_xy = R * sqrt(6*Kh / dt)
        dt = self.time_step.total_seconds()
        diff = np.sqrt(6.0 * self.user_Kh / dt)

        # 论文公式: v_oil = k_ice * v_ice + (1-k_ice)*(v_water + f_w*v_wind + D_xy)
        self.update_positions(k_i * u_ice + (1.0 - k_i) * (u_oce + self.base_fw * u_wind + np.random.uniform(-1, 1, N) * diff),
                              k_i * v_ice + (1.0 - k_i) * (v_oce + self.base_fw * v_wind + np.random.uniform(-1, 1, N) * diff))

# ================= 3. 核心评估与模拟函数 =================
def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    phi1, phi2, dphi, dlo = np.radians(lat1), np.radians(lat2), np.radians(lat2-lat1), np.radians(lon2-lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlo/2)**2
    return 2 * r * np.arcsin(np.sqrt(a))

def run_segment(readers, lat, lon, t, dur, fw, Kh, k_sic, n_part=1,
                time_step_sec=3600, particle_seed=42000):
    o = PolarOilDrift(loglevel=50, seed=particle_seed)
    for r in readers: o.add_reader(r)
    o.add_reader(reader_constant.Reader({'sea_floor_depth_below_sea_level': 100, 'land_binary_mask': 0}))

    o.k_sic, o.base_fw, o.user_Kh = k_sic, fw, Kh
    o.seed_elements(lon=lon, lat=lat, time=t, number=n_part)
    tmp = f"tmp_rand_{uuid.uuid4().hex[:8]}.nc"
    try:
        o.run(duration=timedelta(hours=dur),
              time_step=timedelta(seconds=time_step_sec),
              time_step_output=timedelta(seconds=time_step_sec), outfile=tmp)
        with xr.open_dataset(tmp) as ds:
            if 'trajectory' in ds.dims and n_part > 1:
                # 计算有效粒子位置的中位数。
                lo_res = ds.lon.median(dim='trajectory', skipna=True).values
                la_res = ds.lat.median(dim='trajectory', skipna=True).values
            else:
                lo_res = ds.lon.values.flatten()
                la_res = ds.lat.values.flatten()
            return list(lo_res), list(la_res)
    except Exception as e:
        raise RuntimeError("Trajectory simulation failed") from e
    finally:
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except: pass

# ================= 4. 执行批处理流程 =================
def run_hyospy_batch():
    df_l1 = pd.read_csv(LEVEL1_FILE, low_memory=False)
    df_l1.columns = [c.strip().lower() for c in df_l1.columns]

    print(f" 读取 {len(BATCH_TARGETS)} 条浮标样本！(Group 1 高冰区 | 1h回溯 + 1h预报)")

    for target_idx, (BUOY_ID, start_time_str) in enumerate(BATCH_TARGETS, 1):
        try:
            if "_" in str(start_time_str) and len(str(start_time_str)) == 11:
                START_TIME = pd.to_datetime(start_time_str, format='%Y%m%d_%H')
            else:
                START_TIME = pd.to_datetime(start_time_str, format='mixed')
        except Exception as e:
            print(f" 时间格式解析失败: {start_time_str}。报错信息: {e}")
            continue

        # 回溯窗结束时刻。
        MID_TIME = START_TIME + timedelta(hours=1)

        file_time_str = START_TIME.strftime('%Y%m%d_%H')
        csv_path = os.path.join(OUT_DIR, f"HyosPy_Random_1h1h_Forecast_{int(BUOY_ID)}_{file_time_str}.csv")

        if os.path.exists(csv_path):
            print(f" [断点续传] 样本 [{target_idx}/{len(BATCH_TARGETS)}] ID:{int(BUOY_ID)} 已存在，跳过。")
            continue

        print(f"\n{'='*60}")
        print(f" [Group 1 HighIce 1h+1h {target_idx}/{len(BATCH_TARGETS)}] 正在跑批: ID {int(BUOY_ID)} (启动时间: {START_TIME})")

        true_lats, true_lons = [], []
        # 提取 0、1、2 小时的观测位置。
        for h in range(3):
            b_data = df_l1[df_l1['buoyid'] == BUOY_ID].copy()
            b_data['dt'] = pd.to_datetime(b_data[['year','month','day','hour']])
            idx = abs(b_data['dt'] - (START_TIME + timedelta(hours=h))).idxmin()
            true_lats.append(float(b_data.loc[idx, 'lat']))
            true_lons.append(float(b_data.loc[idx, 'lon']))

        base_readers = get_base_readers(START_TIME)
        uq_records = []
        iter_count = [0]

        # --- 阶段 1: 贝叶斯寻优 (Hindcast 仅 1 小时) ---
        def obj(fw, Kh, k_sic):
            iter_count[0] += 1
            set_name = f"MSO_{iter_count[0]:03d}"

            # 回溯一小时，使用 10,000 个粒子。
            lons, lats = run_segment(base_readers, true_lats[0], true_lons[0], START_TIME, 1, fw, Kh, k_sic, n_part=NUM_PARTICLES)

            save_len_mso = min(len(lons), 2) # 输出仅包含 t=0 和 t=1
            for step in range(save_len_mso):
                uq_records.append({"Set": set_name, "Hour": step, "Lat": true_lats[0] if step==0 else lats[step], "Lon": true_lons[0] if step==0 else lons[step]})

            valid_len = min(len(lons), len(true_lats), 2)

            # 误差计算不包含初始位置。
            errs = [haversine(true_lats[i], true_lons[i], lats[i], lons[i]) for i in range(1, valid_len) if not np.isnan(lons[i])]
            return -np.mean(errs) if errs else -1000

        # 三种冰况均采用正文规定的统一参数边界。
        pbounds = {'fw':(0.0, 0.06), 'Kh':(0.0, 20.0), 'k_sic':(0.5, 1.5)}
        # 本文件保留高冰区批处理入口，其他冰况使用相同参数边界。
        # 输入参数保持原始数值尺度，不进行输入归一化。
        # 负MCSD在GP拟合过程中标准化，alpha为数值正则项。
        utility = UpperConfidenceBound(kappa=2.576)
        opt = BayesianOptimization(f=obj, pbounds=pbounds, random_state=42, verbose=0, acquisition_function=utility)
        opt.set_gp_params(kernel=Matern(length_scale=1.0, nu=2.5),
                          alpha=1e-6, normalize_y=True)

        print(" 执行 MSO 寻优 (1小时高频回溯标定)...")
        # 20 个拉丁超立方初始点，随后执行 100 次贝叶斯优化。
        lhs_keys = ['fw', 'Kh', 'k_sic']
        lhs_unit = qmc.LatinHypercube(d=3, seed=42).random(n=20)
        lhs_points = qmc.scale(lhs_unit,
                               [pbounds[k][0] for k in lhs_keys],
                               [pbounds[k][1] for k in lhs_keys])
        for point in lhs_points:
            opt.probe(params=dict(zip(lhs_keys, point)), lazy=True)
        opt.maximize(init_points=0, n_iter=100)
        theta_star = opt.max['params']

        print(" 寻优完成，生成最优回溯轨迹 (Optimal_Hindcast)...")
        # 生成 1 小时的最优回溯轨迹
        opt_lons, opt_lats = run_segment(base_readers, true_lats[0], true_lons[0], START_TIME, 1, theta_star['fw'], theta_star['Kh'], theta_star['k_sic'], n_part=NUM_PARTICLES)
        for step in range(min(len(opt_lons), 2)):
            uq_records.append({"Set": "Optimal_Hindcast", "Hour": step, "Lat": true_lats[0] if step==0 else opt_lats[step], "Lon": true_lons[0] if step==0 else opt_lons[step]})

        # 压入真实的前 2 小时轨迹点 (0h, 1h, 2h)
        for h in range(3):
            uq_records.append({"Set": "Truth", "Hour": h, "Lat": true_lats[h], "Lon": true_lons[h]})

        # --- 阶段 2: 同步热启动与预报 (Forecast 仅 1 小时) ---
        print(f" 启动 100 套集合预报 (预测未来 1 小时)...")
        forecast_end_time = MID_TIME + timedelta(hours=1)
        target_months = list(set([MID_TIME.strftime('%m'), forecast_end_time.strftime('%m')]))

        for i in range(1, 101):
            set_name = f"Set_{i:03d}"
            set_path = os.path.join(ENSEMBLE_DIR, set_name)

            o_files = [os.path.join(set_path, f"Pseudo_OCEAN_2024_{m}.nc") for m in target_months if os.path.exists(os.path.join(set_path, f"Pseudo_OCEAN_2024_{m}.nc"))]
            w_files = [os.path.join(set_path, f"Pseudo_WIND_2024_{m}.nc") for m in target_months if os.path.exists(os.path.join(set_path, f"Pseudo_WIND_2024_{m}.nc"))]

            if not o_files or not w_files: continue
            if i % 20 == 0: print(f"  正在推演 {set_name} / Set_100 ...")

            try:
                r_o = reader_netCDF_CF_generic.Reader(o_files, standard_name_mapping={'siconc':'sea_ice_area_fraction', 'vxo':'x_sea_water_velocity', 'vyo':'y_sea_water_velocity', 'vxsi':'eastward_sea_ice_velocity', 'vysi':'northward_sea_ice_velocity'})
                r_w = reader_netCDF_CF_generic.Reader(w_files, standard_name_mapping={'u10':'x_wind', 'v10':'y_wind'})

                # 从 t=1 的观测位置启动，预报一小时。
                lons, lats = run_segment([r_o, r_w], true_lats[1], true_lons[1], MID_TIME, 1,
                                         theta_star['fw'], theta_star['Kh'], theta_star['k_sic'],
                                         n_part=NUM_PARTICLES,
                                         time_step_sec=300, particle_seed=50000+i)

                # 保存300秒步长的预报轨迹，Hour以小时为单位。
                for step in range(len(lons)):
                    uq_records.append({"Set": set_name, "Hour": 1+step/12,
                                       "Lat": true_lats[1] if step==0 else lats[step],
                                       "Lon": true_lons[1] if step==0 else lons[step]})
            except Exception as e:
                continue

        df_all = pd.DataFrame(uq_records)
        df_all.to_csv(csv_path, index=False)
        print(f" 完成！最优参数: fw={theta_star['fw']:.4f}, Kh={theta_star['Kh']:.4f}, k_sic={theta_star['k_sic']:.4f}")
        gc.collect()

if __name__ == "__main__":
    run_hyospy_batch()
