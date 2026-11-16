# 文件名: Base_Forcing_Loader_2024.py
import os
import glob
import pandas as pd
from opendrift.readers import reader_netCDF_CF_generic
import xarray as xr
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = r"D:\VScode BO\VScode BO\BO03\data"

# 简单缓存，避免在优化循环中频繁打开同一 netCDF 文件导致底层 HDF/句柄错误
READERS_CACHE = {}

class RobustReader(reader_netCDF_CF_generic.Reader):
    def __init__(self, filename, name=None):
        super(RobustReader, self).__init__(filename)
        self.name = name

    def force_map(self, mapping):
        for internal_name, file_name in mapping.items():
            self.variable_mapping[internal_name] = file_name
            if internal_name not in self.variables:
                self.variables.append(internal_name)
        return self

def _find_monthly_nc_files(root_dirs, y_str, m_str):
    """在目录树中搜索符合 year_month 命名规则的 netCDF 文件。"""
    results = []
    for root in root_dirs:
        if not os.path.isdir(root):
            continue
        for current, _, files in os.walk(root):
            for fn in files:
                if not fn.lower().endswith('.nc'):
                    continue
                if f"{y_str}_{m_str}" in fn or f"{y_str}{m_str}" in fn:
                    results.append(os.path.join(current, fn))
    return sorted(set(results))

def _pick_first_readable(file_list):
    """安全读取探针：尝试多种后端以跳过损坏的文件"""
    for f in file_list:
        for eng in (None, 'netcdf4', 'h5netcdf', 'scipy'):
            try:
                if eng is None:
                    ds = xr.open_dataset(f, decode_times=False)
                else:
                    ds = xr.open_dataset(f, decode_times=False, engine=eng)
                ds.close()
                return f
            except Exception:
                continue
    return None

def get_base_readers(target_time):
    """
    读取当前数据目录的月度强迫文件；缺少文件时抛出异常。
    """
    t = pd.to_datetime(target_time)
    y_str = t.strftime('%Y')
    m_str = t.strftime('%m')
    cache_key = f"{y_str}_{m_str}"

    if cache_key in READERS_CACHE:
        return READERS_CACHE[cache_key]

    # 回溯背景场：ERA5风、GLORYS海流、neXtSIM海冰。
    wind_roots = [os.path.join(BASE_DIR, "WIND", "WIND_MY")]
    ice_roots = [os.path.join(BASE_DIR, "OCEAN", "OCEAN_NEXTSIM")]
    ocean_roots = [os.path.join(BASE_DIR, "OCEAN", "OCEAN_GLO_MY")]
    i_files = _find_monthly_nc_files(ice_roots, y_str, m_str)
    o_files = _find_monthly_nc_files(ocean_roots, y_str, m_str)
    w_files = _find_monthly_nc_files(wind_roots, y_str, m_str)

    if not i_files or not o_files or not w_files:
        raise FileNotFoundError(f"❌ 找不到 {y_str}年{m_str}月 的有效海冰/海流/风场 NC 文件。")

    ice_file = _pick_first_readable(i_files)
    ocean_file = _pick_first_readable(o_files)
    wind_file = _pick_first_readable(w_files)

    if not ice_file or not ocean_file or not wind_file:
        raise FileNotFoundError("❌ 文件存在但完全损坏，无法通过 RobustReader 实例化！")

    # 可打开文件不等于变量齐全；检查后再建立读取器。
    with xr.open_dataset(ice_file, decode_times=False) as ds:
        missing = {'siconc', 'vxsi', 'vysi'} - set(ds.variables)
        if missing:
            raise ValueError(f"强迫文件缺少变量 {sorted(missing)}: {ice_file}")

    r_ice = RobustReader(ice_file, name="neXtSIM_Ice")
    r_ice.force_map({
        'sea_ice_area_fraction': 'siconc',
        'eastward_sea_ice_velocity': 'vxsi',
        'northward_sea_ice_velocity': 'vysi'
    })

    r_oce = RobustReader(ocean_file, name="Ocean")
    r_oce.force_map({
        'eastward_sea_water_velocity': 'uo',
        'northward_sea_water_velocity': 'vo',
        'x_sea_water_velocity': 'uo',
        'y_sea_water_velocity': 'vo'
    })
    r_oce.z_variable = None

    r_wind = RobustReader(wind_file, name="ERA5_Wind")
    r_wind.force_map({
        'x_wind': 'u10',
        'y_wind': 'v10',
        'time': 'valid_time'
    })

    readers = [r_ice, r_oce, r_wind]
    READERS_CACHE[cache_key] = readers
    return readers
