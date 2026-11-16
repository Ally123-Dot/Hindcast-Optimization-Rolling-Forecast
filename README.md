# Arctic Oil-Spill BO

Python framework for Bayesian parameter calibration and ensemble trajectory forecasting in Arctic ice-covered waters.

## 项目说明

本仓库公开高冰区的一小时回溯标定与一小时集合轨迹预报框架。程序利用观测轨迹标定漂移参数，再从回溯结束时的观测位置出发，开展下一小时的集合预报。

通过替换样本清单及对应的数据和输出路径，同一框架也可用于边缘冰区和开阔水域。

## 文件组成

- `Base_Forcing_Loader_2024.py`：读取环境场并映射模型所需变量，是预报框架的依赖模块。
- `Run_HyosPy_Batch_Group1_1h1h.py`：读取样本和观测，执行贝叶斯参数标定、集合轨迹预报及结果保存。

## 参数设置

三类海域采用相同的参数搜索范围：

| 参数 | 含义 | 搜索范围 |
| --- | --- | --- |
| `fw` | 风拖曳系数 | 0–0.06 |
| `Kh` | 水平扩散系数 | 0–20 m²/s |
| `k_sic` | 海冰浓度系数 | 0.5–1.5 |

程序采用20个拉丁超立方采样（LHS）初始点评估和100次后续贝叶斯优化搜索迭代，UCB采集函数的κ为2.576。每次模拟使用10,000个粒子，集合预报配置为100个成员。

## 运行环境

需安装Python及以下依赖：

- NumPy
- pandas
- SciPy
- xarray
- scikit-learn
- OpenDrift
- bayesian-optimization
- 与输入文件兼容的NetCDF读取后端

所用依赖版本需支持代码中的接口，包括 `bayes_opt.acquisition.UpperConfidenceBound`。

## 输入数据

运行前需自行准备以下数据：

- 浮标观测CSV，包含 `buoyid`、`year`、`month`、`day`、`hour`、`lat` 和 `lon` 字段。
- 时间窗口清单，包含 `BuoyID` 和 `Target_Time` 字段，也支持使用 `Start_Time`。
- 海洋、海冰和风场NetCDF数据。
- 按程序目录结构组织的100成员集合扰动场。

观测和环境数据需覆盖回溯及预报时段，变量名称、单位和时间坐标应与读取配置一致。本仓库不包含上述运行数据。

## 运行步骤

1. 将两个Python文件放在同一目录。
2. 检查两个文件中的 `BASE_DIR`，并核对环境读取模块中的数据目录和变量映射。
3. 在预报脚本中设置：
   - `LEVEL1_FILE`：观测数据路径。
   - `INPUT_CSV`：时间窗口样本清单。
   - `ENSEMBLE_DIR`：集合扰动场目录。
   - `OUT_DIR`：轨迹输出目录。
4. 在该目录下运行：

```bash
python Run_HyosPy_Batch_Group1_1h1h.py
```

各时间窗口的轨迹结果以CSV格式保存，包含 `Set`、`Hour`、`Lat` 和 `Lon` 字段。程序会跳过已存在结果文件的窗口；如需重新计算，应先备份并移走对应的旧结果文件。

## 用于其他海域

用于边缘冰区或开阔水域时，将 `INPUT_CSV` 替换为目标海域的样本清单，并核对对应的观测、环境场和集合扰动场路径。建议为不同海域设置独立的 `OUT_DIR`。

三类海域使用相同的参数搜索范围，无需修改 `pbounds`。脚本中的Group 1名称及高冰区提示文字为默认入口标识，不承担自动识别海域或切换参数范围的功能。

## 使用状态

作者已在本地数据和运行环境下成功运行代码。其他环境使用时，需配置兼容的依赖和输入数据，并核对数据路径、变量映射及时间覆盖范围。
