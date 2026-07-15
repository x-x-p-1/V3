"""
====================================================================================
 V3_TEST_信号捕捉器2_0.py — 体制感知型多空信号捕捉器
====================================================================================
 【版本 2.0 核心改进】

 1. 【新增】LowVol_Short 做空信号
    - 利用 Low HV20 下 69.2% 先跌概率的统计优势
    - 触发: HV20_perc < 0.33 + vol_perc < 0.30 + close < MA60

 2. 【新增】体制感知权重调整 (Regime-Aware Weighting)
    - 每条信号根据当前 HV20 分位自动调整权重
    - 做多信号在高波动下权重↑，做空信号在低波动下权重↑
    - 信号评分 = 基础分 × 体制权重因子

 3. 【新增】动态 DTE 建议
    - 根据当前波动率体制推荐最优期权期限
    - Low HV20 → 35-45天 | Mid HV20 → 30-35天
    - High HV20 → 20-25天 | Very High HV20 → 15-20天

 4. 【修复】Storm Eye 做空信号不再排除 Low HV20
    - 原版将 HV_Perc >= 0.35 作为必要条件，排除了最强的做空环境
    - 2.0 版扩展为: 允许 Low HV20 下触发，并给出更高权重

 5. 【增强】Ice Point 双冰点方向提示
    - 原版在 Low HV20 下做多，但数据指向应做空
    - 2.0 版保留信号的同时，给出体制反转提示

 6. 【增强】完整体制上下文输出
    - 每条信号附带: 当前HV20值、HV20分位、波动率体制、置信度
    - 显示转换概率矩阵，预判未来1-2天的体制走向

 ------------------------------------------------------------------------------------
 【文件定位】
   volatility_lab/regime_engine.py  ← 波动率体制分析引擎（底层）
   V3_TEST_信号捕捉器2_0.py        ← 本文件（信号层，依赖regime_engine）
   策略回测_做多信号_v2.py          ← 策略执行层（依赖本文件）

 【数据流】
   RegimeEngine.analyze() → HV20分位/体制 → signal_catcher_2_0 → 信号+权重+DTE建议
====================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter
from gm.api import *


warnings.filterwarnings('ignore')
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ========== 绘图配置 ==========
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 掘金量化配置 ==========
TOKEN = '24b8511a3e2b55de017226f956454e1681a10446'
SYMBOL = 'SHSE.000852'  # 中证1000指数
set_token(TOKEN)

# ========== 日期常量 ==========
NOW = datetime.now()
END_TIME = NOW.strftime('%Y-%m-%d 16:00:00')
TODAY_STR = NOW.strftime('%Y-%m-%d')
# ============================================================
# 第〇章：编译版 RegimeEngine（无需外部import）
# ============================================================


@dataclass
class RegimeStats:
    """体制分析结果容器"""
    symbol: str = ''
    name: str = ''
    low_vol_threshold: float = 0.0
    high_vol_threshold: float = 0.0
    low_vol_pct: float = 0.0
    mid_vol_pct: float = 0.0
    high_vol_pct: float = 0.0
    regime_stats: Dict[str, Dict] = field(default_factory=dict)
    transition_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    regime_stability: float = 0.0
    avg_regime_duration_days: Dict[str, float] = field(default_factory=dict)
    vol_clustering_coeff: float = 0.0
    high_vol_streak_stats: Dict = field(default_factory=dict)
    current_regime: str = ''
    current_regime_confidence: float = 0.0
    signal_friendly: Dict[str, float] = field(default_factory=dict)
    # ---- 扩展字段 ----
    current_hv20: float = 0.0
    current_hv20_percentile: float = 0.0


class RegimeEngine:
    """波动率体制分析引擎（动态阈值版，内嵌于信号捕捉器2.0）"""

    def analyze(self, df: pd.DataFrame, symbol: str = '', name: str = '') -> RegimeStats:
        if len(df) < 60:
            return RegimeStats(symbol=symbol, name=name)

        result = RegimeStats(symbol=symbol, name=name)
        ret = df['close'].pct_change()
        hv20 = ret.rolling(20).std() * np.sqrt(252)
        hv_clean = hv20.dropna()

        if len(hv_clean) < 60:
            return result

        # 动态阈值：33.3%/66.7%分位
        low_thresh = float(np.percentile(hv_clean, 100 / 3))
        high_thresh = float(np.percentile(hv_clean, 200 / 3))
        result.low_vol_threshold = low_thresh
        result.high_vol_threshold = high_thresh

        def classify_regime(hv):
            if hv < low_thresh:
                return '低波动'
            elif hv < high_thresh:
                return '中等波动'
            else:
                return '高波动'

        regimes = hv20.dropna().apply(classify_regime)
        regime_counts = Counter(regimes)
        total = len(regimes)
        result.low_vol_pct = regime_counts.get('低波动', 0) / total * 100
        result.mid_vol_pct = regime_counts.get('中等波动', 0) / total * 100
        result.high_vol_pct = regime_counts.get('高波动', 0) / total * 100

        # ---- 各体制下统计特征 ----
        regime_data = {}
        for regime_name in ['低波动', '中等波动', '高波动']:
            regime_mask = regimes == regime_name
            regime_ret = ret.loc[regime_mask.index[regime_mask]]
            regime_vol = hv20.loc[regime_mask.index[regime_mask]]
            if len(regime_ret) == 0:
                continue
            regime_data[regime_name] = {
                '天数': int(regime_mask.sum()),
                '占比(%)': round(regime_mask.sum() / total * 100, 1),
                '平均波动率(%)': round(float(regime_vol.mean() * 100), 2),
                '日均收益率(%)': round(float(regime_ret.mean() * 100), 4),
                '上涨天数占比(%)': round(float((regime_ret > 0).sum() / len(regime_ret) * 100), 1),
                '年化夏普': round(float(regime_ret.mean() / regime_ret.std() * np.sqrt(252)), 2) if regime_ret.std() > 0 else 0,
            }
        result.regime_stats = regime_data

        # ---- 当前体制 ----
        if len(hv20) > 0:
            current_hv = float(hv20.iloc[-1])
            result.current_regime = classify_regime(current_hv)
            result.current_hv20 = current_hv
            result.current_hv20_percentile = float((hv20.dropna() < current_hv).mean())

            hv_history = hv20.dropna().values
            if len(hv_history) > 0:
                if result.current_regime == '低波动':
                    confidence = 1 - abs(current_hv - 0) / low_thresh if low_thresh > 0 else 0.5
                elif result.current_regime == '高波动':
                    confidence = 1 - min(abs(current_hv - high_thresh) / high_thresh, 1)
                else:
                    mid_point = (low_thresh + high_thresh) / 2
                    dist_from_center = abs(current_hv - mid_point)
                    confidence = 1 - dist_from_center / (high_thresh - low_thresh) if (high_thresh - low_thresh) > 0 else 0.5
                result.current_regime_confidence = max(0, min(1, confidence)) * 100

        # ---- 转换矩阵 ----
        regimes_series = regimes.to_frame('regime')
        regimes_series['next_regime'] = regimes_series['regime'].shift(-1)
        regimes_series = regimes_series.dropna()
        trans_matrix = {}
        for curr_reg in ['低波动', '中等波动', '高波动']:
            curr_data = regimes_series[regimes_series['regime'] == curr_reg]
            if len(curr_data) == 0:
                continue
            next_counts = Counter(curr_data['next_regime'])
            total_curr = len(curr_data)
            trans_matrix[curr_reg] = {
                next_r: round(next_counts.get(next_r, 0) / total_curr * 100, 1)
                for next_r in ['低波动', '中等波动', '高波动']
            }
        result.transition_matrix = trans_matrix

        # ---- 稳定性 ----
        stability_vals = []
        for curr_reg, trans in trans_matrix.items():
            if curr_reg in trans:
                stability_vals.append(trans[curr_reg])
        result.regime_stability = np.mean(stability_vals) if stability_vals else 0

        return result


# ============================================================
# 第一章：从掘金量化获取数据
# ============================================================

def fetch_data(start='2018-01-01') -> pd.DataFrame:
    """获取中证1000日线数据"""
    print("[1/6] 从掘金量化获取中证1000指数数据...")
    start_time = f'{start[:10]} 09:00:00'
    try:
        df_daily = history(symbol=SYMBOL, frequency='1d',
                           start_time=start_time, end_time=END_TIME,
                           fields='open,close,high,low,volume,bob,eob',
                           adjust=ADJUST_NONE, df=True)
        df_daily = df_daily.sort_values('eob').reset_index(drop=True)
        df_daily['date'] = pd.to_datetime(df_daily['eob']).dt.tz_localize(None)
        df = df_daily[['date', 'open', 'close', 'high', 'low', 'volume']].copy()
        df.set_index('date', inplace=True)
        for col in ['open', 'close', 'high', 'low', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()
        print(f"  获取到 {len(df)} 个日线数据, 范围: {df.index.min()} ~ {df.index.max()}")
        return df
    except Exception as e:
        print(f"  掘金量化API调用失败: {e}")
        print("  请检查token和网络连接")
        sys.exit(1)


# ============================================================
# 第二章：因子计算 + 全部信号
# ============================================================

def compute_all_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    在日线DataFrame上计算所有信号指标。
    返回原df的扩展版本（新增信号列和因子列）。
    """
    df = df.copy()

    # ---- 通用因子 ----
    df['ret'] = df['close'].pct_change()
    df['HV20'] = df['ret'].rolling(20).std() * np.sqrt(252)
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    df['vol_ma60'] = df['volume'].rolling(60).mean()
    df['vol_ratio'] = df['vol_ma20'] / df['vol_ma60']
    df['MA5'] = df['close'].rolling(5).mean()
    df['MA20'] = df['close'].rolling(20).mean()
    df['MA60'] = df['close'].rolling(60).mean()
    df['MA120'] = df['close'].rolling(120).mean()
    df['MA180'] = df['close'].rolling(180).mean()
    df['MA250'] = df['close'].rolling(250).mean()

    # ---- 百分位因子（基于120天滚动窗口） ----
    WINDOW = 120
    df['HV20_perc'] = df['HV20'].rolling(WINDOW, min_periods=WINDOW).apply(
        lambda x: (x.iloc[-1] < x).mean(), raw=False)
    df['vol_perc'] = df['vol_ratio'].rolling(WINDOW, min_periods=WINDOW).apply(
        lambda x: (x.iloc[-1] < x).mean(), raw=False)

    # ---- 偏度ZScore / 成交量ZScore（用于DeepV） ----
    df['Bias'] = (df['close'] - df['MA60']) / df['MA60']
    df['Bias_Rolling_Mean'] = df['Bias'].rolling(252, min_periods=60).mean()
    df['Bias_Rolling_Std'] = df['Bias'].rolling(252, min_periods=60).std()
    df['Bias_ZScore'] = (df['Bias'] - df['Bias_Rolling_Mean']) / df['Bias_Rolling_Std']

    df['Vol_Ratio_V'] = df['volume'] / df['vol_ma60']
    df['Vol_Ratio_Mean'] = df['Vol_Ratio_V'].rolling(252, min_periods=60).mean()
    df['Vol_Ratio_Std'] = df['Vol_Ratio_V'].rolling(252, min_periods=60).std()
    df['Vol_ZScore'] = (df['Vol_Ratio_V'] - df['Vol_Ratio_Mean']) / df['Vol_Ratio_Std']

    # ================================================================
    # 【信号A】DeepV 深V反包（做多★★★）
    # ================================================================
    df['In_Killzone'] = df['Bias_ZScore'] < -2.0
    df['Killzone_Alert'] = df['In_Killzone'].rolling(5).max() > 0
    df['Volume_Surge'] = df['Vol_ZScore'] > 2.0
    df['Right_Side_Breakout'] = (df['close'] > df['MA5']) & (df['close'] > df['open'])

    df['DeepV_Raw'] = df['Killzone_Alert'] & df['Volume_Surge'] & df['Right_Side_Breakout']
    df['DeepV_Clean'] = df['DeepV_Raw'] & (~df['DeepV_Raw'].shift(1).fillna(False))

    # ================================================================
    # 【信号B】Ice Point 双冰点
    # ================================================================
    df['Ice_Signal_Raw'] = (df['HV20_perc'] < 0.3) & (df['vol_perc'] < 0.3) & (df['close'] > df['MA120'])
    df['Ice_Signal'] = df['Ice_Signal_Raw']
    df['Ice_Regime_Reverse'] = df['Ice_Signal'] & (df['HV20_perc'] < 0.20)

    # ================================================================
    # 【信号C】Storm Eye 风暴眼（做空）
    # ================================================================
    df['HV_Perc_120'] = df['HV20'].rolling(120, min_periods=60).apply(
        lambda x: (x.iloc[-1] < x).mean(), raw=False)
    df['HV_Perc_180'] = df['HV20'].rolling(180, min_periods=90).apply(
        lambda x: (x.iloc[-1] < x).mean(), raw=False)
    df['vol_wan'] = df['volume'] / 10000
    df['vol_ma120'] = df['vol_wan'].rolling(120).mean()
    df['vol_ma180'] = df['vol_wan'].rolling(180).mean()
    df['vol_diff_pct'] = abs(df['vol_ma120'] - df['vol_ma180']) / df['vol_ma180']

    # 原版风暴眼（窄域）
    cond_vol_original = (df['HV_Perc_120'] >= 0.35) & (df['HV_Perc_120'] <= 0.65) &                         (df['HV_Perc_180'] >= 0.35) & (df['HV_Perc_180'] <= 0.65)
    cond_vol_volume = df['vol_diff_pct'] < 0.04
    df['Storm_Eye_Original_Raw'] = cond_vol_original & cond_vol_volume
    df['Storm_Eye_Original'] = df['Storm_Eye_Original_Raw'] & df['Storm_Eye_Original_Raw'].shift(1).fillna(False)

    # 扩展型风暴眼（宽域）
    cond_vol_extended = (df['HV_Perc_120'] <= 0.65) &                         (df['HV_Perc_180'] <= 0.65) &                         (df['vol_diff_pct'] < 0.04)
    df['Storm_Eye_Extended_Raw'] = cond_vol_extended & (df['HV20_perc'] < 0.40)
    df['Storm_Eye_Extended'] = df['Storm_Eye_Extended_Raw'] & (~df['Storm_Eye_Extended_Raw'].shift(1).fillna(False))

    # ================================================================
    # 【信号D·新增】LowVol_Short（做空★★）
    # ================================================================
    df['LowVol_Short_Raw'] = (df['HV20_perc'] < 0.33) &                               (df['vol_perc'] < 0.30) &                               (df['close'] < df['MA20']) & (df['close'] > df['MA120']) &                               (df['ret'] < -0.005)
    df['LowVol_Short'] = df['LowVol_Short_Raw'] & (~df['LowVol_Short_Raw'].shift(1).fillna(False))

    # ================================================================
    # 【信号E·新增】HV20_Breakdown（做空★★★）
    # ================================================================
    df['HV20_Expand'] = df['HV20'] > df['HV20'].shift(5) * 1.15
    df['HV20_Breakdown_Raw'] = (df['HV20_perc'] < 0.40) & df['HV20_Expand'] & \
                               (df['ret'] < -0.01) & (df['close'] > df['MA20']) & \
                               (df['close'] > df['MA120']) & (df['close'] > df['MA180'])
    df['HV20_Breakdown'] = df['HV20_Breakdown_Raw'] & (~df['HV20_Breakdown_Raw'].shift(1).fillna(False))

    # ================================================================
    # 信号汇总分类
    # ================================================================
    df['Long_Weak'] = df['Ice_Signal']
    df['Long_Strong'] = df['DeepV_Clean']
    df['Short_Weak'] = df['Storm_Eye_Original'] & (~df['Storm_Eye_Original'].shift(1).fillna(False))
    df['Short_Strong'] = df['Storm_Eye_Original'] & df['Storm_Eye_Original'].shift(1).fillna(False)
    df['Short_LowVol'] = df['LowVol_Short']
    df['Short_Extended'] = df['Storm_Eye_Extended']
    df['Short_Breakdown'] = df['HV20_Breakdown']

    # ================================================================
    # 体制感知权重
    # ================================================================
    df['Regime_Weight_Long'] = np.where(
        df['HV20_perc'] > 0.67, 1.5,
        np.where(df['HV20_perc'] < 0.33, 0.5, 1.0)
    )
    df['Regime_Weight_Short'] = np.where(
        df['HV20_perc'] < 0.33, 1.5,
        np.where(df['HV20_perc'] > 0.67, 0.5, 1.0)
    )

    # ================================================================
    # 动态DTE建议
    # ================================================================
    def suggest_dte(hv20_perc):
        if hv20_perc > 0.90:
            return 15, '极短'
        elif hv20_perc > 0.67:
            return 20, '短'
        elif hv20_perc > 0.33:
            return 30, '中'
        else:
            return 40, '长'

    dte_results = df['HV20_perc'].apply(suggest_dte)
    df['Suggested_DTE'] = dte_results.apply(lambda x: x[0])
    df['DTE_Label'] = dte_results.apply(lambda x: x[1])

    # ================================================================
    # 综合信号标签
    # ================================================================
    def signal_label_v2(row):
        labels = []
        if row['Long_Strong']:
            labels.append('做多★★★[深V]')
        elif row['Long_Weak']:
            if row.get('Ice_Regime_Reverse', False):
                labels.append('做多★[双冰·方向反转]')
            else:
                labels.append('做多★[双冰]')
        if row['Short_Breakdown']:
            labels.append('做空★★★[波动加速]')
        if row['Short_Strong']:
            labels.append('做空★★★[风暴眼·连续]')
        if row['Short_LowVol']:
            labels.append('做空★★[低波做空]')
        if row['Short_Extended']:
            labels.append('做空★★[扩展风暴眼]')
        if row['Short_Weak']:
            labels.append('做空★[风暴眼·首日]')
        if not labels:
            return ''
        return ' | '.join(labels)

    df['Signal_V2'] = df.apply(signal_label_v2, axis=1)

    def calc_score(row):
        score = 0
        if row['Long_Strong']:
            score += 3 * row['Regime_Weight_Long']
        if row['Long_Weak']:
            if row.get('Ice_Regime_Reverse', False):
                score -= 1 * row['Regime_Weight_Short']
            else:
                score += 1 * row['Regime_Weight_Long']
        if row['Short_Breakdown']:
            score -= 3 * row['Regime_Weight_Short']
        if row['Short_Strong']:
            score -= 3 * row['Regime_Weight_Short']
        if row['Short_LowVol']:
            score -= 2 * row['Regime_Weight_Short']
        if row['Short_Extended']:
            score -= 2 * row['Regime_Weight_Short']
        if row['Short_Weak']:
            score -= 1 * row['Regime_Weight_Short']
        return round(score, 2)

    df['Composite_Score'] = df.apply(calc_score, axis=1)
    df['Direction'] = df['Composite_Score'].apply(
        lambda s: '做多' if s > 0 else ('做空' if s < 0 else '中性')
    )
    df['Confidence'] = df['Composite_Score'].abs().apply(
        lambda s: '高' if s >= 4 else ('中' if s >= 2 else '低')
    )

    return df


# ============================================================
# 第二章+：江恩网格上下文标注
# ============================================================

def add_grid_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    为信号数据添加江恩网格上下文信息。
    每条信号将得到：
      - Grid_Cell_Pos:      在格子中的位置 0(底)~1(顶)
      - Grid_Nearest_Fib:   最近的Fib比率名称
      - Grid_Fib_Dist_pct:  距最近Fib线的百分比距离(+表在上方)
      - Grid_Cell_Triggered: 该格子在信号日前是否已被触发过
      - Grid_Zone_Label:    简明区域标签
    """
    try:
        from golden_grid import build_gann_grid_cells, grid_date_to_x
        from shapely.geometry import LineString, Point
    except ImportError as e:
        print(f"  ⚠ 网格上下文不可用 (缺少依赖: {e})")
        df['Grid_Cell_Pos'] = np.nan
        df['Grid_Nearest_Fib'] = ''
        df['Grid_Fib_Dist_pct'] = np.nan
        df['Grid_Cell_Triggered'] = False
        df['Grid_Zone_Label'] = ''
        return df

    # ---- 构建江恩网格（一次性） ----
    cells = build_gann_grid_cells()
    print(f"  江恩网格: {len(cells)} 个格子")

    # ---- 预计算所有格子触发状态（价格历史上是否穿过） ----
    triggered_cells = set()
    for i, cell in enumerate(cells):
        poly = cell["polygon"]
        min_x, min_y, max_x, max_y = poly.bounds
        for idx, row in df.iterrows():
            kx = grid_date_to_x(idx)
            klow, khigh = row['low'], row['high']
            if min_x <= kx <= max_x:
                kline = LineString([(kx, klow), (kx, khigh)])
                if poly.intersects(kline):
                    triggered_cells.add(i)
                    break
    print(f"  已触发格子: {len(triggered_cells)} / {len(cells)}")

    # ---- 逐行计算网格上下文 ----
    df['Grid_Cell_Pos'] = np.nan
    df['Grid_Nearest_Fib'] = ''
    df['Grid_Fib_Dist_pct'] = np.nan
    df['Grid_Cell_Triggered'] = False
    df['Grid_Zone_Label'] = ''

    for idx, row in df.iterrows():
        x = grid_date_to_x(idx)
        price = row['close']
        pt = Point(x, price)
        found_cell = None
        found_idx = -1
        for ci, cell in enumerate(cells):
            if cell["polygon"].intersects(pt):
                found_cell = cell
                found_idx = ci
                break
        if found_cell is None:
            continue

        cell_range = found_cell['max_y'] - found_cell['min_y']
        pos = (price - found_cell['min_y']) / cell_range if cell_range > 0 else 0.5
        pos = max(0.0, min(1.0, pos))

        nearest_ratio = min(
            found_cell['fib_lines'].keys(),
            key=lambda r: abs(price - found_cell['fib_lines'][r])
        )
        nearest_price = found_cell['fib_lines'][nearest_ratio]
        fib_dist = (price - nearest_price) / nearest_price * 100

        if pos >= 0.7:
            zone = '格顶'
        elif pos <= 0.3:
            zone = '格底'
        else:
            zone = '格中'

        df.at[idx, 'Grid_Cell_Pos'] = round(pos, 3)
        df.at[idx, 'Grid_Nearest_Fib'] = f'Fib-{nearest_ratio}'
        df.at[idx, 'Grid_Fib_Dist_pct'] = round(fib_dist, 2)
        df.at[idx, 'Grid_Cell_Triggered'] = found_idx in triggered_cells
        df.at[idx, 'Grid_Zone_Label'] = zone

    n_found = df['Grid_Zone_Label'].ne('').sum()
    print(f"  网格定位成功: {n_found}/{len(df)} 行")
    return df


# ============================================================
# 第二章++：网格感知评分加权（Phase 2）
# ============================================================

def apply_grid_score_adjustments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phase 2: 网格感知评分加权
    根据价格在江恩网格中的位置，对信号评分进行微调。

    调整逻辑:
      【做空信号】
        - 格顶(>=70%) + Fib-0.618/0.764（阻力位）→ 加分 -1.0（更强做空）
        - 格顶(>=70%) + 其他Fib           → 加分 -0.5
        - 格底(<=30%)（支撑位）            → 扣分 +0.5（减弱做空）

      【做多信号】
        - 格底(<=30%) + Fib-0.236/0.382（支撑位）→ 加分 +1.0（更强做多）
        - 格底(<=30%) + 其他Fib           → 加分 +0.5
        - 格顶(>=70%)（阻力位）            → 扣分 -0.5（减弱做多）
    """
    df = df.copy()
    df['Grid_Score_Bonus'] = 0.0

    # 只有有信号且网格定位成功的行才调整
    mask = (df['Signal_V2'] != '') & (df['Grid_Zone_Label'] != '') & (~pd.isna(df['Grid_Cell_Pos']))
    adjusted_count = 0

    for idx in df[mask].index:
        row = df.loc[idx]
        bonus = 0.0
        zone = row['Grid_Zone_Label']
        fib = row['Grid_Nearest_Fib']
        sig = str(row['Signal_V2'])
        is_short = '做空' in sig
        is_long = '做多' in sig

        if is_short:
            if zone == '格顶':  # 阻力位做空 → 加分（评分更负）
                if fib in ('Fib-0.618', 'Fib-0.764'):
                    bonus = -1.0   # 强阻力位，强力做空
                else:
                    bonus = -0.5   # 一般阻力位
            elif zone == '格底':  # 支撑位做空 → 扣分（评分回调）
                bonus = +0.5

        if is_long:
            if zone == '格底':  # 支撑位做多 → 加分（评分更正）
                if fib in ('Fib-0.236', 'Fib-0.382'):
                    bonus = +1.0   # 强支撑位，强力做多
                else:
                    bonus = +0.5   # 一般支撑位
            elif zone == '格顶':  # 阻力位做多 → 扣分（评分回调）
                bonus = -0.5

        if bonus != 0.0:
            df.at[idx, 'Grid_Score_Bonus'] = bonus
            df.at[idx, 'Composite_Score'] = round(row['Composite_Score'] + bonus, 2)
            adjusted_count += 1

    print(f"  网格评分调整: {adjusted_count}/{mask.sum()} 条信号评分被调整")

    # 重新计算方向和置信度
    df['Direction'] = df['Composite_Score'].apply(
        lambda s: '做多' if s > 0 else ('做空' if s < 0 else '中性')
    )
    df['Confidence'] = df['Composite_Score'].abs().apply(
        lambda s: '高' if s >= 4 else ('中' if s >= 2 else '低')
    )

    return df


# ============================================================
# 第三章：体制分析 + 信号整合
# ============================================================

def get_regime_context(df: pd.DataFrame) -> Dict:
    """获取当前波动率体制上下文"""
    engine = RegimeEngine()
    stats = engine.analyze(df, symbol=SYMBOL, name='中证1000')
    context = {
        'current_regime': stats.current_regime,
        'current_hv20': stats.current_hv20 * 100,
        'current_hv20_percentile': stats.current_hv20_percentile * 100,
        'confidence': stats.current_regime_confidence,
        'low_threshold': stats.low_vol_threshold * 100,
        'high_threshold': stats.high_vol_threshold * 100,
        'transition_matrix': stats.transition_matrix,
        'regime_stability': stats.regime_stability,
        'signal_friendly': stats.signal_friendly,
    }
    return context


def print_regime_context(ctx: Dict):
    """打印体制上下文"""
    emoji = {'低波动': 'G', '中等波动': 'Y', '高波动': 'R'}
    reg = ctx['current_regime']
    print(f"\n{'='*70}")
    print(f"  当前波动率体制")
    print(f"{'='*70}")
    print(f"  体制: [{emoji.get(reg, '?')}] {reg}")
    print(f"  HV20: {ctx['current_hv20']:.2f}%  (百分位: {ctx['current_hv20_percentile']:.1f}%)")
    print(f"  置信度: {ctx['confidence']:.1f}%")
    print(f"  阈值: 低<{ctx['low_threshold']:.2f}% | 中<{ctx['high_threshold']:.2f}% | 高>={ctx['high_threshold']:.2f}%")
    print(f"\n  转换概率矩阵:")
    print(f"  {'从/到':>12} {'低波动':>10} {'中等波动':>10} {'高波动':>10}")
    print(f"  {'-'*45}")
    for curr in ['低波动', '中等波动', '高波动']:
        if curr in ctx['transition_matrix']:
            row = f"  {curr:>10}"
            for nxt in ['低波动', '中等波动', '高波动']:
                row += f" {ctx['transition_matrix'][curr].get(nxt, 0):>9.1f}%"
            print(row)
    print(f"  体制稳定性(留存率均值): {ctx['regime_stability']:.1f}%")
    print(f"{'='*70}\n")


# ============================================================
# 第四章：信号报告输出
# ============================================================

def print_signal_report(df: pd.DataFrame, ctx: Dict):
    """打印2.0增强版的信号报告"""
    signals = df[df['Signal_V2'] != ''].copy()
    signals = signals.sort_index()

    print(f"\n{'='*80}")
    print(f"  V3_TEST 信号捕捉器 2.0  - 体制感知型信号报告")
    print(f"  数据范围: {df.index.min().strftime('%Y-%m-%d')} ~ {df.index.max().strftime('%Y-%m-%d')}")
    print(f"  报告生成: {TODAY_STR}")
    print(f"{'='*80}")
    print_regime_context(ctx)

    print(f"\n{'='*80}")
    print(f"  信号总览")
    print(f"{'='*80}")
    print(f"  历史信号总数: {len(signals)}")
    if len(signals) == 0:
        print("  无任何信号产生")
        return

    n_ls = int(signals['Long_Strong'].sum())
    n_lw = int(signals['Long_Weak'].sum())
    n_ss = int(signals['Short_Strong'].sum())
    n_sw = int(signals['Short_Weak'].sum())
    n_slv = int(signals['Short_LowVol'].sum())
    n_se = int(signals['Short_Extended'].sum())
    n_sb = int(signals['Short_Breakdown'].sum())

    print(f"\n  信号分布:")
    print(f"    做多***[深V]:          {n_ls:>4}")
    print(f"    做多*[双冰点]:          {n_lw:>4}")
    print(f"    做空***[波动加速]:      {n_sb:>4}")
    print(f"    做空***[风暴眼连续]:    {n_ss:>4}")
    print(f"    做空**[低波做空]新增:   {n_slv:>4}")
    print(f"    做空**[扩展风暴眼]新增: {n_se:>4}")
    print(f"    做空*[风暴眼首日]:      {n_sw:>4}")

    print(f"\n  最近20个信号详情:")
    print(f"{'='*100}")
    recent = signals.tail(20)
    for date, row in recent.iterrows():
        date_str = date.strftime('%Y-%m-%d')
        sig = row['Signal_V2']
        hv_p = row['HV20_perc'] * 100
        hv_v = row['HV20'] * 100
        dte = int(row['Suggested_DTE'])
        score = row['Composite_Score']
        if hv_p > 66.7:
            icon = '[R]'
        elif hv_p < 33.3:
            icon = '[G]'
        else:
            icon = '[Y]'
        # 网格上下文
        zone = row.get('Grid_Zone_Label', '')
        pos = row.get('Grid_Cell_Pos', np.nan)
        fib = row.get('Grid_Nearest_Fib', '')
        fib_dist = row.get('Grid_Fib_Dist_pct', np.nan)
        trig = row.get('Grid_Cell_Triggered', False)
        grid_info = ''
        if zone and not (isinstance(pos, float) and np.isnan(pos)):
            pos_pct = int(pos * 100)
            fd = f"{fib_dist:+.1f}%" if not np.isnan(fib_dist) else ''
            t_mark = '触' if trig else '新'
            grid_info = f"  {zone}({pos_pct}%){fib}{fd}[{t_mark}]"
        # 网格加分显示
        gb = row.get('Grid_Score_Bonus', 0.0)
        bonus_str = f' 网格{gb:+.1f}' if gb != 0.0 else ''
        print(f"  {date_str} {icon} HV20:{hv_v:.1f}%({hv_p:.0f}%) {sig:<42s} DTE:{dte}天 {score:>+5.1f}{grid_info}{bonus_str}")

    # 今日信号
    print(f"\n{'='*80}")
    print(f"  今日信号检测: {TODAY_STR}")
    print(f"{'='*80}")
    if TODAY_STR in signals.index:
        tr = signals.loc[TODAY_STR]
        print(f"  [今日信号触发]")
        print(f"  信号: {tr['Signal_V2']}")
        print(f"  收盘价: {tr['close']:.2f}")
        print(f"  HV20: {tr['HV20']*100:.2f}% (百分位: {tr['HV20_perc']*100:.1f}%)")
        print(f"  综合评分: {tr['Composite_Score']:+.2f}")
        print(f"  推荐DTE: {int(tr['Suggested_DTE'])}天")
        if not pd.isna(tr.get('Grid_Cell_Pos', np.nan)):
            zone = tr['Grid_Zone_Label']
            pos = int(tr['Grid_Cell_Pos'] * 100)
            fib = tr['Grid_Nearest_Fib']
            fd = tr['Grid_Fib_Dist_pct']
            trig = '已触发' if tr['Grid_Cell_Triggered'] else '未触发'
            gb = tr.get('Grid_Score_Bonus', 0.0)
            gb_str = f' | 网格加分: {gb:+.1f}' if gb != 0.0 else ''
            print(f"  网格位置: {zone}({pos}%) | 最近Fib: {fib}({fd:+.1f}%) | 格子状态: {trig}{gb_str}")
        if '做多' in str(tr['Signal_V2']):
            print(f"  推荐结构: 牛市价差 (100%-105%)")
        elif '做空' in str(tr['Signal_V2']):
            print(f"  推荐结构: 熊市价差 (95%-100%)")
    else:
        print(f"  今日无信号触发")

    print(f"\n{'='*80}\n")


# ============================================================
# 第五章：可视化
# ============================================================

def plot_signals_v2(df: pd.DataFrame, start_date: str = '2021-01-01'):
    """2.0增强版可视化
    
    Parameters
    ----------
    df : DataFrame
        包含信号的数据
    start_date : str
        图表起始日期，默认'2021-01-01'
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(22, 12),
                                    gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f'中证1000 信号捕捉器 2.0 (体制感知型) - {start_date} 至 {TODAY_STR}',
                 fontsize=16, fontweight='bold')

    cutoff = pd.Timestamp(start_date)
    df_plot = df[df.index >= cutoff].copy()
    if len(df_plot) == 0:
        df_plot = df

    ax1.plot(df_plot.index, df_plot['close'], color='black', linewidth=1, alpha=0.7, label='中证1000')
    ax1.plot(df_plot.index, df_plot['MA120'], color='blue', linestyle='--', alpha=0.4, label='MA120')
    ax1.plot(df_plot.index, df_plot['MA180'], color='orange', linestyle='--', alpha=0.3, label='MA180')
    ax1.plot(df_plot.index, df_plot['MA60'], color='gray', linestyle=':', alpha=0.4, label='MA60')

    for i in range(len(df_plot) - 1):
        idx = df_plot.index[i]
        hp = df_plot.loc[idx, 'HV20_perc']
        if hp > 0.67:
            c = '#FFE0E0'
        elif hp < 0.33:
            c = '#E0FFE0'
        else:
            c = '#FFFFE0'
        ax1.axvspan(idx, df_plot.index[i+1], alpha=0.15, color=c)

    sp = df_plot[df_plot['Signal_V2'] != '']
    signal_configs = [
        ('Long_Strong', 'red', '^', 200, '做多*** 深V'),
        ('Long_Weak', 'green', '^', 80, '做多* 双冰点'),
        ('HV20_Breakdown', 'darkred', 'v', 200, '做空*** 波动加速'),
        ('Short_Strong', 'red', 'v', 150, '做空*** 风暴眼连续'),
        ('Short_LowVol', 'purple', 'v', 120, '做空** 低波做空'),
        ('Short_Extended', 'orange', 'v', 100, '做空** 扩展风暴眼'),
        ('Short_Weak', 'gold', 'v', 70, '做空* 风暴眼首日'),
    ]
    for label, color, marker, size, legend_label in signal_configs:
        subset = sp[sp[label] == True]
        if len(subset) > 0:
            ax1.scatter(subset.index, subset['close'], marker=marker,
                       color=color, s=size, edgecolor='black', zorder=10,
                       label=legend_label)

    # Simplified legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0],[0], marker='^', color='w', markerfacecolor='red', markersize=15, label='做多*** 深V'),
        Line2D([0],[0], marker='^', color='w', markerfacecolor='green', markersize=12, label='做多* 双冰点'),
        Line2D([0],[0], marker='v', color='w', markerfacecolor='darkred', markersize=15, label='做空*** 波动加速'),
        Line2D([0],[0], marker='v', color='w', markerfacecolor='red', markersize=12, label='做空*** 风暴眼连续'),
        Line2D([0],[0], marker='v', color='w', markerfacecolor='purple', markersize=12, label='做空** 低波做空'),
        Line2D([0],[0], marker='v', color='w', markerfacecolor='orange', markersize=10, label='做空** 扩展风暴眼'),
        Line2D([0],[0], marker='v', color='w', markerfacecolor='gold', markersize=8, label='做空* 风暴眼首日'),
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=8, ncol=2)
    ax1.set_ylabel('指数点位')
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(df_plot.index, df_plot['HV20_perc']*100, 0,
                     where=(df_plot['HV20_perc']*100 >= 66.7), color='red', alpha=0.2, label='高波动区')
    ax2.fill_between(df_plot.index, df_plot['HV20_perc']*100, 0,
                     where=(df_plot['HV20_perc']*100 <= 33.3), color='green', alpha=0.2, label='低波动区')
    ax2.plot(df_plot.index, df_plot['HV20_perc']*100, color='blue', linewidth=1.5, label='HV20百分位(%)')
    ax2.axhline(66.7, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(33.3, color='green', linestyle='--', alpha=0.5)

    score_colors = df_plot['Composite_Score'].apply(lambda s: 'red' if s > 0 else ('green' if s < 0 else 'gray'))
    ax2.bar(df_plot.index, df_plot['Composite_Score'], color=score_colors, alpha=0.5, width=1, label='综合评分')
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_ylabel('百分位 / 评分')
    ax2.set_xlabel('日期')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ============================================================
# 第六章：导出信号数据
# ============================================================

def export_signals_csv(df: pd.DataFrame, filepath: str = None):
    """导出信号数据到CSV"""
    signals = df[df['Signal_V2'] != ''].copy()
    if len(signals) == 0:
        print("  无信号可导出")
        return

    export_cols = ['close', 'Signal_V2', 'HV20', 'HV20_perc',
                   'Suggested_DTE', 'Composite_Score', 'Direction', 'Confidence',
                   'Long_Strong', 'Long_Weak', 'Short_Strong', 'Short_Weak',
                   'Short_LowVol', 'Short_Extended', 'HV20_Breakdown']
    export_cols = [c for c in export_cols if c in signals.columns]

    if filepath is None:
        filepath = f'V3_TEST_signals_v2_{TODAY_STR}.csv'

    signals[export_cols].to_csv(filepath, encoding='utf-8-sig')
    print(f"  已导出 {len(signals)} 条信号至: {filepath}")


# ============================================================
# 第七章：主函数入口
# ============================================================

def main():
    """主流程"""
    print(f"\n{'#'*80}")
    print(f"  V3_TEST 信号捕捉器 2.0")
    print(f"  运行时间: {NOW.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*80}")

    df = fetch_data()
    print("\n[2/6] 计算因子和信号...")
    df = compute_all_signals(df)
    print(f"  计算完成，共 {len(df)} 个交易日")

    print("\n[2+/6] 添加江恩网格上下文...")
    df = add_grid_context(df)

    print("\n[2++/6] 应用网格感知评分加权...")
    df = apply_grid_score_adjustments(df)

    print("\n[3/6] 运行波动率体制分析...")
    ctx = get_regime_context(df)
    print(f"  当前体制: {ctx['current_regime']}, HV20: {ctx['current_hv20']:.2f}%, 百分位: {ctx['current_hv20_percentile']:.1f}%")

    print("\n[4/6] 生成信号报告...")
    print_signal_report(df, ctx)

    print("\n[5/6] 导出信号数据...")
    export_signals_csv(df)

    print("\n[6/6] 生成可视化图表...")
    try:
        plot_signals_v2(df)
        print("  图表已生成")
    except Exception as e:
        print(f"  图表生成失败: {e}")

    print(f"\n{'#'*80}")
    print(f"  分析完成")
    print(f"{'#'*80}")
    return df, ctx


# ============================================================
# 可复用导出接口
# ============================================================

def get_signals_v2(start='2018-01-01'):
    """完整流程：取数据 -> 算信号 -> 返回"""
    df = fetch_data(start)
    df = compute_all_signals(df)
    ctx = get_regime_context(df)
    signals = df[df['Signal_V2'] != ''][
        ['close', 'Signal_V2', 'HV20', 'HV20_perc',
         'Suggested_DTE', 'Composite_Score', 'Direction', 'Confidence',
         'Long_Strong', 'Long_Weak', 'Short_Weak', 'Short_Strong',
         'Short_LowVol', 'Short_Extended', 'HV20_Breakdown']
    ].copy()
    return df, signals, ctx


if __name__ == '__main__':
    df, ctx = main()
