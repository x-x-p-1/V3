"""
==========================================================================
 黄金方格子_含信号.py — 黄金方格子 × V3_TEST信号捕捉器 动态整合
==========================================================================
 【Phase 3 · 全视觉集成】
 
 将V3_TEST_信号捕捉器2_0.py的7种信号动态绘制在黄金方格子（江恩网格+斐波那契）图表上。
 包含：
   - 7种信号类型（深V/双冰/风暴眼/低波做空/波动加速/扩展风暴眼）
   - 江恩网格 + 斐波那契层级
   - 已触发格子着色（透明度与触发次数成正比）
   - 波动率体制背景（红=高波动/黄=中/绿=低）
   - 网格位置标签（格顶/格中/格底）标注在信号点上
   - Grid_Score_Bonus 可视化

 依赖:
   - V3_TEST_信号捕捉器2_0.py  → fetch_data(), compute_all_signals(), add_grid_context()
   - golden_grid.py             → build_gann_grid_cells(), grid_date_to_x()
   - config.py                   → GRID_ANCHORS, FIB_RATIOS, GANN_MULTIPLIERS

 数据来源:
   - 日线数据: 掘金量化 API
   - 网格: golden_grid.py 基于 config 锚点构建

 用法:
   python 黄金方格子_含信号.py
==========================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MatplotlibPolygon
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from shapely.geometry import LineString, Polygon as ShapelyPolygon, Point
from shapely.ops import polygonize, unary_union
from datetime import datetime
import sys
import warnings
from collections import Counter

warnings.filterwarnings('ignore')
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 导入自建模块 ==========
from golden_grid import build_gann_grid_cells, grid_date_to_x, grid_x_to_date
from config import GRID_ANCHORS, GRID_A_DATE, FIB_RATIOS, GANN_MULTIPLIERS

# ========== 从V3_TEST导入核心函数 ==========
from V3_TEST_信号捕捉器2_0 import (
    fetch_data,
    compute_all_signals,
    add_grid_context,
    apply_grid_score_adjustments,
    get_regime_context,
    RegimeEngine,
)


# ============================================================
# 1. 完整流水线：获取数据 → 全部信号 → 网格上下文 → 网格评分加权
# ============================================================
print("=" * 70)
print("  黄金方格子 × V3_TEST 信号捕捉器 2.0 — 全视觉集成")
print("=" * 70)

START_DATE = '2018-01-01'

print(f"\n[1/5] 从掘金量化获取数据并计算全部信号 (7种类型)...")
try:
    df = fetch_data(start=START_DATE)
    df = compute_all_signals(df)
except Exception as e:
    print(f"  数据获取或信号计算失败: {e}")
    sys.exit(1)

print(f"  日线数据: {len(df)} 条 ({df.index.min().date()} ~ {df.index.max().date()})")
signals_mask = df['Signal_V2'] != ''
n_signals = signals_mask.sum()
print(f"  信号总数: {n_signals}")
if n_signals > 0:
    print(f"    做多★★★[深V]:         {df['Long_Strong'].sum():>4.0f}")
    print(f"    做多★[双冰]:           {df['Long_Weak'].sum():>4.0f}")
    print(f"    做空★★★[波动加速]:     {df['Short_Breakdown'].sum():>4.0f}")
    print(f"    做空★★★[风暴眼·连续]:  {df['Short_Strong'].sum():>4.0f}")
    print(f"    做空★★[低波做空]:      {df['Short_LowVol'].sum():>4.0f}")
    print(f"    做空★★[扩展风暴眼]:    {df['Short_Extended'].sum():>4.0f}")
    print(f"    做空★[风暴眼·首日]:    {df['Short_Weak'].sum():>4.0f}")

print(f"\n[2/5] 添加江恩网格上下文 (Phase 1)...")
df = add_grid_context(df)

print(f"\n[3/5] 应用网格感知评分加权 (Phase 2)...")
df = apply_grid_score_adjustments(df)

print(f"\n[4/5] 运行波动率体制分析...")
ctx = get_regime_context(df)
print(f"  当前体制: {ctx['current_regime']}, HV20百分位: {ctx['current_hv20_percentile']:.1f}%")


# ============================================================
# 5. 聚合周线+计算X坐标
# ============================================================
print(f"\n[5/5] 构建黄金方格子 & 渲染...")

# 聚合周线（用于K线渲染，减少密集度）
df_weekly = df.resample('W-FRI').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

# 计算X坐标（从锚点A开始的周数）
df['x'] = (df.index - GRID_A_DATE).days / 7.0
df_weekly['x'] = (df_weekly.index - GRID_A_DATE).days / 7.0

# 信号数据子集（方便后续循环）
signals_df = df[signals_mask].copy()
signals_df['x'] = (signals_df.index - GRID_A_DATE).days / 7.0


# ============================================================
# 6. 构建黄金方格子（江恩网格）
# ============================================================
print("  构建江恩网格...")
cells = build_gann_grid_cells()
print(f"  网格格子数: {len(cells)}")


# ============================================================
# 7. 网格触发检测（含触发次数统计）
# ============================================================
print("  检测K线-格子碰撞 (含触发次数统计)...")
triggered_polygons = set()
cell_trigger_count = Counter()  # 统计每个格子被触发次数

for i, cell in enumerate(cells):
    poly = cell["polygon"]
    min_x, min_y, max_x, max_y = poly.bounds
    trigger_count = 0
    for _, row in df.iterrows():
        kx = row['x']
        klow, khigh = row['low'], row['high']
        if min_x <= kx <= max_x:
            kline = LineString([(kx, klow), (kx, khigh)])
            if poly.intersects(kline):
                triggered_polygons.add(i)
                trigger_count += 1
    if trigger_count > 0:
        cell_trigger_count[i] = trigger_count

max_trigger = max(cell_trigger_count.values()) if cell_trigger_count else 1
print(f"  已触发的格子: {len(triggered_polygons)} / {len(cells)}")
print(f"  最高触发次数: {max_trigger}")


# ============================================================
# 8. 渲染图表
# ============================================================
print("  渲染图表...")
fig, ax = plt.subplots(figsize=(44, 15))
fig.canvas.manager.set_window_title('黄金方格子 × V3_TEST 信号捕捉器 2.0 — 全视觉集成')

# 视野范围
view_min_x = df['x'].min() - 5
view_max_x = df['x'].max() + 15
view_min_y = df['low'].min() - 500
view_max_y = df['high'].max() + 800

view_box = ShapelyPolygon([(view_min_x, view_min_y), (view_max_x, view_min_y),
                           (view_max_x, view_max_y), (view_min_x, view_max_y)])

# ---- 8a. 波动率体制背景 (新增) ----
print("  绘制波动率体制背景...")
regime_colors = {
    '高波动': '#FFE0E0',
    '中等波动': '#FFFFE0',
    '低波动': '#E0FFE0',
}
regime_alpha = 0.12

# 每隔20个交易日绘制一个体制色带
step = 20
for i in range(0, len(df) - 1, step):
    idx_start = df.index[i]
    idx_end = df.index[min(i + step, len(df) - 1)]
    x_start = (idx_start - GRID_A_DATE).days / 7.0
    x_end = (idx_end - GRID_A_DATE).days / 7.0
    hp = df.loc[idx_start, 'HV20_perc']
    if hp > 0.67:
        color = regime_colors['高波动']
    elif hp < 0.33:
        color = regime_colors['低波动']
    else:
        color = regime_colors['中等波动']
    ax.axvspan(x_start, x_end, alpha=regime_alpha, color=color, zorder=0)

# ---- 8b. 绘制江恩线 ----
print("  绘制江恩线...")
x_A = 0
x_B = grid_date_to_x(GRID_ANCHORS['B'][0])
x_C = grid_date_to_x(GRID_ANCHORS['C'][0])
y_A, y_B, y_C = GRID_ANCHORS['A'][1], GRID_ANCHORS['B'][1], GRID_ANCHORS['C'][1]
scale_AB = abs(y_B - y_A) / abs(x_B - x_A)
scale_BC = abs(y_C - y_B) / abs(x_C - x_B)

future_x = df['x'].max() + 500
for m in GANN_MULTIPLIERS:
    line_up = LineString([(x_A, y_A), (future_x, y_A + scale_AB * m * (future_x - x_A))])
    if line_up.intersects(view_box):
        x, y = line_up.xy
        ax.plot(x, y, color='blue', alpha=0.25, linewidth=0.8, zorder=1)
    line_down = LineString([(x_B, y_B), (future_x, y_B - scale_BC * m * (future_x - x_B))])
    if line_down.intersects(view_box):
        x, y = line_down.xy
        ax.plot(x, y, color='blue', alpha=0.25, linewidth=0.8, zorder=1)

# ---- 8c. 绘制网格格子（触发次数越多的格子越不透明） ----
print("  绘制网格格子...")
for i, cell in enumerate(cells):
    min_x_b, min_y_b, max_x_b, max_y_b = cell["polygon"].bounds
    if max_x_b < view_min_x or min_x_b > view_max_x or max_y_b < view_min_y or min_y_b > view_max_y:
        continue

    x, y = cell["polygon"].exterior.xy
    if i in triggered_polygons:
        # 触发次数 → 透明度（0.10 ~ 0.40）
        n_trig = cell_trigger_count.get(i, 1)
        alpha_val = 0.10 + 0.30 * min(n_trig / max_trigger, 1.0)
        ax.add_patch(MatplotlibPolygon(list(zip(x, y)), closed=True,
                                       alpha=alpha_val, facecolor='purple',
                                       edgecolor='purple', linewidth=1.5, zorder=3))
        # 绘制Fib线
        for ratio in FIB_RATIOS:
            fib_y = min_y_b + (max_y_b - min_y_b) * ratio
            fib_line = LineString([(min_x_b - 100, fib_y), (max_x_b + 100, fib_y)])
            clipped = cell["polygon"].intersection(fib_line)
            if clipped.is_empty:
                continue
            segments = [clipped] if clipped.geom_type == 'LineString' else list(clipped.geoms)
            for seg in segments:
                cx, cy = seg.xy
                # Fib线根据比率不同颜色深浅
                if ratio in (0.382, 0.618):
                    ax.plot(cx, cy, color='darkorange', linestyle='--', linewidth=2.0, zorder=4)
                    ax.text(cx[0] + 2, cy[0] + 30, f"Fib-{ratio}", color='darkorange',
                            fontsize=8, zorder=5)
                elif ratio in (0.236, 0.764):
                    ax.plot(cx, cy, color='orange', linestyle=':', linewidth=1.2, zorder=4)
                else:  # 0.5
                    ax.plot(cx, cy, color='gold', linestyle='-.', linewidth=1.0, zorder=4)
    else:
        ax.plot(x, y, color='gray', linestyle='-', linewidth=0.3, alpha=0.3, zorder=2)

# ---- 8d. 绘制日线K线 ----
print("  绘制K线...")
up = df['close'] >= df['open']
down = df['close'] < df['open']

ax.vlines(df['x'][up], df['low'][up], df['high'][up],
          color='#ff3333', linewidth=0.6, zorder=6)
ax.vlines(df['x'][down], df['low'][down], df['high'][down],
          color='#00b300', linewidth=0.6, zorder=6)
ax.bar(df['x'][up], df['close'][up] - df['open'][up],
       bottom=df['open'][up], color='#ff3333', edgecolor='#ff3333',
       width=0.11, zorder=6)
ax.bar(df['x'][down], df['open'][down] - df['close'][down],
       bottom=df['close'][down], color='#00b300', edgecolor='#00b300',
       width=0.11, zorder=6)

# ---- 8e. 绘制7种信号（含网格上下文标注） ----
print("  绘制信号 (7种类型 + 网格位置)...")

# 信号配置：列名, 颜色, 标记, 基础大小, 标签, 垂直偏移
SIGNAL_PLOT_CONFIG = [
    ('Long_Strong',       'red',     '^', 200, '做多★★★[深V]', -200),
    ('Long_Weak',         'green',   '^',  80, '做多★[双冰]',  -150),
    ('Short_Breakdown',   'darkred', 'v', 200, '做空★★★[波动加速]', 200),
    ('Short_Strong',      '#CC0000', 'v', 150, '做空★★★[风暴眼·连续]', 180),
    ('Short_LowVol',      'purple',  'v', 120, '做空★★[低波做空]', 150),
    ('Short_Extended',    'orange',  'v', 100, '做空★★[扩展风暴眼]', 150),
    ('Short_Weak',        'gold',    'v',  70, '做空★[风暴眼·首日]', 120),
]

for col_name, color, marker, base_size, label, y_offset in SIGNAL_PLOT_CONFIG:
    subset = signals_df[signals_df[col_name] == True]
    if len(subset) == 0:
        continue

    for _, row in subset.iterrows():
        x_pos = row['x']
        y_pos = row['close']
        score_abs = abs(row['Composite_Score'])

        # 大小 = 基础大小 × 评分因子 (评分越高越大)
        if score_abs >= 4:
            size_factor = 1.3
        elif score_abs >= 2:
            size_factor = 1.0
        else:
            size_factor = 0.7
        marker_size = base_size * size_factor

        # 做多信号标在K线下方，做空标在上方
        draw_y = y_pos + y_offset * size_factor

        ax.scatter(x_pos, draw_y, marker=marker, color=color, s=marker_size,
                   edgecolor='black', linewidth=0.5, zorder=10, alpha=0.9)

        # ---- 网格位置标签 (格顶/格底 才标注) ----
        zone = row.get('Grid_Zone_Label', '')
        bonus = row.get('Grid_Score_Bonus', 0.0)
        if zone in ('格顶', '格底') and not pd.isna(row.get('Grid_Cell_Pos', np.nan)):
            pos_pct = int(row['Grid_Cell_Pos'] * 100)
            label_text = f"{zone}({pos_pct}%)"
            if bonus != 0.0:
                label_text += f" {'B' if bonus > 0 else 'P'}{bonus:+.1f}"
            # 标签颜色与格子区域匹配
            label_color = '#FF4444' if zone == '格顶' else '#44AA44'
            ax.annotate(label_text, (x_pos, draw_y),
                        xytext=(8, -8 if marker == '^' else 8),
                        textcoords='offset points', fontsize=6,
                        color=label_color, fontweight='bold',
                        alpha=0.8, zorder=11)

        # ---- 网格加分标记 (Grid_Score_Bonus != 0) ----
        if bonus != 0.0:
            bonus_color = '#00AA00' if bonus > 0 else '#AA0000'
            bonus_sign = '+' if bonus > 0 else ''
            ax.annotate(f"网格{bonus_sign}{bonus:.1f}", (x_pos, draw_y),
                        xytext=(-5, -18 if marker == '^' else 18),
                        textcoords='offset points', fontsize=5.5,
                        color=bonus_color, fontweight='bold',
                        alpha=0.9, zorder=11,
                        bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                                  edgecolor=bonus_color, alpha=0.7))

# ---- 8f. 锚点标注 ----
for label_key, (date_str, price) in GRID_ANCHORS.items():
    x_anchor = grid_date_to_x(date_str)
    ax.scatter(x_anchor, price, marker='s', color='blue', s=100, zorder=12)
    ax.annotate(f"锚点{label_key}\n{date_str}\n{price}", (x_anchor, price),
                xytext=(10, 10), textcoords='offset points', fontsize=9,
                color='blue', fontweight='bold', zorder=12,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# ---- 8g. 图例（完整7种信号 + 网格元素） ----
legend_elements = [
    # 信号图例
    Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=15, label='做多★★★ [深V反包]'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=12, label='做多★ [双冰点]'),
    Line2D([0], [0], marker='v', color='w', markerfacecolor='darkred', markersize=15, label='做空★★★ [波动加速]'),
    Line2D([0], [0], marker='v', color='w', markerfacecolor='#CC0000', markersize=13, label='做空★★★ [风暴眼·连续]'),
    Line2D([0], [0], marker='v', color='w', markerfacecolor='purple', markersize=12, label='做空★★ [低波做空]'),
    Line2D([0], [0], marker='v', color='w', markerfacecolor='orange', markersize=11, label='做空★★ [扩展风暴眼]'),
    Line2D([0], [0], marker='v', color='w', markerfacecolor='gold', markersize=9, label='做空★ [风暴眼·首日]'),
    # 网格图例
    Line2D([0], [0], color='purple', linewidth=3, label='已触发格子(深浅=触发次数)'),
    Line2D([0], [0], color='gray', linewidth=1, label='未触发格子'),
    Line2D([0], [0], color='darkorange', linestyle='--', linewidth=2, label='Fib-0.382/0.618'),
    Line2D([0], [0], color='orange', linestyle=':', linewidth=1.5, label='Fib-0.236/0.764'),
    Line2D([0], [0], color='gold', linestyle='-.', linewidth=1, label='Fib-0.5'),
    # 体制图例
    Line2D([0], [0], color='#FFE0E0', linewidth=4, label='高波动区', alpha=0.5),
    Line2D([0], [0], color='#E0FFE0', linewidth=4, label='低波动区', alpha=0.5),
    # 区域标签图例
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#FF4444', markersize=8, label='格顶(阻力位)'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#44AA44', markersize=8, label='格底(支撑位)'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=8, ncol=2,
          framealpha=0.9, edgecolor='gray')

# ---- 8h. 坐标轴格式化 ----
def format_date(x, pos=None):
    target_date = GRID_A_DATE + pd.Timedelta(days=x * 7)
    return target_date.strftime('%Y-%m-%d')

ax.xaxis.set_major_formatter(FuncFormatter(format_date))
plt.xticks(rotation=45, fontsize=8)

ax.set_xlim(view_min_x, view_max_x)
ax.set_ylim(view_min_y, view_max_y)

today_str = datetime.now().strftime('%Y-%m-%d')
ax.set_title(f"中证1000 黄金方格子 × V3_TEST 信号捕捉器 2.0（全视觉集成·数据至{today_str}）",
             fontsize=20, fontweight='bold')
ax.set_ylabel("指数点位", fontsize=12)
ax.set_xlabel("日期 (周)", fontsize=12)

# ---- 8i. 添加网格位置统计信息到图表左下角 ----
stats_text = (
    f"信号统计: 共{n_signals}条\n"
    f"  网格加分调整: {(signals_df['Grid_Score_Bonus'] != 0).sum()}条\n"
    f"  格顶信号: {(signals_df['Grid_Zone_Label'] == '格顶').sum()}条\n"
    f"  格底信号: {(signals_df['Grid_Zone_Label'] == '格底').sum()}条\n"
    f"  格中信号: {(signals_df['Grid_Zone_Label'] == '格中').sum()}条\n"
    f"当前体制: {ctx['current_regime']} (HV20%: {ctx['current_hv20_percentile']:.0f}%)"
)
ax.text(0.01, 0.01, stats_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='bottom', horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'),
        zorder=20)

plt.tight_layout()
plt.show()

print(f"\n{'='*70}")
print(f"  ✅ 完成! 共 {n_signals} 个信号动态绘制在黄金方格子上。")
print(f"  📊 网格评分调整: {(signals_df['Grid_Score_Bonus'] != 0).sum()} 条")
print(f"  📊 格顶: {(signals_df['Grid_Zone_Label'] == '格顶').sum()} 条")
print(f"  📊 格底: {(signals_df['Grid_Zone_Label'] == '格底').sum()} 条")
print(f"  📊 格中: {(signals_df['Grid_Zone_Label'] == '格中').sum()} 条")
print(f"{'='*70}")
