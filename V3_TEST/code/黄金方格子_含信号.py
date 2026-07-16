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
import plotly.graph_objects as go
import plotly.io as pio
from shapely.geometry import LineString, Polygon as ShapelyPolygon, Point
from shapely.ops import polygonize, unary_union
from datetime import datetime
import sys
import warnings
from collections import Counter

warnings.filterwarnings('ignore')
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

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
# 8. 渲染图表 (Plotly 交互式)
# ============================================================
print("  渲染图表 (Plotly 交互式)...")
fig = go.Figure()

# 视野范围
view_min_x = df['x'].min() - 5
view_max_x = df['x'].max() + 15
view_min_y = df['low'].min() - 500
view_max_y = df['high'].max() + 800

view_box = ShapelyPolygon([(view_min_x, view_min_y), (view_max_x, view_min_y),
                           (view_max_x, view_max_y), (view_min_x, view_max_y)])

# ---- 8a. 波动率体制背景 ----
print("  绘制波动率体制背景...")
regime_colors_map = {
    '高波动': 'rgba(255,224,224,0.12)',
    '中等波动': 'rgba(255,255,224,0.12)',
    '低波动': 'rgba(224,255,224,0.12)',
}

step = 20
for i in range(0, len(df) - 1, step):
    idx_start = df.index[i]
    idx_end = df.index[min(i + step, len(df) - 1)]
    x_start = (idx_start - GRID_A_DATE).days / 7.0
    x_end = (idx_end - GRID_A_DATE).days / 7.0
    hp = df.loc[idx_start, 'HV20_perc']
    if hp > 0.67:
        color = regime_colors_map['高波动']
    elif hp < 0.33:
        color = regime_colors_map['低波动']
    else:
        color = regime_colors_map['中等波动']
    fig.add_vrect(x0=x_start, x1=x_end, fillcolor=color,
                  line_width=0, layer='below')

# ---- 8b. 绘制江恩线 ----
print("  绘制江恩线...")
x_A = 0
x_B = grid_date_to_x(GRID_ANCHORS['B'][0])
x_C = grid_date_to_x(GRID_ANCHORS['C'][0])
y_A, y_B, y_C = GRID_ANCHORS['A'][1], GRID_ANCHORS['B'][1], GRID_ANCHORS['C'][1]
scale_AB = abs(y_B - y_A) / abs(x_B - x_A)
scale_BC = abs(y_C - y_B) / abs(x_C - x_B)

future_x = df['x'].max() + 500
gann_lines_x, gann_lines_y = [], []
for m in GANN_MULTIPLIERS:
    line_up = LineString([(x_A, y_A), (future_x, y_A + scale_AB * m * (future_x - x_A))])
    if line_up.intersects(view_box):
        gx, gy = line_up.xy
        gann_lines_x += list(gx) + [None]
        gann_lines_y += list(gy) + [None]
    line_down = LineString([(x_B, y_B), (future_x, y_B - scale_BC * m * (future_x - x_B))])
    if line_down.intersects(view_box):
        gx, gy = line_down.xy
        gann_lines_x += list(gx) + [None]
        gann_lines_y += list(gy) + [None]

fig.add_trace(go.Scatter(
    x=gann_lines_x, y=gann_lines_y, mode='lines',
    line=dict(color='blue', width=0.8),
    opacity=0.25, showlegend=False, hoverinfo='skip'
))

# ---- 8c. 绘制网格格子 ----
print("  绘制网格格子...")
untriggered_x, untriggered_y = [], []
for i, cell in enumerate(cells):
    min_x_b, min_y_b, max_x_b, max_y_b = cell["polygon"].bounds
    if max_x_b < view_min_x or min_x_b > view_max_x or max_y_b < view_min_y or min_y_b > view_max_y:
        continue

    px, py = cell["polygon"].exterior.xy
    x_poly = list(px) + [None]
    y_poly = list(py) + [None]

    if i in triggered_polygons:
        n_trig = cell_trigger_count.get(i, 1)
        alpha_val = 0.10 + 0.30 * min(n_trig / max_trigger, 1.0)
        fig.add_trace(go.Scatter(
            x=list(px), y=list(py),
            fill="toself",
            fillcolor=f'rgba(128,0,128,{alpha_val:.2f})',
            mode='lines',
            line=dict(color='purple', width=1.5),
            showlegend=False, hoverinfo='skip'
        ))
        for ratio in FIB_RATIOS:
            fib_y = min_y_b + (max_y_b - min_y_b) * ratio
            fib_line = LineString([(min_x_b - 100, fib_y), (max_x_b + 100, fib_y)])
            clipped = cell["polygon"].intersection(fib_line)
            if clipped.is_empty:
                continue
            segments = [clipped] if clipped.geom_type == 'LineString' else list(clipped.geoms)
            for seg in segments:
                cx, cy = seg.xy
                if ratio in (0.382, 0.618):
                    fig.add_trace(go.Scatter(
                        x=list(cx), y=list(cy), mode='lines',
                        line=dict(color='darkorange', width=2, dash='dash'),
                        showlegend=False, hoverinfo='skip'
                    ))
                    fig.add_annotation(
                        x=cx[0] + 2, y=cy[0] + 30,
                        text=f"Fib-{ratio}", font=dict(color='darkorange', size=10),
                        showarrow=False
                    )
                elif ratio in (0.236, 0.764):
                    fig.add_trace(go.Scatter(
                        x=list(cx), y=list(cy), mode='lines',
                        line=dict(color='orange', width=1.2, dash='dot'),
                        showlegend=False, hoverinfo='skip'
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=list(cx), y=list(cy), mode='lines',
                        line=dict(color='gold', width=1, dash='dashdot'),
                        showlegend=False, hoverinfo='skip'
                    ))
    else:
        untriggered_x += x_poly
        untriggered_y += y_poly

if untriggered_x:
    fig.add_trace(go.Scatter(
        x=untriggered_x, y=untriggered_y, mode='lines',
        line=dict(color='rgba(128,128,128,0.3)', width=0.3),
        showlegend=False, hoverinfo='skip'
    ))

# ---- 8d. 绘制日线K线 ----
print("  绘制K线...")
up = df['close'] >= df['open']
down = df['close'] < df['open']

for mask, c in [(up, '#ff3333'), (down, '#00b300')]:
    sub = df[mask]
    hl_x, hl_y = [], []
    body_x, body_y = [], []
    for _, r in sub.iterrows():
        hl_x += [r['x'], r['x'], None]
        hl_y += [r['low'], r['high'], None]
        body_x += [r['x'], r['x'], None]
        body_y += [r['open'], r['close'], None]
    fig.add_trace(go.Scatter(
        x=hl_x, y=hl_y, mode='lines',
        line=dict(color=c, width=0.5),
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=body_x, y=body_y, mode='lines',
        line=dict(color=c, width=3),
        showlegend=False, hoverinfo='skip'
    ))

# ---- 8e. 绘制7种信号 ----
print("  绘制信号 (7种类型 + 网格位置)...")

SIGNAL_PLOT_CONFIG = [
    ('Long_Strong',       'red',     'triangle-up',   10, '做多★★★[深V]', -200),
    ('Long_Weak',         'green',   'triangle-up',    8, '做多★[双冰]',  -150),
    ('Short_Breakdown',   'darkred', 'triangle-down', 10, '做空★★★[波动加速]', 200),
    ('Short_Strong',      '#CC0000', 'triangle-down',  9, '做空★★★[风暴眼·连续]', 180),
    ('Short_LowVol',      'purple',  'triangle-down',  8, '做空★★[低波做空]', 150),
    ('Short_Extended',    'orange',  'triangle-down',  8, '做空★★[扩展风暴眼]', 150),
    ('Short_Weak',        'gold',    'triangle-down',  7, '做空★[风暴眼·首日]', 120),
]

for col_name, color, marker, base_size, label, y_offset in SIGNAL_PLOT_CONFIG:
    subset = signals_df[signals_df[col_name] == True]
    if len(subset) == 0:
        continue

    sx, sy, sizes = [], [], []
    label_x, label_y, label_texts, label_colors = [], [], [], []
    bonus_x, bonus_y, bonus_texts, bonus_colors = [], [], [], []

    for _, row in subset.iterrows():
        x_pos = row['x']
        y_pos = row['close']
        score_abs = abs(row['Composite_Score'])

        if score_abs >= 4:
            size_factor = 1.3
        elif score_abs >= 2:
            size_factor = 1.0
        else:
            size_factor = 0.7
        marker_size = base_size * size_factor

        draw_y = y_pos + y_offset * size_factor
        sx.append(x_pos)
        sy.append(draw_y)
        sizes.append(marker_size * 1.5)

        zone = row.get('Grid_Zone_Label', '')
        bonus = row.get('Grid_Score_Bonus', 0.0)

        if zone in ('格顶', '格底') and not pd.isna(row.get('Grid_Cell_Pos', np.nan)):
            pos_pct = int(row['Grid_Cell_Pos'] * 100)
            lt = f"{zone}({pos_pct}%)"
            if bonus != 0.0:
                lt += f" {'B' if bonus > 0 else 'P'}{bonus:+.1f}"
            lc = '#FF4444' if zone == '格顶' else '#44AA44'
            label_x.append(x_pos)
            label_y.append(draw_y)
            label_texts.append(lt)
            label_colors.append(lc)

        if bonus != 0.0:
            bonus_color = '#00AA00' if bonus > 0 else '#AA0000'
            bonus_sign = '+' if bonus > 0 else ''
            bonus_x.append(x_pos)
            bonus_y.append(draw_y)
            bonus_texts.append(f"网格{bonus_sign}{bonus:.1f}")
            bonus_colors.append(bonus_color)

    fig.add_trace(go.Scatter(
        x=sx, y=sy, mode='markers',
        marker=dict(
            symbol=marker, size=sizes, color=color,
            line=dict(color='black', width=0.5)
        ),
        opacity=0.9,
        name=label,
        hoverinfo='name'
    ))

    if label_x:
        fig.add_trace(go.Scatter(
            x=label_x, y=label_y, mode='text',
            text=label_texts,
            textfont=dict(color=label_colors, size=7, family='Arial Black'),
            textposition='top center',
            showlegend=False, hoverinfo='skip'
        ))

    if bonus_x:
        fig.add_trace(go.Scatter(
            x=bonus_x, y=bonus_y, mode='text',
            text=bonus_texts,
            textfont=dict(color=bonus_colors, size=6),
            textposition='bottom center',
            showlegend=False, hoverinfo='skip'
        ))

# ---- 8f. 锚点标注 ----
anchor_x, anchor_y, anchor_texts = [], [], []
for label_key, (date_str, price) in GRID_ANCHORS.items():
    x_anchor = grid_date_to_x(date_str)
    anchor_x.append(x_anchor)
    anchor_y.append(price)
    anchor_texts.append(f"锚点{label_key}<br>{date_str}<br>{price}")

fig.add_trace(go.Scatter(
    x=anchor_x, y=anchor_y, mode='markers+text',
    marker=dict(symbol='square', color='blue', size=10),
    text=anchor_texts,
    textposition='top center',
    textfont=dict(color='blue', size=10, family='Arial Black'),
    name='锚点',
    hoverinfo='text'
))

# ---- 8g. 坐标轴 & 布局 ----
year_step = 52
tick_x = np.arange(0, view_max_x, year_step)
tick_text = []
for tx in tick_x:
    d = GRID_A_DATE + pd.Timedelta(days=tx * 7)
    tick_text.append(d.strftime('%Y-%m-%d'))

today_str = datetime.now().strftime('%Y-%m-%d')
fig.update_layout(
    title=dict(
        text=f"中证1000 黄金方格子 × V3_TEST 信号捕捉器 2.0（全视觉集成·数据至{today_str}）",
        font=dict(size=20)
    ),
    xaxis=dict(
        range=[view_min_x, view_max_x],
        tickvals=tick_x,
        ticktext=tick_text,
        tickangle=45,
        title=dict(text="日期 (周)", font=dict(size=12))
    ),
    yaxis=dict(
        range=[view_min_y, view_max_y],
        title=dict(text="指数点位", font=dict(size=12))
    ),
    height=700,
    hovermode='closest',
    showlegend=True,
    legend=dict(
        x=0, y=1, font=dict(size=9),
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='gray',
        borderwidth=1
    ),
    margin=dict(l=50, r=30, t=50, b=50)
)

# ---- 8h. 统计信息 ----
stats_text = (
    f"信号统计: 共{n_signals}条<br>"
    f"网格加分调整: {(signals_df['Grid_Score_Bonus'] != 0).sum()}条<br>"
    f"格顶信号: {(signals_df['Grid_Zone_Label'] == '格顶').sum()}条<br>"
    f"格底信号: {(signals_df['Grid_Zone_Label'] == '格底').sum()}条<br>"
    f"格中信号: {(signals_df['Grid_Zone_Label'] == '格中').sum()}条<br>"
    f"当前体制: {ctx['current_regime']} (HV20%: {ctx['current_hv20_percentile']:.0f}%)"
)
fig.add_annotation(
    x=0.01, y=0.01, xref='paper', yref='paper',
    text=stats_text, showarrow=False,
    align='left', font=dict(size=10),
    bgcolor='rgba(255,255,255,0.8)',
    bordercolor='gray', borderwidth=1
)

# ---- 自动保存HTML ----
output_path = '黄金方格子_含信号.html'
pio.write_html(fig, file=output_path, auto_open=True)
print(f"  ✅ 图表已保存至: {output_path}")

print(f"\n{'='*70}")
print(f"  ✅ 完成! 共 {n_signals} 个信号动态绘制在黄金方格子上。")
print(f"  📊 网格评分调整: {(signals_df['Grid_Score_Bonus'] != 0).sum()} 条")
print(f"  📊 格顶: {(signals_df['Grid_Zone_Label'] == '格顶').sum()} 条")
print(f"  📊 格底: {(signals_df['Grid_Zone_Label'] == '格底').sum()} 条")
print(f"  📊 格中: {(signals_df['Grid_Zone_Label'] == '格中').sum()} 条")
print(f"{'='*70}")
