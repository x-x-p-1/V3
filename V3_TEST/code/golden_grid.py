"""
==========================================================================
 golden_grid.py — 黄金方格子/江恩网格几何计算
==========================================================================
 纯函数模块，不依赖策略类。
 负责：
   - 日期↔X坐标转换
   - 江恩网格构建
   - 点→格查找
==========================================================================
"""

import pandas as pd
from shapely.geometry import Point, LineString, Polygon as ShapelyPolygon
from shapely.ops import polygonize, unary_union

from config import GRID_ANCHORS, GRID_A_DATE, FIB_RATIOS, GANN_MULTIPLIERS


def grid_date_to_x(date_val):
    """日期 → X坐标（从锚点A开始的周数）"""
    if isinstance(date_val, str):
        date_val = pd.to_datetime(date_val)
    if hasattr(date_val, "tz"):
        date_val = date_val.tz_localize(None)
    return (date_val - GRID_A_DATE).days / 7.0


def grid_x_to_date(x):
    """X坐标 → 日期"""
    return GRID_A_DATE + pd.Timedelta(days=x * 7)


def build_gann_grid_cells():
    """
    构建黄金方格子（江恩网格），返回每个格子的列表。
    每个格子：{polygon, min_y, max_y, fib_lines: {ratio: price}}
    """
    from datetime import datetime as dt
    x_A, y_A = 0, GRID_ANCHORS["A"][1]
    x_B = grid_date_to_x(GRID_ANCHORS["B"][0])
    x_C = grid_date_to_x(GRID_ANCHORS["C"][0])
    y_B, y_C = GRID_ANCHORS["B"][1], GRID_ANCHORS["C"][1]

    scale_AB = abs(y_B - y_A) / abs(x_B - x_A)
    scale_BC = abs(y_C - y_B) / abs(x_C - x_B)

    future_x = grid_date_to_x(dt.now()) + 500
    gann_lines = []
    for m in GANN_MULTIPLIERS:
        gann_lines.append(LineString([(x_A, y_A), (future_x, y_A + scale_AB * m * (future_x - x_A))]))
        gann_lines.append(LineString([(x_B, y_B), (future_x, y_B - scale_BC * m * (future_x - x_B))]))

    bbox_min_x, bbox_max_x = x_A - 500, future_x + 500
    bbox_min_y, bbox_max_y = -50000, 50000
    gann_lines.extend([
        LineString([(bbox_min_x, bbox_min_y), (bbox_max_x, bbox_min_y)]),
        LineString([(bbox_min_x, bbox_max_y), (bbox_max_x, bbox_max_y)]),
        LineString([(bbox_min_x, bbox_min_y), (bbox_min_x, bbox_max_y)]),
        LineString([(bbox_max_x, bbox_min_y), (bbox_max_x, bbox_max_y)])
    ])

    noded_lines = unary_union(gann_lines)
    raw_polygons = list(polygonize(noded_lines))

    cells = []
    for poly in raw_polygons:
        min_x_b, min_y_b, max_x_b, max_y_b = poly.bounds
        if (min_y_b <= bbox_min_y + 10 or max_y_b >= bbox_max_y - 10 or
            min_x_b <= bbox_min_x + 10 or max_x_b >= bbox_max_x - 10):
            continue
        fib_lines = {}
        for ratio in FIB_RATIOS:
            fib_lines[ratio] = min_y_b + (max_y_b - min_y_b) * ratio
        cells.append({
            "polygon": poly,
            "min_y": min_y_b,
            "max_y": max_y_b,
            "fib_lines": fib_lines
        })
    return cells


def find_grid_cell(x_date, price, cells):
    """给定日期X坐标和价格，找到所属的网格格子"""
    pt = Point(x_date, price)
    for cell in cells:
        if cell["polygon"].contains(pt):
            return cell
    return None
