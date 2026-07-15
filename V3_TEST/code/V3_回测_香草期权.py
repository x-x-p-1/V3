"""
====================================================================================
 V3_回测_香草期权.py - 信号驱动ATM香草期权回测
====================================================================================
【核心逻辑】
 基于V3_TEST_信号捕捉器2_0.py的全套信号，在信号触发后下一交易日
 买入对应的ATM香草期权（Call/Put），持有至达标止盈（6%）或到期。

【期权费率 - 基于2026-07-08真实OTC报价，名义本金100万/份】
  1M Call: 3.30% (33,000元)  1M Put: 3.87% (38,700元)
  2M Call: 4.29% (42,900元)  2M Put: 5.64% (56,400元)

【信号映射】
  做多信号（双冰/深V）-> 1M ATM Call
  做空信号（风暴眼系列）-> 1M ATM Put
【网格过滤】做多仅格底 / 做空仅格顶
【仓位管理】总资金20万，各方向10万，单次1张
【退出规则】止盈6%（BS估值含剩余时间价值）、到期结算
====================================================================================
"""

import pandas as pd
import numpy as np
import sys, os, warnings, argparse, json, math
from datetime import datetime, timedelta
from collections import defaultdict

warnings.filterwarnings("ignore")
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import V3_TEST_信号捕捉器2_0 as v3

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from _pricing import bs_call, bs_put

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ============================================================
# 配置参数
# ============================================================
TOTAL_CAPITAL = 200000
LONG_RATIO = 0.5
SHORT_RATIO = 0.5
MAX_LEGS = 8
MAX_PER_SIDE = 4

# 期权费率（基于2026-07-08真实OTC报价）
OPT_RATES = {
    "1M": {"call": 0.0330, "put": 0.0387},
    "2M": {"call": 0.0429, "put": 0.0564},
    "3M": {"call": 0.0480, "put": 0.0682},
}
NOMINAL = 1000000
TERM_DAYS = {"1M": 30, "2M": 60, "3M": 90}

# 信号映射（含各信号专属止盈阈值）
SIG_MAP = {
    "做多★★★[深V]":        {"dir": "做多", "term": "1M", "type": "call", "tp": 0.15},
    "做多★[双冰]":          {"dir": "做多", "term": "1M", "type": "call", "tp": 0.06},
    "做多★[双冰·方向反转]": {"dir": "做空", "term": "1M", "type": "put", "tp": 0.06},
    "做空★★★[风暴眼·连续]": {"dir": "做空", "term": "1M", "type": "put", "tp": 0.06},
    "做空★★[低波做空]":    {"dir": "做空", "term": "1M", "type": "put", "tp": 0.06},
    "做空★★[扩展风暴眼]":  {"dir": "做空", "term": "1M", "type": "put", "tp": 0.06},
    "做空★[风暴眼·首日]":  {"dir": "做空", "term": "1M", "type": "put", "tp": 0.06},
}

TP_THRESHOLD = 0.06       # 默认止盈阈值（各信号可在SIG_MAP中覆盖）
MAX_HOLD = 40

# ============================================================
# 期权持仓类
# ============================================================
class SignalOption:
    _next_id = 1
    def __init__(self, sig, direc, opt_type, term, prem, tdays,
                 od, oi, pid, gz="", gp=0.5, hv=0.5, cs=0, tp=0.06):
        self.id = SignalOption._next_id
        SignalOption._next_id += 1
        self.signal_label = sig
        self.direction = direc
        self.option_type = opt_type
        self.term = term
        self.premium = prem
        self.term_days = tdays
        self.open_date = od
        self.open_index = oi
        self.K = oi
        self.expire_date = od + timedelta(days=tdays)
        self.pair_id = pid
        self.grid_zone = gz
        self.grid_pos = gp
        self.hv20_perc = hv
        self.composite_score = cs
        self.tp_threshold = tp          # 专属止盈阈值
        self.closed = False
        self.close_date = None
        self.close_reason = ""
        self.profit_loss = 0.0
        self.close_index = None
        self.gross_value = 0.0
        self.close_method = ""
        self._cached_vals = None
        self._cache_start = None

    def intrinsic(self, idx):
        if self.option_type == "call":
            pts = max(0, idx - self.K)
        else:
            pts = max(0, self.K - idx)
        return pts * (NOMINAL / self.K)

    def bs_val(self, idx, T_rem, sig, r=0.014, q=0.025):
        if T_rem <= 0: return self.intrinsic(idx)
        if self.option_type == "call":
            p = bs_call(idx, self.K, T_rem, r, sig, q)
        else:
            p = bs_put(idx, self.K, T_rem, r, sig, q)
        return p * (NOMINAL / self.K)

    def get_tv(self, idx, dt, cur_idx=None):
        if self._cached_vals and cur_idx is not None:
            off = cur_idx - self._cache_start
            if 0 <= off < len(self._cached_vals):
                return self._cached_vals[off]
        return self.intrinsic(idx)

    def precache(self, start_idx, dates, idxs, sigmas, r=0.014, q=0.025):
        self._cache_start = start_idx
        max_fwd = min(len(dates) - start_idx, self.term_days + 10)
        vals = []
        for off in range(1, max_fwd + 1):
            pos = start_idx + off
            if pos >= len(dates): break
            S = idxs[pos]
            rem = max(0, (self.expire_date - dates[pos]).days) / 365.0
            v = self.bs_val(S, rem, sigmas[pos], r, q)
            vals.append(v)
        self._cached_vals = vals

    def check_exit(self, idx, dt, high=None, low=None):
        if self.closed: return False, ""
        tp = self.tp_threshold
        if self.option_type == "call" and high is not None:
            if high / self.open_index - 1.0 >= tp:
                return True, "止盈: 盘中最高涨幅%.1f%%>=%.0f%%" % ((high/self.open_index-1.0)*100, tp*100)
        elif self.option_type == "put" and low is not None:
            if low / self.open_index - 1.0 <= -tp:
                return True, "止盈: 盘中最低跌幅%.1f%%>=%.0f%%" % (abs(low/self.open_index-1.0)*100, tp*100)
        if dt >= self.expire_date:
            return True, "到期平仓 价值%.0f元" % self.intrinsic(idx)
        return False, ""

    def close(self, cd, idx, reason="", trig=None, cur_idx=None):
        if "止盈" in reason:
            self.gross_value = self.get_tv(idx, cd, cur_idx)
            self.close_index = trig if trig else idx
            self.close_method = "止盈"
        else:
            self.gross_value = self.intrinsic(idx)
            self.close_index = idx
            self.close_method = "到期" if "到期" in reason else "强制"
        self.profit_loss = self.gross_value - self.premium
        self.close_date = cd
        self.close_reason = reason
        self.closed = True
        return self.gross_value
# ============================================================
# 回测主函数
# ============================================================
def run_backtest(enable_grid=False, start_capital=200000,
                 quiet=False, save_nav=False, start_trade_date=None):
    """信号驱动香草期权回测"""
    sname = "V3信号香草期权"
    if enable_grid: sname += " + 网格过滤"

    print("=" * 65)
    print("  %s" % sname)
    print("  本金%d万 - 1M ATM香草 - 6%%止盈(深V15%%)" % (start_capital//10000))
    print("  做多: 1M Call(3.30%%), 做空: 1M Put(3.87%%)")
    if enable_grid: print("  网格过滤: 做多仅格底 / 做空仅格顶")
    else: print("  网格过滤: 关闭")
    print("=" * 65)

    # ---- 1. 获取信号数据 ----
    print("[1/5] 获取数据并运行V3信号捕捉器...")
    df, ctx = v3.main()
    total_sigs = (df["Signal_V2"] != "").sum()
    print("  数据: %d个交易日, 信号: %d" % (len(df), total_sigs))

    # ---- 2. 准备波动率 ----
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df["hv20"] = df["log_ret"].rolling(20).std() * np.sqrt(252)
    df["hv60"] = df["log_ret"].rolling(60).std() * np.sqrt(252)
    df["sigma"] = df["hv20"].fillna(df["hv60"]).fillna(0.25)
    sigmas = df["sigma"].tolist()
    idxs = df["close"].tolist()
    highs = df["high"].tolist()
    lows = df["low"].tolist()

    # ---- 3. 逐日回测 ----
    print("[2/5] 逐日回测...")

    cash = float(start_capital)
    legs = []
    closed = []
    nav = []
    prem_paid = 0.0
    gross_recv = 0.0
    realized_pnl = 0.0
    sig_stats = defaultdict(lambda: {"trig":0,"open":0,"tp":0,"exp":0,"win":0,"loss":0,"pnl":0.0})
    pending = []
    last_open = {}

    dates = df.index.tolist()

    # 交易起始日
    tstart = 0
    first_date = dates[0]
    if start_trade_date:
        sd = pd.Timestamp(start_trade_date)
        for j, d in enumerate(dates):
            if d >= sd:
                tstart = j
                break
        first_date = dates[tstart]
        print("  交易起始: %s (idx:%d/%d)" % (start_trade_date, tstart, len(dates)))

    for i, date in enumerate(dates):
        if i < tstart:
            nav.append({"date":date,"equity":cash,"equity_tv":cash,
                       "cash":cash,"realized":cash,"active":0,"index":idxs[i]})
            continue

        row = df.loc[date]
        idx = idxs[i]
        hi = highs[i]
        lo = lows[i]
        ds = date.strftime("%Y-%m-%d")

        # 检查前一日信号
        if i > 0:
            prev = df.loc[dates[i-1]]
            sig_str = str(prev.get("Signal_V2",""))
            if sig_str and sig_str != "":
                for part in [s.strip() for s in sig_str.split("|") if s.strip()]:
                    if part not in SIG_MAP: continue
                    key = "%s_%s" % (dates[i-1].strftime("%Y%m%d"), part)
                    if key in last_open: continue
                    info = SIG_MAP[part]
                    gz = str(prev.get("Grid_Zone_Label",""))
                    gp = prev.get("Grid_Cell_Pos",0.5)

                    if enable_grid:
                        if info["dir"] == "做多" and gz != "格底": continue
                        if info["dir"] == "做空" and gz != "格顶": continue

                    pending.append({
                        "sig":part, "dir":info["dir"],
                        "type":info["type"], "term":info["term"],
                        "od":dates[i-1],"oi":prev["close"],
                        "gz":gz,"gp":gp,
                        "hv":prev.get("HV20_perc",0.5),
                        "cs":prev.get("Composite_Score",0),
                        "tp":info.get("tp", TP_THRESHOLD),
                        "key":key,
                    })
                    sig_stats[part]["trig"] += 1

        # 执行信号(T+1)
        exec_keys = set()
        for ps in pending:
            if ps["key"] in exec_keys: continue
            if ps["key"] in last_open:
                exec_keys.add(ps["key"])
                continue
            rate = OPT_RATES[ps["term"]][ps["type"]]
            prem = NOMINAL * rate
            tdays = TERM_DAYS[ps["term"]]

            # 仓位检查
            ncall = len([l for l in legs if not l.closed and l.option_type=="call"])
            nput = len([l for l in legs if not l.closed and l.option_type=="put"])
            if ps["dir"] == "做多" and ncall >= MAX_PER_SIDE: continue
            if ps["dir"] == "做空" and nput >= MAX_PER_SIDE: continue
            if ncall + nput >= MAX_LEGS: continue
            if cash < prem * 1.1: continue

            leg = SignalOption(ps["sig"],ps["dir"],ps["type"],ps["term"],
                               prem,tdays,date,idx,hash(ps["key"])%100000,
                               ps["gz"],ps["gp"],ps["hv"],ps["cs"],
                               tp=ps["tp"])
            leg.precache(i, dates, idxs, sigmas)
            cash -= prem
            prem_paid += prem
            legs.append(leg)
            last_open[ps["key"]] = date
            sig_stats[ps["sig"]]["open"] += 1
            tc = "Call" if ps["type"]=="call" else "Put"
            gs = " [%s]" % ps["gz"] if ps["gz"] else ""
            print("  > 开仓 %s %s%s -> %s %.0f元 指数:%.0f" % (ds, ps["sig"], gs, tc, prem, idx))
            exec_keys.add(ps["key"])
        pending = [p for p in pending if p["key"] not in last_open]

        # 检查退出
        to_close = []
        for leg in legs:
            if leg.closed: continue
            should, reason = leg.check_exit(idx, date, high=hi, low=lo)
            if should: to_close.append((leg, reason))

        for leg, reason in to_close:
            trig = None
            tp = leg.tp_threshold
            if "止盈" in reason:
                trig = leg.open_index * (1+tp) if leg.option_type=="call" else leg.open_index * (1-tp)
            val = leg.close(date, idx, reason, trig, i)
            cash += val
            gross_recv += val
            realized_pnl += leg.profit_loss
            ss = sig_stats[leg.signal_label]
            ss["pnl"] += leg.profit_loss
            if leg.profit_loss > 0: ss["win"] += 1
            else: ss["loss"] += 1
            if "止盈" in reason: ss["tp"] += 1
            else: ss["exp"] += 1
            tc = "Call" if leg.option_type=="call" else "Put"
            em = "+" if leg.profit_loss > 0 else "-"
            hold = (date - leg.open_date).days
            print("    %s 平仓 %s %s #%d 开:%s 持有%d天 盈亏:%.0f元" % (em, ds, leg.signal_label, leg.id, str(leg.open_date.date()), hold, leg.profit_loss))
            closed.append({
                "id":leg.id,"sig":leg.signal_label,"type":leg.option_type,
                "dir":leg.direction,"od":leg.open_date,"cd":date,
                "oi":leg.open_index,"ci":leg.close_index,
                "prem":leg.premium,"gv":val,"pnl":leg.profit_loss,
                "reason":reason,"method":leg.close_method,
                "gz":leg.grid_zone,"hv":leg.hv20_perc,
                "hold":hold,
            })

        # 每日净值
        tv = cash
        tv_tv = cash
        for leg in legs:
            if not leg.closed:
                tv += leg.intrinsic(idx)
                tv_tv += leg.get_tv(idx, date, i)
        nav.append({"date":date,"equity":tv,"equity_tv":tv_tv,
                    "cash":cash,"realized":start_capital+realized_pnl,
                    "active":len([l for l in legs if not l.closed]),"index":idx})

    # ---- 4. 结束处理 ----
    print("[3/5] 回测结束处理...")
    fd = dates[-1]
    fi = idxs[-1]
    unclosed = [l for l in legs if not l.closed]
    uncl_val = sum(l.intrinsic(fi) for l in unclosed)
    if unclosed:
        print("  截止%s仍有%d腿未到期(内在价值%.0f元)" % (str(fd.date()), len(unclosed), uncl_val))
    else:
        print("  所有合约已到期或止盈，回测完整")

    final = cash + uncl_val
    pnl = final - start_capital
    ret = pnl / start_capital * 100
    years = max((fd - first_date).days / 365.0, 0.01)
    ann_ret = ((final / start_capital) ** (1/years) - 1) * 100

    # ---- 5. 统计 ----
    print("[4/5] 生成统计报告...")
    nav_df = pd.DataFrame(nav)
    nav_df["peak"] = nav_df["equity"].cummax()
    nav_df["dd"] = nav_df["equity"] / nav_df["peak"] - 1
    mdd = nav_df["dd"].min() * 100
    nav_df["rp"] = nav_df["realized"].cummax()
    nav_df["rdd"] = nav_df["realized"] / nav_df["rp"] - 1
    mdd_r = nav_df["rdd"].min() * 100
    nav_df["tvp"] = nav_df["equity_tv"].cummax()
    nav_df["tvdd"] = nav_df["equity_tv"] / nav_df["tvp"] - 1
    mdd_tv = nav_df["tvdd"].min() * 100

    tdf = pd.DataFrame(closed)
    nav_df["dr"] = nav_df["equity"].pct_change()
    avg_r = nav_df["dr"].mean()
    std_r = nav_df["dr"].std()
    sharpe = (avg_r*252) / (std_r*252**0.5) if std_r>0 else 0
    dn = nav_df["dr"][nav_df["dr"]<0]
    dstd = dn.std()
    sortino = (avg_r*252) / (dstd*252**0.5) if dstd>0 else 0
    calmar = (ann_ret/100) / (abs(mdd)/100) if mdd<0 else 0

    # 输出报告
    print("")
    print("  " + "="*60)
    print("  回测结果汇总")
    print("  " + "="*60)
    print("  初始: %.0f万  期末: %.2f万  盈亏: %.0f元 (%.2f%%), 年化: %.2f%%" % (
        start_capital/10000, final/10000, pnl, ret, ann_ret))
    print("  夏普: %.2f  索提诺: %.2f  卡玛: %.2f" % (sharpe, sortino, calmar))
    print("  Route A(已实现)DD: %.2f%%  B(含时间价值)DD: %.2f%%  C(仅内在)DD: %.2f%%" % (mdd_r, mdd_tv, mdd))
    print("  权利金支出: %.2f万  回收: %.2f万" % (prem_paid/10000, gross_recv/10000))
    if unclosed: print("  未到期: %d腿 (%d元)" % (len(unclosed), uncl_val))

    # 按信号统计
    print("")
    print("  按信号类型统计:")
    fmt = "  %s %4s %4s %4s %4s %4s %4s %6s %9s"
    print(fmt % ("信号类型","触发","开仓","止盈","到期","盈利","亏损","胜率","总盈亏"))
    print("  " + "-"*80)
    tot_trig=0; tot_open=0; tot_w=0; tot_l=0; tot_p=0.0
    for sn in sorted(sig_stats.keys()):
        s = sig_stats[sn]
        if s["open"]==0: continue
        wr = s["win"]/(s["win"]+s["loss"])*100 if (s["win"]+s["loss"])>0 else 0
        avg = s["pnl"]/s["open"]
        print(fmt % (sn[:20],str(s["trig"]),str(s["open"]),str(s["tp"]),str(s["exp"]),
                     str(s["win"]),str(s["loss"]),"%.1f%%"%wr,"%+.0f"%s["pnl"]))
        tot_trig+=s["trig"]; tot_open+=s["open"]; tot_w+=s["win"]; tot_l+=s["loss"]; tot_p+=s["pnl"]
    tot_wr = tot_w/(tot_w+tot_l)*100 if (tot_w+tot_l)>0 else 0
    print("  " + "-"*80)
    print(fmt % ("合计",str(tot_trig),str(tot_open),"","",str(tot_w),str(tot_l),"%.1f%%"%tot_wr,"%+.0f"%tot_p))

    # 方向统计
    if len(tdf)>0:
        ct = tdf[tdf["type"]=="call"]
        pt = tdf[tdf["type"]=="put"]
        cp = ct["pnl"].sum() if len(ct)>0 else 0
        pp = pt["pnl"].sum() if len(pt)>0 else 0
        cw = (ct["pnl"]>0).mean()*100 if len(ct)>0 else 0
        pw = (pt["pnl"]>0).mean()*100 if len(pt)>0 else 0
        print("  方向统计: Call(做多) %d笔 总%.0f元 胜率%.1f%%" % (len(ct), cp, cw) if len(ct)>0 else "  Call: 0笔")
        print("           Put(做空) %d笔 总%.0f元 胜率%.1f%%" % (len(pt), pp, pw) if len(pt)>0 else "  Put: 0笔")
        print("  平均持有: %.1f天  中位: %.0f天" % (tdf["hold"].mean(), tdf["hold"].median()))

    # 逐年
    tdf["year"] = tdf["od"].dt.year
    nav_df["year"] = nav_df["date"].dt.year
    print("  逐年统计:")
    for yr in sorted(nav_df["year"].unique()):
        yn = nav_df[nav_df["year"]==yr]
        if len(yn)==0: continue
        yr_r = (yn["equity"].iloc[-1]/yn["equity"].iloc[0]-1)*100
        yt = tdf[tdf["year"]==yr]
        yw = (yt["pnl"]>0).sum() if len(yt)>0 else 0
        print("  %d: %.2f%%  交易%d笔  胜率%.1f%%" % (yr, yr_r, len(yt), yw/len(yt)*100 if len(yt)>0 else 0))

    print("")
    print("="*65)
    print("  回测完成")
    print("="*65)

    # 图表
    if HAS_PLOTLY:
        try: _gen_charts(nav_df, tdf, df, start_capital, final, pnl, ret, ann_ret, mdd, mdd_r, mdd_tv, sharpe, sortino, calmar, tot_wr, first_date, fd, sname, enable_grid)
        except Exception as e:
            import traceback; print("图表失败: %s" % e); traceback.print_exc()

    # 保存NAV
    if save_nav:
        nd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nav_data")
        os.makedirs(nd, exist_ok=True)
        gs = "_grid" if enable_grid else "_nofilter"
        nf = os.path.join(nd, "nav_v3_vanilla%s.csv" % gs)
        ne = nav_df[["date","equity","equity_tv","cash","realized","active","index"]].copy()
        ne["date"] = ne["date"].dt.strftime("%Y-%m-%d")
        ne.to_csv(nf, index=False, encoding="utf-8-sig")
        print("NAV saved: %s" % nf)

    return {
        "total_final":final,"total_pnl":pnl,"total_return_pct":ret,
        "annual_return_pct":ann_ret,"max_drawdown_pct":mdd,
        "max_dd_realized":mdd_r,"max_dd_tv":mdd_tv,
        "total_trades":len(tdf),"sharpe":sharpe,"sortino":sortino,"calmar":calmar,
        "win_rate_pct":tot_wr,
        "call_pnl":cp if len(tdf)>0 else 0,"put_pnl":pp if len(tdf)>0 else 0,
        "trades_df":tdf if len(tdf)>0 else None,"nav_df":nav_df,
        "signal_stats":dict(sig_stats),
    }
# ============================================================
# 图表生成
# ============================================================
def _gen_charts(nav_df, tdf, df, sc, final, pnl, ret, ann,
                mdd, mdd_r, mdd_tv, sh, so, ca, wr, fd1, fd2, sname, eg):
    """生成Plotly回测图表"""
    nd = nav_df.copy()
    if hasattr(nd["date"].iloc[0], "tz"):
        nd["dt"] = nd["date"].dt.tz_localize(None)
    else:
        nd["dt"] = nd["date"]
    nd["MA20"] = nd["index"].rolling(20).mean()
    nd["MA60"] = nd["index"].rolling(60).mean()

    fig = make_subplots(rows=5, cols=1,
        subplot_titles=("中证1000 + V3信号","净值曲线","回撤","持仓","每笔盈亏"),
        vertical_spacing=0.06, row_heights=[0.28,0.20,0.12,0.18,0.22], shared_xaxes=True)

    # 子图1: 指数
    fig.add_trace(go.Scatter(x=nd["dt"], y=nd["index"],
        mode="lines", name="中证1000", line=dict(color="black",width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=nd["dt"], y=nd["MA20"],
        mode="lines", name="MA20", line=dict(color="blue",width=0.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=nd["dt"], y=nd["MA60"],
        mode="lines", name="MA60", line=dict(color="red",width=0.8,dash="dot")), row=1, col=1)

    # V3信号标记
    v3d = df[df["Signal_V2"]!=""].copy()
    for sn,clr,sym in [
        ("深V","red","triangle-up"),
        ("双冰","green","triangle-up"),
        ("风暴眼·连续","red","triangle-down"),
        ("扩展风暴眼","orange","triangle-down"),
        ("低波做空","purple","triangle-down"),
        ("风暴眼·首日","gold","triangle-down"),
    ]:
        sub = v3d[v3d["Signal_V2"].str.contains(sn,na=False)]
        if len(sub)>0:
            fig.add_trace(go.Scatter(x=sub.index, y=sub["close"],
                mode="markers", name=sn[:8],
                marker=dict(color=clr,size=7,symbol=sym,line=dict(width=1,color="black"),opacity=0.7)),
                row=1, col=1)

    # 子图2: 净值
    fig.add_hline(y=sc, line_dash="dash", line_color="gray", opacity=0.5,
                  annotation_text="初始%.0f万"%(sc/10000), row=2, col=1)
    fig.add_trace(go.Scatter(x=nd["dt"], y=nd["realized"],
        mode="lines", name="A已实现(DD:%.1f%%)"%mdd_r,
        line=dict(color="green",width=1.5,dash="dash")), row=2, col=1)
    fig.add_trace(go.Scatter(x=nd["dt"], y=nd["equity_tv"],
        mode="lines", name="B含时间价值(DD:%.1f%%)"%mdd_tv,
        line=dict(color="orange",width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=nd["dt"], y=nd["equity"],
        mode="lines", name="C总权益(DD:%.1f%%)"%mdd,
        line=dict(color="navy",width=2),
        fill="tonexty", fillcolor="rgba(0,0,128,0.05)"), row=2, col=1)

    # 子图3: 回撤
    fig.add_trace(go.Scatter(x=nd["dt"], y=nd["dd"]*100,
        mode="lines", name="回撤",
        line=dict(color="red",width=1.5),
        fill="tozeroy", fillcolor="rgba(255,0,0,0.15)"), row=3, col=1)

    # 子图4: 持仓
    fig.add_trace(go.Bar(x=nd["dt"], y=nd["active"],
        name="持仓腿数", marker_color="steelblue", opacity=0.7), row=4, col=1)

    # 子图5: 盈亏
    fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.4, row=5, col=1)
    if len(tdf)>0:
        for ot,clr,lb in [("call","red","Call做多"),("put","green","Put做空")]:
            sub = tdf[tdf["type"]==ot]
            if len(sub)>0:
                grp = sub.groupby("od")
                fig.add_trace(go.Scatter(
                    x=list(grp.groups.keys()), y=[g["pnl"].sum() for _,g in grp],
                    mode="markers", name=lb,
                    marker=dict(size=10,color=clr,opacity=0.7,line=dict(width=1,color="black"))),
                    row=5, col=1)

    # 布局
    fig.update_layout(height=1100,
        title=dict(text="V3信号驱动香草期权 - %s (%s~%s)" % (sname, str(fd1.date()), str(fd2.date())), x=0.5),
        hovermode="x unified", showlegend=True,
        legend=dict(orientation="h", y=1.01, x=0.5, xanchor="center", font=dict(size=9)),
        margin=dict(l=50,r=30,t=80,b=50), template="plotly_white")

    t = "总收益:%+.0f元(%+.2f%%) 年化:%.2f%% 夏普:%.2f 索提诺:%.2f 卡玛:%.2f<br>" % (pnl,ret,ann,sh,so,ca)
    t += "胜率:%.1f%%, 总交易:%d笔<br>" % (wr, len(tdf))
    t += "网格过滤:%s<br>" % ("开启" if eg else "关闭")
    t += "回撤A:%.2f%%, B:%.2f%%, C:%.2f%%" % (mdd_r,mdd_tv,mdd)
    fig.add_annotation(xref="paper",yref="paper",x=0.98,y=0.98,text=t,showarrow=False,
        font=dict(size=11),align="left",bgcolor="rgba(255,255,255,0.85)",
        bordercolor="gray",borderwidth=1)

    outdir = os.path.dirname(os.path.abspath(__file__))
    gs = "_grid" if eg else "_nofilter"
    of = os.path.join(outdir, "V3_香草回测%s_%s.html" % (gs, datetime.now().strftime("%Y%m%d_%H%M%S")))
    with open(of,"w",encoding="utf-8") as f:
        fig.write_html(f, include_plotlyjs="cdn", include_mathjax=False,
                      full_html=True, auto_open=False,
                      config={"responsive":True,"locale":"zh-CN"})
    print("图表: %s" % of)

# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V3信号驱动香草期权回测")
    parser.add_argument("--capital", type=float, default=200000, help="起始资金")
    parser.add_argument("--enable-grid", action="store_true", help="启用网格过滤")
    parser.add_argument("--quiet", action="store_true", help="静默模式")
    parser.add_argument("--save-nav", action="store_true", help="保存逐日净值")
    parser.add_argument("--start-trade", type=str, default=None, help="交易起始日期")
    args = parser.parse_args()

    results = run_backtest(enable_grid=args.enable_grid,
                          start_capital=args.capital,
                          quiet=args.quiet, save_nav=args.save_nav,
                          start_trade_date=args.start_trade)

    summary = {
        "total_final": results["total_final"],
        "total_pnl": results["total_pnl"],
        "total_return_pct": round(results["total_return_pct"],2),
        "annual_return_pct": round(results["annual_return_pct"],2),
        "max_drawdown_pct": round(results["max_drawdown_pct"],2),
        "max_dd_realized": round(results["max_dd_realized"],2),
        "max_dd_tv": round(results["max_dd_tv"],2),
        "total_trades": results["total_trades"],
        "sharpe": results["sharpe"],
        "sortino": results["sortino"],
        "calmar": results["calmar"],
        "win_rate_pct": results["win_rate_pct"],
        "grid_enabled": args.enable_grid,
    }
    print("[SUMMARY_JSON]" + json.dumps(summary, ensure_ascii=False, default=str) + "[/SUMMARY_JSON]")