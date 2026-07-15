"""
_pricing.py — Black-Scholes 香草期权定价模块

从价差组合定价.py提取的BS公式
"""

import math
from scipy.stats import norm


def bs_call(S, K, T, r, sigma, q=0.0):
    """BSM欧式看涨期权定价"""
    if T <= 1e-10:
        return max(S - K, 0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    call_price = S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return call_price


def bs_put(S, K, T, r, sigma, q=0.0):
    """BSM欧式看跌期权定价"""
    if T <= 1e-10:
        return max(K - S, 0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    put_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)
    return put_price


def spread_binomial_price(current_S, open_index, side, T_rem, r, sigma, q, n=100):
    """
    价差期权二叉树定价（BS近似）
    side: 'bull' (牛市价差: long ATM call + short 105% call)
          'bear' (熊市价差: long ATM put + short 95% put)
    """
    if T_rem <= 1e-10:
        change = current_S / open_index - 1.0
        if side == 'bull':
            return 1_000_000 * max(0, min(change, 0.05))
        else:
            return 1_000_000 * max(0, min(-change, 0.05))

    K_atm = open_index
    if side == 'bull':
        K_otm = open_index * 1.05
        val_per_share = bs_call(current_S, K_atm, T_rem, r, sigma, q) - bs_call(current_S, K_otm, T_rem, r, sigma, q)
    else:
        K_otm = open_index * 0.95
        val_per_share = bs_put(current_S, K_atm, T_rem, r, sigma, q) - bs_put(current_S, K_otm, T_rem, r, sigma, q)

    return val_per_share * (1_000_000 / K_atm)


def calc_fair_premium_binomial(current_index, side, current_sigma):
    """计算开仓时的公平权利金（BS近似）"""
    r = 0.014
    q = 0.025
    if side == 'bull':
        T = 180 / 365.0  # 6M
    else:
        T = 90 / 365.0   # 3M
    return spread_binomial_price(current_index, current_index, side, T, r, current_sigma, q)


if __name__ == "__main__":
    S = 5500.0
    K = 5500.0
    T = 90 / 365.0
    r = 0.014
    sigma = 0.25
    q = 0.025

    call_p = bs_call(S, K, T, r, sigma, q)
    put_p = bs_put(S, K, T, r, sigma, q)
    print(f"ATM Call: {call_p:.4f} 点 (溢价率: {call_p/S*100:.2f}%)")
    print(f"ATM Put:  {put_p:.4f} 点 (溢价率: {put_p/S*100:.2f}%)")
