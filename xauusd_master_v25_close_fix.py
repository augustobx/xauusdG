"""
XAUUSD Master Signal Bot v19
Corrected Tkinter variable initialization error.
GUI configurable global auto-trade score.
Clearer analysis logs in GUI console.
Market regime detection influencing signal scoring.
All prior features (Telegram text lot input, auto-cancel, etc.) retained and refined.
"""

import threading, time, csv, datetime as dt, sys, math, os, pathlib
import pandas as pd, numpy as np
import MetaTrader5 as mt5
import requests
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import pathlib
import json
from json import JSONDecodeError


# ===== CONFIG =================================================
BOT_TOKEN = "7995927150:AAGJr1Mg1PXOLtkoftZXCf45-xQbDPY9V94" 
CHAT_ID   = "5638074896" 
#CHAT_ID   = "-1002268226946" 
PAR = "XAUUSD"
MIN_SL_ROOM_RATIO = 0.33 
MIN_ABSOLUTE_SL_DISTANCE_POINTS = 30 
BOT_TOKEN_ID_STR = BOT_TOKEN.split(':')[0] # V17.1: Guardar el ID del bot para comparaciones
# === Parámetros de Gestión de Riesgo ===
RISK_PCT_PER_TRADE    = 0.01
MIN_REWARD_RISK_RATIO = 2.0
MAX_DAILY_DRAWDOWN    = 0.05

# === Telegram inline control ==================================
API = f"https://api.telegram.org/bot{BOT_TOKEN}"
signals_by_id = {} # Stores signal details: {signal_id: signal_dict}
# V17: {user_id: {"signal_id": sid, "chat_id": cid, "original_signal_message_id": orig_msg_id, "prompt_message_id": prompt_msg_id, "timestamp": ts}}
waiting_for_lot_input = {} 
waiting_for_lot_lock = threading.Lock() # Lock for accessing waiting_for_lot_input


TIMEFRAMES = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5,
              "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
              "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4}

RSI_PERIOD, ATR_PERIOD = 14, 14
ADX_PERIOD_REGIME = 14 # V18: Periodo para el ADX usado en get_market_regime
ATR_BUFFER_FACTOR = 0.10 
SPAM_MINUTES = 30
SPAM_PRICE_FACTOR = 0.5
MAGIC: int = 12345678

SCALPING_TIMEFRAMES = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5}
INTRADAY_TIMEFRAMES = {"M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1}
ANTICIPATION_TIMEFRAMES = {"M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1}
SWING_TIMEFRAMES = {"H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4}
TIMEFRAMES_BY_MODE = {
    "Scalping": SCALPING_TIMEFRAMES, "Intradía": INTRADAY_TIMEFRAMES,
    "Anticipación": ANTICIPATION_TIMEFRAMES, "Swing": SWING_TIMEFRAMES,
    "Default": TIMEFRAMES 
}

SESSION_CONFIG = { 
    "Tokio":   {"mode":"Anticipación", "threshold":2.0, "auto_trade_score_threshold": 2.30, "sl_mult":1.8,"tp_r":2.5, "update_interval_monitor": 30, "max_sl_points": 300, "min_tp_points": 200, "min_net_profit_scalp_pts": 0, "be_trigger_r": 0.7, "be_offset_pts": 15, "max_duration_sec": 8*60*60},
    "Londres": {"mode":"Intradía", "threshold":1.5, "auto_trade_score_threshold": 2.35, "sl_mult":1.5,"tp_r":2.0, "update_interval_monitor": 20, "max_sl_points": 250, "min_tp_points": 150, "min_net_profit_scalp_pts": 0, "be_trigger_r": 0.6, "be_offset_pts": 10, "max_duration_sec": 4*60*60},
    "NY":      {"mode": "Scalping", "threshold": 1.7, "auto_trade_score_threshold": 2.45, "sl_mult": 0.8, "max_sl_points": 200, "min_abs_sl_broker_pts_scalp": 150, "tp_r": 1.0, "min_tp_points": 150, "update_interval_monitor": 3, "min_net_profit_scalp_pts": 40, "be_trigger_r": 0.4, "be_offset_pts": 5, "max_duration_sec": 20*60 },
    "Post":    {"mode":"Swing", "threshold":2.0, "auto_trade_score_threshold": 2.40, "sl_mult":2.0,"tp_r":3.0, "update_interval_monitor": 60, "max_sl_points": 400, "min_tp_points": 300, "min_net_profit_scalp_pts": 0, "be_trigger_r": 0.8, "be_offset_pts": 20, "max_duration_sec": 24*60*60}
}

# === Scalping Optimized Additions (v22) ===
MAX_SPREAD_ATR_RATIO_SCALP = 0.025  # 2.5% of ATR
LOT_SIZE_FACTOR_TEST = 0.25         # Reduce lot size for live tests (set 1.0 for real)
ATR_VOL_FILTER_MULT = 1.2           # Operar solo si ATR actual <= ATR20d * 1.2
CONFIRMATION_MAP = {"M1": "M3", "M3": "M15", "M5": "M15"}  # Multi‑tf confirmación
# =========================================

if not mt5.initialize():
    print(f"{dt.datetime.now()} - MT5 init failed, error code: {mt5.last_error()}"); sys.exit()
else:
    print(f"{dt.datetime.now()} - MT5 initialized successfully.")
    acc_info = mt5.account_info()
    if acc_info: print(f"Account Login: {acc_info.login}, Server: {acc_info.server}, Balance: {acc_info.balance} {acc_info.currency}")
    else: print(f"Could not get account info. Error: {mt5.last_error()}")

DIR_PATH = pathlib.Path(__file__).resolve().parent
LOG_OPEN_PATH = DIR_PATH / "xauusd_trades_open.csv"; LOG_CLOSED_PATH = DIR_PATH / "xauusd_trades_closed.csv"
open_trades_lock = threading.Lock(); signals_by_id_lock = threading.Lock(); plan_lock = threading.Lock()

PLAN_PATH = DIR_PATH / "risk_plan.json"
try:
    with open(PLAN_PATH) as f: PLAN = json.load(f)
except (FileNotFoundError, JSONDecodeError):
    initial_bot_capital = 10000; account_info_init = mt5.account_info()
    if account_info_init: initial_bot_capital = account_info_init.balance; print(f"Initializing PLAN with MT5 balance: {initial_bot_capital}")
    # V18.1: Añadir auto_trade_score_global a la creación inicial del PLAN si el archivo no existe
    PLAN = {"capital": initial_bot_capital, "risk_pct": 0.01, "profit_locked": 0, "profit_target": 300,
            "user_whitelist": [5638074896]} # ELIMINADO: , "auto_trade_score_global": 0.0}
    with open(PLAN_PATH, "w") as f: json.dump(PLAN, f, indent=2)

PLAN.setdefault("profit_locked", 0)
PLAN.setdefault("profit_target", 300)
PLAN.setdefault("capital_start", PLAN.get("capital_start", PLAN["capital"]))
PLAN.setdefault("user_whitelist", [5638074896]) # TU USER ID AQUI
PLAN.setdefault("auto_trade_score_global", 3.5)

user_whitelist = set(PLAN.get("user_whitelist", []))
# V18.1: Leer el valor de auto_trade_score_global para la GUI (esto es si quieres que la GUI lo muestre al inicio)
# La variable auto_trade_score_var en la GUI ya lee de PLAN.get("auto_trade_score_global", 0.0)
# por lo que este setdefault asegura que PLAN tenga la clave.

# --- GUI Setup ---
root = tk.Tk(); root.title("XAUUSD Master v19"); root.geometry("1000x780") 

# V19: Definir variables Tkinter DESPUÉS de root = tk.Tk()
manual_session_mode_override = tk.StringVar(value="AUTO") 
auto_trade_score_manual_var = tk.DoubleVar(value=PLAN.get("auto_trade_score_global", 0.0))

# ... (resto de la configuración de la GUI como en tu script v18)
left=tk.Frame(root); left.pack(side="left",fill="both",expand=True)
console=scrolledtext.ScrolledText(left,wrap=tk.WORD,font=('Consolas',9),state='disabled')
console.pack(fill="both",expand=True)
right=tk.Frame(root,width=320,relief="sunken",bd=1); right.pack(side="right",fill="y")

signals_panel=tk.Frame(right); signals_panel.pack(fill="both",expand=True)
session_lbl=tk.Label(right,font=('Arial',10,'bold')); session_lbl.pack(pady=4)
open_panel = tk.Frame(right); open_panel.pack(fill="both", expand=True, pady=(4, 0))
tk.Label(open_panel, text="Trades abiertos", font=("Arial", 9, "bold")).pack(anchor="w")
risk_frame = tk.Frame(right, relief="groove", bd=1, padx=4, pady=4)
tk.Label(risk_frame, text="Plan de Riesgo y Cuenta", font=("Arial", 9, "bold")).pack()
lbl_equity_mt5 = tk.Label(risk_frame, text="Equidad MT5: N/A")
lbl_balance_mt5 = tk.Label(risk_frame, text="Balance MT5: N/A")
lbl_capital_bot = tk.Label(risk_frame, text="Balance Bot Registrado: N/A")
lbl_risk_pct = tk.Label(risk_frame, text="Riesgo %: N/A")
lbl_risk_usd = tk.Label(risk_frame, text="Riesgo USD (Equidad): N/A")
lbl_locked   = tk.Label(risk_frame, text="Profit Lock: N/A")
lbl_equity_mt5.pack(anchor="w"); lbl_balance_mt5.pack(anchor="w"); lbl_capital_bot.pack(anchor="w")
lbl_risk_pct.pack(anchor="w"); lbl_risk_usd.pack(anchor="w"); lbl_locked.pack(anchor="w")
risk_frame.pack(fill="x", padx=4, pady=6)


auto_trade_score_gui_frame = tk.Frame(right, pady=2) 
tk.Label(auto_trade_score_gui_frame, text="Score Auto-Trade Global (0 desc.):").pack(side="left", padx=2)
auto_trade_score_entry_gui = tk.Entry(auto_trade_score_gui_frame, textvariable=auto_trade_score_manual_var, width=5)
auto_trade_score_entry_gui.pack(side="left", padx=2)


def save_auto_trade_score_from_gui():
    try:
        new_score = auto_trade_score_manual_var.get()
        if new_score < 0: 
            messagebox.showwarning("Valor Inválido", "Score no puede ser negativo. Se usará 0.")
            auto_trade_score_manual_var.set(0.0); new_score = 0.0
        with plan_lock: PLAN["auto_trade_score_global"] = new_score
        save_plan() 
        log(f"Umbral Score Auto-Trade Global guardado: {new_score if new_score > 0 else 'Desactivado'}")
    except tk.TclError: 
        messagebox.showerror("Error de Valor", "Ingrese un número válido para el score.");
        auto_trade_score_manual_var.set(PLAN.get("auto_trade_score_global", 0.0))
tk.Button(auto_trade_score_gui_frame, text="Guardar Score Auto", command=save_auto_trade_score_from_gui).pack(side="left", padx=2)
auto_trade_score_gui_frame.pack(fill="x", padx=4)

manual_mode_frame = tk.Frame(right, pady=2)
tk.Label(manual_mode_frame, text="Forzar Modo Sesión:").pack(side="left", padx=2)
session_modes = ["AUTO"] + list(SESSION_CONFIG.keys()) 
manual_mode_combobox = ttk.Combobox(manual_mode_frame, textvariable=manual_session_mode_override, 
                                    values=session_modes, width=12, state="readonly")
manual_mode_combobox.pack(side="left", padx=2); manual_mode_combobox.set("AUTO") 



def apply_manual_mode(): 
    mode_selected = manual_session_mode_override.get()
    refresh_session_label_manual() 
    if mode_selected == "AUTO": log("Modo de sesión cambiado a: Automático (basado en hora).")
    else: log(f"Modo de sesión FORZADO a: {mode_selected} (Tipo: {SESSION_CONFIG[mode_selected]['mode']}).")
tk.Button(manual_mode_frame, text="Aplicar Modo", command=apply_manual_mode).pack(side="left", padx=2)
manual_mode_frame.pack(fill="x", padx=4)

controls=tk.Frame(right); controls.pack(fill="x",pady=4)
tk.Label(controls,text="Controles Principales").pack()





def calculate_pivots(df):
    prev = df.iloc[-2]
    high, low, close = prev.high, prev.low, prev.close
    pivot = (high + low + close) / 3
    r1 = 2*pivot - low; s1 = 2*pivot - high
    r2 = pivot + (high - low); s2 = pivot - (high - low)
    return {"pivot": pivot, "R1": r1, "R2": r2, "S1": s1, "S2": s2}

def detect_volume_spike(df, mult=2.0):
    vols = df.tick_volume
    mean_vol = vols.iloc[-21:-1].mean()
    return vols.iloc[-1] >= mean_vol * mult if not np.isnan(mean_vol) else False

def is_bullish_engulfing(df):
    o1,c1 = df.open.iloc[-2], df.close.iloc[-2]
    o2,c2 = df.open.iloc[-1], df.close.iloc[-1]
    return (c1 < o1 and c2 > o2 and o2 < c1 and c2 > o1)

def is_hammer(df):
    o,c,h,l = df.open.iloc[-1], df.close.iloc[-1], df.high.iloc[-1], df.low.iloc[-1]
    body = abs(c - o); tail = min(o,c) - l
    return tail >= 2 * body and (h - max(o,c)) <= body


# --- INICIO DE FUNCIONES MODIFICADAS/NUEVAS PARA V17 ---




def save_plan(): # From your v15
    with plan_lock:
        PLAN["user_whitelist"] = list(user_whitelist)
        with open(PLAN_PATH, "w") as f:
            json.dump(PLAN, f, indent=2)

open_trades = []

# ---- v27.1 Fix: definición de already_in_trade ----
def already_in_trade(tf_name, direction):
    """
    Devuelve True si ya existe un trade abierto para el timeframe y dirección dados.
    """
    with open_trades_lock:
        return any(tr.get('tf') == tf_name and tr.get('dir') == direction for tr in open_trades)
# ----------------------------------------------------
last_signal_meta = {}
monitor_running  = False

def now(): return dt.datetime.now()

# V19: cfg() ahora considera el modo manual
def cfg():
    manual_mode = manual_session_mode_override.get() #StringVar global
    if manual_mode != "AUTO" and manual_mode in SESSION_CONFIG:
        return SESSION_CONFIG[manual_mode]
    current_time_session_name = session_name() 
    return SESSION_CONFIG[current_time_session_name]

def session_name():
    h=now().hour+now().minute/60
    if 4<=h<10: return "Londres"
    if 10<=h<18: return "NY"
    if 22<=h or h<2: return "Tokio"
    return "Post"

gui_active = True 
def _log_to_gui_thread_safe(msg_to_log):
    global gui_active
    if gui_active and root and root.winfo_exists():
        try:
            console.configure(state='normal')
            console.insert(tk.END, f"{now().strftime('%H:%M:%S')} | {msg_to_log}\n")
            console.configure(state='disabled'); console.see(tk.END)
        except tk.TclError: gui_active = False; print(f"{now().strftime('%H:%M:%S')} | (GUI Error during log) {msg_to_log}")
    elif not gui_active and root and root.winfo_exists(): print(f"{now().strftime('%H:%M:%S')} | (GUI Inactive/Log) {msg_to_log}")
    else: print(f"{now().strftime('%H:%M:%S')} | {msg_to_log}")

def log(msg):
    global gui_active
    if gui_active and root: 
        try:
            if root.winfo_exists(): root.after(0, lambda: _log_to_gui_thread_safe(msg)); return
        except tk.TclError: gui_active = False 
    print(f"{now().strftime('%H:%M:%S')} | {msg}")

def tg(msg):
    try: requests.post(f"{API}/sendMessage", data={"chat_id":CHAT_ID,"text":msg,"parse_mode":"HTML"},timeout=10)
    except requests.exceptions.RequestException as e: log(f"Telegram send error: {e}")
    except Exception as e: log(f"Telegram generic error: {e}")

def save_csv(path, rows):
    if not rows: return
    hdr=not path.exists(); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path,"a",newline="", encoding="utf-8") as f: 
        w=csv.DictWriter(f,fieldnames=rows[0].keys())
        if hdr: w.writeheader()
        w.writerows(rows)

def df_rates(tf, n=250):
    data=mt5.copy_rates_from_pos(PAR,tf,0,n)
    if data is None or len(data) < n:
        raise RuntimeError(f"No data or insufficient for {PAR} {tf} (req: {n}, got: {len(data) if data else 0})")
    df=pd.DataFrame(data); df['time']=pd.to_datetime(df.time,unit='s'); return df

# --- Indicadores ---
ema=lambda s,n: s.ewm(span=n, adjust=False).mean()
def rsi(s,p):
    d=s.diff(); up=np.where(d>0,d,0); dn=np.where(d<0,-d,0)
    ru=pd.Series(up).rolling(p).mean(); rd=pd.Series(dn).rolling(p).mean()
    rs_val = rd.replace(0, 1e-10); rs_val = ru/rs_val; return 100-100/(1+rs_val)
def atr(df,p):
    tr=pd.concat([(df.high-df.low), abs(df.high-df.close.shift()), abs(df.low-df.close.shift())],axis=1).max(axis=1)
    return tr.rolling(p).mean() # V18: Changed to simple rolling mean as per your original. ewm(alpha=1/p) is Wilder's.
def vwap(df, period=20):
    pv=(df.close*df.tick_volume).rolling(period).sum(); vol=df.tick_volume.rolling(period).sum()
    return pv/vol.replace(0, 1e-10)
def stoch_rsi(rsis, p=14):
    min_r=rsis.rolling(p).min(); max_r=rsis.rolling(p).max()
    denom = (max_r - min_r); return (rsis - min_r) / denom.replace(0, 1)
def macd_hist(s,fast=12,slow=26,signal=9):
    emaf=ema(s,fast); emas=ema(s,slow); macd=emaf-emas; return macd-ema(macd,signal)
def bollinger_b(s,n=20,mult=2):
    ma=s.rolling(n).mean(); std=s.rolling(n).std(); upper=ma+mult*std; lower=ma-mult*std
    denom = (upper - lower); return (s - lower) / denom.replace(0, 1)
def keltner_bias(df,n=20,mult=1.5): # Bias from center MA of Keltner
    ma=df.close.rolling(n).mean(); range_avg=atr(df,n); upper=ma+mult*range_avg; lower=ma-mult*range_avg
    channel_width = (upper-lower).replace(0,1)
    return (df.close-ma) / channel_width 
def cci(df,n=50):
    tp=(df.high+df.low+df.close)/3; ma=tp.rolling(n).mean(); md=abs(tp-ma).rolling(n).mean()
    denom = (0.015 * md); return (tp-ma) / denom.replace(0, np.nan)

def get_dmi_adx(df_input, period=14):
    df_copy = df_input.copy()
    df_copy['h-l'] = df_copy['high'] - df_copy['low']
    df_copy['h-pc'] = abs(df_copy['high'] - df_copy['close'].shift(1))
    df_copy['l-pc'] = abs(df_copy['low'] - df_copy['close'].shift(1))
    df_copy['tr_calc'] = df_copy[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    smooth_tr = df_copy['tr_calc'].ewm(alpha=1/period, adjust=False).mean()
    df_copy['dm_plus_raw'] = np.where((df_copy['high'].diff() > df_copy['low'].diff().abs()) & (df_copy['high'].diff() > 0), df_copy['high'].diff(), 0.0)
    df_copy['dm_minus_raw'] = np.where((df_copy['low'].diff().abs() > df_copy['high'].diff()) & (df_copy['low'].diff().abs() > 0), df_copy['low'].diff().abs(), 0.0)
    smooth_dm_plus = df_copy['dm_plus_raw'].ewm(alpha=1/period, adjust=False).mean()
    smooth_dm_minus = df_copy['dm_minus_raw'].ewm(alpha=1/period, adjust=False).mean()
    safe_smooth_tr = smooth_tr.replace(0, np.nan)
    di_plus = 100 * (smooth_dm_plus / safe_smooth_tr)
    di_minus = 100 * (smooth_dm_minus / safe_smooth_tr)
    dx = (abs(di_plus - di_minus) / (di_plus + di_minus).replace(0, np.nan)) * 100
    adx_line = dx.ewm(alpha=1/period, adjust=False).mean()
    return pd.DataFrame({'plus_di': di_plus, 'minus_di': di_minus, 'adx': adx_line}, index=df_input.index)

def indicators(df_input): # V18: Changed param name to df_input
    df_input['ema_s'] = ema(df_input['close'], 20); df_input['ema_l'] = ema(df_input['close'], 50)
    df_input['ema_bias']= ema(df_input['close'], 200); df_input['rsi'] = rsi(df_input['close'], RSI_PERIOD)
    df_input['atr'] = atr(df_input, ATR_PERIOD) 
    # V18: Usar ADX_PERIOD_REGIME que definimos globalmente
    dmi_adx_df = get_dmi_adx(df_input, period=ADX_PERIOD_REGIME) 
    df_input = pd.concat([df_input, dmi_adx_df], axis=1) # Añadir las nuevas columnas al df
    return df_input


# V18: New function to determine market regime
def get_market_regime(df_input, adx_threshold_trend=25, adx_threshold_range=20):
    if not all(col in df_input.columns for col in ['adx', 'plus_di', 'minus_di']):
        log("Error: Faltan columnas ADX/DI en DataFrame para get_market_regime. Intentando calcular...")
        temp_df_for_adx = df_input.copy() 
        dmi_adx_temp = get_dmi_adx(temp_df_for_adx, period=ADX_PERIOD_REGIME) 
        if not all(col in dmi_adx_temp.columns for col in ['adx', 'plus_di', 'minus_di']):
            log("Error fatal: No se pudieron calcular ADX/DI para get_market_regime.")
            return "UNDEFINED"
        last_adx = dmi_adx_temp['adx'].iloc[-1]
        last_plus_di = dmi_adx_temp['plus_di'].iloc[-1]
        last_minus_di = dmi_adx_temp['minus_di'].iloc[-1]
    else:
        last_adx = df_input['adx'].iloc[-1]
        last_plus_di = df_input['plus_di'].iloc[-1]
        last_minus_di = df_input['minus_di'].iloc[-1]
    if pd.isna(last_adx) or pd.isna(last_plus_di) or pd.isna(last_minus_di): return "UNDEFINED"
    if last_adx >= adx_threshold_trend: return "STRONG_TREND_UP" if last_plus_di > last_minus_di else "STRONG_TREND_DOWN"
    elif last_adx < adx_threshold_range: return "RANGE"
    else: return "WEAK_TREND_UP" if last_plus_di > last_minus_di else "WEAK_TREND_DOWN"


def calc_lots(sl_price, entry_price): # Tu función calc_lots
    account_info = mt5.account_info()
    if not account_info: log("Error: No account info for calc_lots. Using 0.01 lots."); return 0.01
    current_equity = account_info.equity
    with plan_lock: profit_locked = PLAN["profit_locked"]; risk_percentage = PLAN["risk_pct"]
    equity_for_risk = current_equity - profit_locked
    if equity_for_risk <= 0: log(f"Warning: Equity for risk ({equity_for_risk:.2f}) <= 0. Using 0.01 lots."); return 0.01
    risk_usd_to_take = equity_for_risk * risk_percentage
    sym_info = mt5.symbol_info(PAR)
    if not sym_info: log(f"Error: No symbol info {PAR} for calc_lots. Using 0.01 lots."); return 0.01
    point_val, tick_val = sym_info.point, sym_info.trade_tick_value
    volume_step, min_volume = sym_info.volume_step, sym_info.volume_min
    if tick_val is None or tick_val == 0 or point_val == 0: log(f"Error: Invalid tick/point for {PAR}. Using {min_volume} lots."); return min_volume
    sl_points_distance = abs(entry_price - sl_price)
    if sl_points_distance < sym_info.point : log(f"Warning: SL distance ({sl_points_distance}) very small. Using {min_volume} lots."); return min_volume 
    risk_per_lot_at_sl = (sl_points_distance / point_val) * tick_val
    if risk_per_lot_at_sl <= 1e-5: log(f"Warning: Risk per lot at SL ({risk_per_lot_at_sl}) too small. Using {min_volume} lots."); return min_volume 
    calculated_lots = risk_usd_to_take / risk_per_lot_at_sl
    calculated_lots = math.floor(calculated_lots / volume_step) * volume_step
    calculated_lots = max(min_volume, calculated_lots)
    decimals = 0
    if volume_step > 0 and volume_step < 1: decimals = abs(math.floor(math.log10(volume_step)))
    elif volume_step == 1: decimals = 0
    else: decimals = 2 
    return round(calculated_lots, decimals)

def send_mt5_order(sig): # Tu función send_mt5_order
    sym  = mt5.symbol_info(PAR); tick = mt5.symbol_info_tick(PAR)
    if not (sym and tick and tick.time > 0): 
        log(f"⚠️ {sig.get('tf','N/A')} {sig.get('dir','N/A')}: No hay tick/símbolo válido para enviar orden."); return False
    dir_long = (sig["dir"] == "LONG"); typ = mt5.ORDER_TYPE_BUY if dir_long else mt5.ORDER_TYPE_SELL
    actual_order_price = tick.ask if dir_long else tick.bid
    original_signal_entry = sig["entry"]; stop_loss_price = sig["sl"]
    initial_risk_distance_price = abs(original_signal_entry - stop_loss_price)
    if initial_risk_distance_price < sym.point: 
        log(f"⚠️ ORDEN CANCELADA ({sig['dir']}) {sig['tf']}: Distancia SL original ({initial_risk_distance_price/sym.point:.0f} pts) demasiado pequeña.")
        tg(f"⚠️ ORDEN CANCELADA ({sig['dir']}) {sig['tf']}: Distancia SL original muy pequeña.")
        return False
    order_would_hit_sl_immediately = False; current_valid_distance_to_sl = 0.0
    if dir_long: 
        if actual_order_price <= stop_loss_price: order_would_hit_sl_immediately = True
        current_valid_distance_to_sl = actual_order_price - stop_loss_price 
    else: 
        if actual_order_price >= stop_loss_price: order_would_hit_sl_immediately = True
        current_valid_distance_to_sl = stop_loss_price - actual_order_price
    min_absolute_sl_distance_price_value = MIN_ABSOLUTE_SL_DISTANCE_POINTS * sym.point; abort_message = None
    if order_would_hit_sl_immediately:
        abort_message = (f"ORDEN CANCELADA ({sig.get('dir','N/A')}) {sig.get('tf','N/A')}: Precio actual ({actual_order_price:.2f}) "
                         f"activaría inmediatamente el SL ({stop_loss_price:.2f}). Dist: {current_valid_distance_to_sl/sym.point:.0f} pts")
    elif initial_risk_distance_price > sym.point and current_valid_distance_to_sl < (initial_risk_distance_price * MIN_SL_ROOM_RATIO):
        abort_message = (f"ORDEN CANCELADA ({sig.get('dir','N/A')}) {sig.get('tf','N/A')}: Distancia actual a SL ({current_valid_distance_to_sl/sym.point:.0f} pts) "
                         f"es < {MIN_SL_ROOM_RATIO*100:.0f}% de la original ({initial_risk_distance_price/sym.point:.0f} pts). "
                         f"Actual: {actual_order_price:.2f}, SL: {stop_loss_price:.2f}")
    elif current_valid_distance_to_sl < min_absolute_sl_distance_price_value:
        abort_message = (f"ORDEN CANCELADA ({sig.get('dir','N/A')}) {sig.get('tf','N/A')}: Distancia actual a SL ({current_valid_distance_to_sl/sym.point:.0f} pts) "
                         f"es < al mínimo absoluto ({MIN_ABSOLUTE_SL_DISTANCE_POINTS} pts). "
                         f"Actual: {actual_order_price:.2f}, SL: {stop_loss_price:.2f}")
    if abort_message: log(f"⚠️ {abort_message}"); tg(f"⚠️ {abort_message}"); return False
    def _send(fill_mode_param, dev_param=30):
        req = {"action":mt5.TRADE_ACTION_DEAL,"symbol":PAR,"volume":sig["lots"],"type":typ,
               "price":actual_order_price,"sl":sig["sl"],"tp":sig["tp"],"deviation":dev_param,
               "magic":12345,"comment":f"{sig['tf']} {sig['dir']}","type_filling":fill_mode_param,}
        return mt5.order_send(req)
    res = _send(mt5.ORDER_FILLING_IOC)
    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
        sig["ticket"]=res.order; sig["pos_volume"]=res.volume
        log(f"✅ MT5 posición abierta ticket {sig.get('ticket','N/A')} para {sig['tf']} {sig['dir']}")
        tg(f"✅ Orden ejecutada {sig['tf']} {sig['dir']} – ticket {sig.get('ticket','N/A')}"); return True
    if res and res.retcode == 10019:
        allowed_filling_types = sym.filling_modes; fill_ok = None
        if mt5.ORDER_FILLING_FOK in allowed_filling_types: fill_ok = mt5.ORDER_FILLING_FOK
        elif mt5.ORDER_FILLING_RETURN in allowed_filling_types: fill_ok = mt5.ORDER_FILLING_RETURN
        if fill_ok is not None:
            log(f"ℹ️ Retrying order for {sig['tf']} {sig['dir']} with filling mode: {fill_ok} (was IOC due to 10019)")
            res2 = _send(fill_ok)
            if res2 and res2.retcode == mt5.TRADE_RETCODE_DONE:
                sig["ticket"]=res2.order; sig["pos_volume"]=res2.volume
                log(f"✅ MT5 pos abierta ticket {sig.get('ticket','N/A')} para {sig['tf']} {sig['dir']} (retry fill {fill_ok})")
                tg(f"✅ Orden ejecutada {sig['tf']} {sig['dir']} – ticket {sig.get('ticket','N/A')}"); return True
            res = res2 
        else: log(f"⚠️ No alternative filling mode for {sig['tf']} {sig['dir']} after IOC failed with 10019.")
    retcode_log = res.retcode if res else "No Response"; comment_log = res.comment if res else "N/A"
    log(f"⚠️ No se pudo enviar orden para {sig['tf']} {sig['dir']} ({retcode_log} - {comment_log})")
    tg(f"⚠️ No se pudo enviar orden {sig['tf']} {sig['dir']} ({retcode_log})"); return False

def close_mt5_position(tr):
    import time
    # 1) Validar ticket y obtener posición
    ticket = tr.get("ticket")
    if not ticket:
        log(f"⚠️ Ticket no encontrado para cerrar trade ID {tr.get('id','N/A')}")
        return False
    pos_list = mt5.positions_get(ticket=ticket)
    if not pos_list:
        log(f"⚠️ Posición {ticket} no encontrada en MT5 para cerrar.")
        return False
    pos = pos_list[0]
    symbol = pos.symbol

    # 2) Volumen y tipo de cierre
    vol = round(pos.volume, 2)
    close_type = mt5.ORDER_TYPE_SELL if tr.get("dir") == "LONG" else mt5.ORDER_TYPE_BUY

    # 3) Modo de filling correcto
    filling_mode_close = mt5.ORDER_FILLING_IOC

    # 4) Intentos de cierre con distintos desvíos
    final_res = None
    for dev in (30, 60, 100):
        tick = mt5.symbol_info_tick(symbol)
        if not tick or tick.time == 0:
            log(f"⚠️ No hay tick info para cerrar {ticket}")
            time.sleep(1)
            continue
        price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask

        req = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "position":     ticket,
            "symbol":       symbol,
            "volume":       vol,
            "type":         close_type,
            "price":        price,
            "deviation":    dev,
            "type_filling": filling_mode_close,
            "magic":        MAGIC,
            "comment":      "Close by bot"
        }
        final_res = mt5.order_send(req)
        if final_res and final_res.retcode == mt5.TRADE_RETCODE_DONE:
            log(f"✅ Posición {ticket} cerrada (dev={dev})")
            return True
        if final_res and final_res.retcode not in (10030, 10016, 10025):
            break
        log(f"ℹ️ Re-intentando cierre para {ticket} (dev={dev}, retcode={final_res.retcode if final_res else 'N/A'})...")
        time.sleep(0.2)

    # 5) Log final con detalle
    retcode_log = final_res.retcode if final_res else "No Response"
    comment_log = final_res.comment if final_res else "N/A"
    log(f"⚠️ MT5 close error final para {ticket}: {retcode_log} [{comment_log}]")
    log(f"🔍 Request final: {req}")
    return False




def _destroy_widget_safe(widget):
    if widget and widget.winfo_exists(): widget.destroy()

def add_signal_ui(sig):
    if not root.winfo_exists(): return
    root.after(0, lambda s=sig: _add_signal_ui_actual(s))

def _add_signal_ui_actual(sig):
    f = tk.Frame(signals_panel, relief="ridge", bd=1, padx=2, pady=2)
    tk.Label(f, text=f"{sig['tf']} {sig['dir']} {sig['entry']:.2f} SL {sig['sl']:.2f}", font=("Consolas", 8)).pack(side="left")
    tk.Button(f, text="Tomar", command=lambda s_param=sig, fr_param=f: take_signal(s_param, fr_param)).pack(side="right", padx=2)
    def rechazar():
        tg(f"✖️ Señal descartada {sig['tf']} {sig['dir']}"); log(f"DESCARTADA {sig['tf']} {sig['dir']} {sig['entry']:.2f}")
        _destroy_widget_safe(f); # V15: ; is fine, it's sequential
        with signals_by_id_lock: signals_by_id.pop(sig['id'], None)
    tk.Button(f, text="Rechazar", command=rechazar).pack(side="right", padx=2)
    f.pack(fill="x", pady=1); sig["ui_frame"] = f

def refresh_open_panel():
    if not root.winfo_exists(): return
    root.after(0, _refresh_open_panel_actual)

def _refresh_open_panel_actual():
    for w in open_panel.winfo_children()[1:]: _destroy_widget_safe(w)
    with open_trades_lock: current_open_trades = list(open_trades)
    for tr in current_open_trades:
        line = tk.Frame(open_panel); txt  = f"{tr['tf']} {tr['dir']} @ {tr['entry']:.2f}"
        tk.Label(line, text=txt, font=("Consolas", 8)).pack(side="left")
        tk.Button(line, text="Cerrar", width=6, command=lambda t=tr: manual_close(t)).pack(side="right")
        line.pack(fill="x", padx=2, pady=1)

def manual_close(tr_to_close): # Tu función
    pos_exists_manual = mt5.positions_get(ticket=tr_to_close.get("ticket"))
    if not pos_exists_manual:
        log(f"MANUAL CLOSE: Posición {tr_to_close.get('ticket','N/A')} ya no existe. Registrando cierre.")
        last_tick_manual = mt5.symbol_info_tick(PAR)
        price_manual = last_tick_manual.bid if tr_to_close["dir"] == "LONG" else last_tick_manual.ask if last_tick_manual and last_tick_manual.time > 0 else tr_to_close["entry"]
        close_trade(tr_to_close, price_manual, "MANUAL_ALREADY_CLOSED")
    elif close_mt5_position(tr_to_close):
        tick  = mt5.symbol_info_tick(PAR); price = tick.bid if tr_to_close["dir"] == "LONG" else tick.ask
        close_trade(tr_to_close, price, "MANUAL_GUI")
    else:
        log(f"⚠️ MANUAL CLOSE: Fallo al cerrar {tr_to_close.get('ticket','N/A')} desde GUI.")
        tg(f"⚠️ MANUAL CLOSE: Fallo al cerrar {tr_to_close.get('ticket','N/A')}. Verificar MT5.")
        refresh_risk_labels(); return
    with open_trades_lock:
        if tr_to_close in open_trades: open_trades.remove(tr_to_close)
    refresh_open_panel(); refresh_risk_labels()

# V19: `cancel_all_pending_signals` (basada en V17)
def cancel_all_pending_signals(taken_signal_id=None):
    cancelled_signals_summary = [] 
    sids_cancelled_in_main = [] 
    with signals_by_id_lock:
        sids_to_remove_from_main = [sid for sid in signals_by_id if sid != taken_signal_id]
        for sid in sids_to_remove_from_main:
            pending_sig = signals_by_id.pop(sid, None)
            if pending_sig:
                sids_cancelled_in_main.append(sid)
                info = f"{pending_sig['tf']} {pending_sig['dir']} @ {pending_sig['entry']:.2f}"
                if not any(summary_info.startswith(info.split(" @")[0]) for summary_info in cancelled_signals_summary):
                    cancelled_signals_summary.append(info)
                log(f"Auto-Cancel (main list): {info} (ID: {sid})")
                if root.winfo_exists() and pending_sig.get("ui_frame"):
                    root.after(0, lambda widget=pending_sig.get("ui_frame"): _destroy_widget_safe(widget))
                if "tg_message_id" in pending_sig:
                    try: requests.post(f"{API}/editMessageReplyMarkup", data={"chat_id": CHAT_ID, "message_id": pending_sig["tg_message_id"], "reply_markup": json.dumps({})}, timeout=5)
                    except Exception as e: log(f"Error editing TG reply_markup for signal {sid}: {e}")
    with waiting_for_lot_lock:
        users_to_clear_prompt = []
        for user_id, prompt_data in list(waiting_for_lot_input.items()): 
            if prompt_data["signal_id"] != taken_signal_id and prompt_data["signal_id"] in sids_cancelled_in_main:
                users_to_clear_prompt.append(user_id)
                try: requests.post(f"{API}/editMessageText", data={"chat_id": prompt_data["chat_id"], "message_id": prompt_data["prompt_message_id"], "text": "Esta solicitud de lotaje ha sido cancelada porque se tomó otra operación o la señal original ya no es válida.", "parse_mode": "HTML"}, timeout=5)
                except Exception as e: log(f"Error editando prompt de lotaje cancelado para user {user_id}: {e}")
        for user_id in users_to_clear_prompt: waiting_for_lot_input.pop(user_id, None); log(f"Prompt de lotaje para usuario {user_id} cancelado.")
    if cancelled_signals_summary:
        log(f"Señal tomada. {len(cancelled_signals_summary)} señales pendientes fueron auto-canceladas.")
        tg_summary_message = f"⚠️ <b>Señales Pendientes Auto-Canceladas</b> ⚠️\n(Debido a una nueva operación confirmada)\n\n"
        for item_info in cancelled_signals_summary: tg_summary_message += f"  - <code>{item_info}</code>\n"
        tg(tg_summary_message)

# V19: `take_signal` (basada en V17, llama a cancel_all_pending_signals)
def take_signal(sig, frame=None): 
    if frame is None: 
        if "lots" not in sig or not isinstance(sig.get("lots"), (int, float)) or sig.get("lots",0) <= 0: 
            log(f"Error: Lotaje inválido/no especificado para señal {sig.get('id')} en take_signal (Telegram): {sig.get('lots')}")
            error_message_tg = f"⚠️ Error: Lotaje inválido ({sig.get('lots')}) para la señal. Intente de nuevo."
            origin_chat_id = sig.get("chat_id_origin", CHAT_ID)
            try: requests.post(f"{API}/sendMessage", data={"chat_id": origin_chat_id, "text": error_message_tg}, timeout=5)
            except: pass; return 
        sig["opened"] = now()
        order_successful = send_mt5_order(sig) 
        if order_successful and sig.get("ticket"):
            with open_trades_lock: open_trades.append(sig)
            save_csv(LOG_OPEN_PATH, [sig])
            log(f"TOMADO (TG/Input) {sig['tf']} {sig['dir']} lots {sig['lots']} - Ticket: {sig.get('ticket')}")
            cancel_all_pending_signals(taken_signal_id=sig['id']) 
        else: log(f"Fallo al abrir orden para señal {sig['id']} desde TG/Input. No se añade a open_trades."); return 
        refresh_open_panel()
        if "ui_frame" in sig and sig.get("ui_frame") and sig["ui_frame"].winfo_exists(): 
             root.after(0, lambda widget=sig["ui_frame"]: _destroy_widget_safe(widget))
        refresh_risk_labels(); return
    top = tk.Toplevel(root); top.title("Confirmar Trade XAUUSD"); top.transient(root); top.grab_set()
    tk.Label(top, text=f"{sig['tf']} {sig['dir']} @ {sig['entry']:.2f} (SL: {sig['sl']:.2f})", font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=3, pady=(10, 5), padx=10, sticky="ew")
    lote_recomendado = calc_lots(sig["sl"], sig["entry"])
    tk.Label(top, text="Lote Recomendado (Riesgo):").grid(row=1, column=0, sticky="w", padx=10, pady=2)
    lots_var = tk.DoubleVar(value=lote_recomendado)
    entry_lotes = tk.Entry(top, textvariable=lots_var, width=10)
    entry_lotes.grid(row=1, column=1, pady=2, sticky="w"); tk.Label(top, text="lotes").grid(row=1, column=2, sticky="w", padx=2)
    try: 
        acc_info_gui = mt5.account_info(); sym_info_gui = mt5.symbol_info(PAR); tick_gui = mt5.symbol_info_tick(PAR)
        if acc_info_gui and sym_info_gui and tick_gui and tick_gui.time > 0 :
            free_margin_gui = acc_info_gui.margin_free
            price_margin_calc = tick_gui.ask if sig["dir"] == "LONG" else tick_gui.bid
            order_type_margin = mt5.ORDER_TYPE_BUY if sig["dir"] == "LONG" else mt5.ORDER_TYPE_SELL
            margin_per_lot_gui = mt5.order_calc_margin(order_type_margin, PAR, 1.0, price_margin_calc)
            max_lot_margin_gui = 0
            if margin_per_lot_gui and margin_per_lot_gui > 0:
                max_lot_margin_gui = math.floor((free_margin_gui / margin_per_lot_gui) * 100) / 100; max_lot_margin_gui = max(0, max_lot_margin_gui)
            tk.Label(top, text="Lote Máx. por Margen (aprox.):").grid(row=2, column=0, sticky="w", padx=10, pady=2)
            tk.Label(top, text=f"{max_lot_margin_gui:.2f}", fg="blue").grid(row=2, column=1, sticky="w"); tk.Label(top, text="lotes").grid(row=2, column=2, sticky="w", padx=2)
            max_lot_broker_gui = sym_info_gui.volume_max
            tk.Label(top, text="Lote Máx. por Orden (Broker):").grid(row=3, column=0, sticky="w", padx=10, pady=2)
            tk.Label(top, text=f"{max_lot_broker_gui:.2f}", fg="blue").grid(row=3, column=1, sticky="w"); tk.Label(top, text="lotes").grid(row=3, column=2, sticky="w", padx=2)
            min_lot_broker_gui = sym_info_gui.volume_min
            tk.Label(top, text="Lote Mín. por Orden (Broker):").grid(row=4, column=0, sticky="w", padx=10, pady=2)
            tk.Label(top, text=f"{min_lot_broker_gui:.2f}", fg="blue").grid(row=4, column=1, sticky="w"); tk.Label(top, text="lotes").grid(row=4, column=2, sticky="w", padx=2)
        else: tk.Label(top, text="No se pudo obtener info de máx. lotaje (tick inválido o sin info).").grid(row=2, column=0, columnspan=3, sticky="w", padx=10, pady=2)
    except Exception as e: log(f"Error calculando lotaje máximo para GUI: {e}"); tk.Label(top, text="Error calculando máx. lotaje.").grid(row=2, column=0, columnspan=3, sticky="w", padx=10, pady=2)
    tk.Label(top, text="Advertencia: Lote máx. por margen ignora gestión de riesgo.", font=("Arial", 7), fg="red").grid(row=5, column=0, columnspan=3, pady=(5,10), padx=10, sticky="ew")
    def aceptar():
        lots_entered = lots_var.get()
        min_lot_check = mt5.symbol_info(PAR).volume_min if mt5.symbol_info(PAR) else 0.01
        if lots_entered <= 0 or lots_entered < min_lot_check:
            messagebox.showerror("Error de Lote", f"El lotaje debe ser positivo y >= {min_lot_check:.2f}.", parent=top); return
        sig.update({"lots": lots_entered, "opened": now()})
        order_successful = send_mt5_order(sig) 
        if order_successful and sig.get("ticket"):
            with open_trades_lock: open_trades.append(sig)
            save_csv(LOG_OPEN_PATH, [sig])
            log(f"TOMADO (GUI) {sig['tf']} {sig['dir']} lots {lots_entered} - Ticket: {sig.get('ticket')}")
            cancel_all_pending_signals(taken_signal_id=sig['id']) 
        else: log(f"Fallo al abrir orden para señal {sig['id']} desde GUI."); _destroy_widget_safe(top); return
        refresh_open_panel()
        if frame and frame.winfo_exists(): _destroy_widget_safe(frame) 
        refresh_risk_labels(); _destroy_widget_safe(top)
    tk.Button(top, text="Aceptar Trade", command=aceptar, bg="green", fg="white", font=("Arial", 10, "bold")).grid(row=6, column=0, columnspan=3, pady=(5,10), padx=10, ipady=5, sticky="ew")
    top.update_idletasks(); width = top.winfo_width(); height = top.winfo_height()
    x = (top.winfo_screenwidth()//2)-(width//2); y = (top.winfo_screenheight()//2)-(height//2)
    top.geometry(f'{width}x{height}+{x}+{y}'); entry_lotes.focus_set(); root.wait_window(top)

# tg_signal de V16 (guarda message_id y original_text_tg)
def tg_signal(sig):
    text = (f"🎯 <b>{sig['tf']} {sig['dir']}</b>\n💰 <b>Entry:</b> <code>{sig['entry']:.2f}</code>\n"
            f"🛑 <b>SL:</b>    <code>{sig['sl']:.2f}</code>\n🎯 <b>TP:</b>    <code>{sig['tp']:.2f}</code>\n"
            f"📝 <b>Score:</b> <code>{sig['score']}</code>")
    sig["original_text_tg"] = text # V17: Guardar el texto original para editarlo después
    kb ={"inline_keyboard":[[{"text": "Tomar", "callback_data": f"take_{sig['id']}"},
                             {"text": "Rechazar", "callback_data": f"drop_{sig['id']}"}]]}
    try:
        response = requests.post(f"{API}/sendMessage", data={"chat_id": CHAT_ID, "text": text,"parse_mode": "HTML", "reply_markup": json.dumps(kb)}, timeout=10)
        response.raise_for_status(); response_data = response.json() 
        if response_data.get("ok"):
            sig["tg_message_id"] = response_data["result"]["message_id"] 
            log(f"Señal {sig['id']} enviada a Telegram, message_id: {sig['tg_message_id']}")
        else: log(f"Error en resp Telegram al enviar señal {sig['id']}: {response_data.get('description')}")
        with signals_by_id_lock: signals_by_id[sig['id']] = sig 
    except requests.exceptions.RequestException as e: log(f"Error enviando señal {sig['id']} a TG: {e}")
    except Exception as e: log(f"Error genérico en tg_signal para señal {sig['id']}: {e}")

# V18.3: bonus_by_mode con lógica VWAP ligeramente reestructurada para claridad y Pylance.
def bonus_by_mode(df, tf_name, direction, market_regime_param="UNDEFINED"):
    mode = cfg()['mode']
    min_data_points = max(20, RSI_PERIOD, ATR_PERIOD, 50, 200, ADX_PERIOD_REGIME + 10)
    if len(df) < min_data_points:
        return 0.0
    
    last = df.iloc[-1]
    bonus = 0.0 
    comp_bonus_log = {} # Para logging

    try:
        if mode == "Scalping":
            vwap_b, stoch_b, ema_micro_b = 0.0, 0.0, 0.0
            
            # Copiar DataFrames/Series si las funciones de indicadores pudieran modificarlos
            # Si tus funciones de indicadores no modifican el df de entrada, .copy() no es estrictamente necesario
            v = vwap(df) # Asumiendo que vwap no modifica 'df'
            sr_series = stoch_rsi(df['rsi']) # Asumiendo que 'rsi' ya está en df
            ema_micro_val = ema(df['close'], 8)

            # --- Lógica VWAP (Reestructurada) ---
            if not pd.isna(v.iloc[-1]):
                is_price_aligned_with_vwap_for_direction = (direction == "LONG" and last.close > v.iloc[-1]) or \
                                                           (direction == "SHORT" and last.close < v.iloc[-1])
                
                is_vwap_rebound_in_range = (market_regime_param == "RANGE" and \
                                            ((direction == "LONG" and last.low < v.iloc[-1] < last.close and last.close > last.open) or \
                                             (direction == "SHORT" and last.high > v.iloc[-1] > last.close and last.close < last.open)))
                
                if "TREND" in market_regime_param and is_price_aligned_with_vwap_for_direction:
                    vwap_b = 0.20
                elif market_regime_param == "RANGE" and is_vwap_rebound_in_range:
                    vwap_b = 0.15
                elif is_price_aligned_with_vwap_for_direction: # Condición base si no hay régimen específico o no coincide
                    vwap_b = 0.10
            comp_bonus_log["V"] = vwap_b

            # --- Lógica StochRSI ---
            if not pd.isna(sr_series.iloc[-1]) and (not ema_micro_val.empty and not pd.isna(ema_micro_val.iloc[-1])): # V18.3 Check ema_micro_val
                sr = sr_series.iloc[-1]
                stoch_range_cond = (direction == "LONG" and sr < 0.15) or (direction == "SHORT" and sr > 0.85)
                stoch_trend_pullback_cond = ("TREND" in market_regime_param and \
                                            ((direction == "LONG" and sr < 0.3 and last.close > ema_micro_val.iloc[-1]) or \
                                             (direction == "SHORT" and sr > 0.7 and last.close < ema_micro_val.iloc[-1])))
                stoch_base_cond = (direction == "LONG" and sr < 0.2) or (direction == "SHORT" and sr > 0.8)

                if market_regime_param == "RANGE" and stoch_range_cond: stoch_b = 0.25
                elif stoch_trend_pullback_cond: stoch_b = 0.15
                elif stoch_base_cond: stoch_b = 0.10
            comp_bonus_log["S"] = stoch_b
            
            # --- Lógica EMA Micro ---
            if not ema_micro_val.empty and not pd.isna(ema_micro_val.iloc[-1]): # V18.3 Check ema_micro_val
                ema_micro_trend_cond = "TREND" in market_regime_param and \
                                       ((direction == "LONG" and last.close > ema_micro_val.iloc[-1]) or \
                                        (direction == "SHORT" and last.close < ema_micro_val.iloc[-1]))
                if ema_micro_trend_cond: ema_micro_b = 0.20
            comp_bonus_log["E8"] = ema_micro_b
            bonus = vwap_b + stoch_b + ema_micro_b
        
        elif mode == "Intradía": # MACD Hist, Bollinger %B
            macd_b, bb_b = 0.0, 0.0
            mh_series = macd_hist(df['close'].copy()); bb_series = bollinger_b(df['close'].copy()) # Usar .copy() por seguridad
            if not pd.isna(mh_series.iloc[-1]):
                mh = mh_series.iloc[-1]
                if "TREND" in market_regime_param and ((direction == "LONG" and mh > 1e-5) or (direction == "SHORT" and mh < -1e-5)):
                    macd_b = 0.25
                elif market_regime_param == "RANGE" and abs(mh) < 1e-5: 
                    macd_b = 0.05 
                elif ((direction == "LONG" and mh > 0) or (direction == "SHORT" and mh < 0)): 
                    macd_b = 0.15
            comp_bonus_log["MACD"] = macd_b
            
            if not pd.isna(bb_series.iloc[-1]):
                bb = bb_series.iloc[-1]
                # V18.3: Asegurar que ema(df.close,10) tiene suficientes datos y no es NaN
                ema10_val = ema(df.close,10)
                if not ema10_val.empty and not pd.isna(ema10_val.iloc[-1]):
                    if market_regime_param == "RANGE" and ((direction == "LONG" and bb < 0.1) or (direction == "SHORT" and bb > 0.9)): 
                        bb_b = 0.25
                    elif "TREND" in market_regime_param and ((direction == "LONG" and bb > 0.75 and last.close > ema10_val.iloc[-1]) or \
                                                             (direction == "SHORT" and bb < 0.25 and last.close < ema10_val.iloc[-1])):
                        bb_b = 0.20
                    elif ((direction == "LONG" and bb < 0.2) or (direction == "SHORT" and bb > 0.8)): 
                        bb_b = 0.10
            comp_bonus_log["BB%"] = bb_b
            bonus = macd_b + bb_b

        elif mode == "Anticipación": 
            keltner_b, cci_b = 0.0, 0.0
            kb_series=keltner_bias(df.copy()); c_series=cci(df.copy()) 
            if not pd.isna(kb_series.iloc[-1]):
                kb = kb_series.iloc[-1] 
                if market_regime_param == "RANGE" and ((direction == "LONG" and kb < -0.45) or (direction == "SHORT" and kb > 0.45)): 
                    keltner_b = 0.25
                elif "TREND" in market_regime_param and ((direction == "LONG" and kb > 0.1) or (direction == "SHORT" and kb < -0.1)): 
                    keltner_b = 0.20
            comp_bonus_log["Keltner"] = keltner_b
            
            if not pd.isna(c_series.iloc[-1]):
                c = c_series.iloc[-1]
                if (direction == "LONG" and c < -120) or (direction == "SHORT" and c > 120): 
                    cci_b = 0.25
            comp_bonus_log["CCI"] = cci_b
            bonus = keltner_b + cci_b

        elif mode == "Swing": 
            adx_str_b, fib_b = 0.0, 0.0
            last_adx_val = df['adx'].iloc[-1] if 'adx' in df.columns and not pd.isna(df['adx'].iloc[-1]) else np.nan
            
            if not pd.isna(last_adx_val):
                if "STRONG_TREND" in market_regime_param and last_adx_val >=25 : adx_str_b = 0.25
                elif "WEAK_TREND" in market_regime_param and last_adx_val >=20 : adx_str_b = 0.10 
            comp_bonus_log["ADXStr"] = adx_str_b
            
            if len(df) >= 20: 
                recent_high=df.high.iloc[-20:].max(); recent_low=df.low.iloc[-20:].min()
                if not (pd.isna(recent_high) or pd.isna(recent_low) or recent_high == recent_low):
                    ext = recent_high + 0.618*(recent_high-recent_low) if direction=="LONG" else recent_low - 0.618*(recent_high-recent_low)
                    if "TREND" in market_regime_param and \
                       ((direction=="LONG" and last.close < ext) or (direction=="SHORT" and last.close > ext)):
                        fib_b = 0.25
            comp_bonus_log["FibExt"] = fib_b
            bonus = adx_str_b + fib_b
            
    except IndexError: log(f"Error IndexError en bonus_by_mode ({mode},{tf_name})")
    except Exception as e: log(f"Error bonus_by_mode ({mode},{tf_name}): {type(e).__name__} {e}")
    
    active_bonuses_str = ", ".join([f"{k}={v:.2f}" for k,v in comp_bonus_log.items() if v != 0.0])
    if active_bonuses_str or mode == "Scalping" or cfg()['mode'] == mode : 
        log(f"BONUS_DETAILS ({tf_name} {direction} M:{mode} R:{market_regime_param}): Comp=[{active_bonuses_str}] TotalModeBonus={bonus:.2f}")
    return bonus


def analyse(tf_name, tf_code):
    # MOD v27: si ya hay cualquier trade abierto, suspender nuevas señales
    with open_trades_lock:
        if open_trades:
            return

    current_session_cfg = cfg()
    current_mode = current_session_cfg['mode']

    try:
        required_bars = 250
        df_raw = df_rates(tf_code, n=required_bars)
        df = indicators(df_raw)
    except RuntimeError as e:
        log(f"ANALYSE_ERROR ({tf_name}): Data error: {e}")
        return
    except Exception as e:
        log(f"ANALYSE_ERROR ({tf_name}): Indicator calc error: {e}")
        return

    # 1) primero definimos last, sym_point y direction
    last = df.iloc[-1]
    sym_info_analyse = mt5.symbol_info(PAR)
    if not sym_info_analyse:
        log(f"ANALYSE_ERROR ({tf_name}): No symbol info for {PAR}")
        return
    sym_point = sym_info_analyse.point
    direction = "LONG" if last.ema_s > last.ema_l else "SHORT"

    # 2) ahora calculamos pivotes y los bonos usando esas variables
    pivots    = calculate_pivots(df_rates(mt5.TIMEFRAME_H1, n=50))
    price     = last.close
    near_SR   = any(abs(price - pivots[k]) < sym_point * 5 for k in ["pivot", "R1", "S1"])
    vol_spike = detect_volume_spike(df)
    bull_eng  = is_bullish_engulfing(df)
    hamm      = is_hammer(df)

    # 3) inicializamos score y aplicamos bonos S/R, volumen y patrón
    score = 1.0
    sr_bonus      = 0.15 if near_SR else 0.0
    vol_bonus     = 0.10 if vol_spike else 0.0
    pattern_bonus = 0.20 if (bull_eng and direction == "LONG") else 0.0
    score        += sr_bonus + vol_bonus + pattern_bonus
    log(f"BONUSES (SR={sr_bonus:+.2f},VOL={vol_bonus:+.2f},PAT={pattern_bonus:+.2f})")

    # 4) comprobaciones ATR
    atr_v = last.atr
    if pd.isna(atr_v) or atr_v < (sym_point * 5):
        return

    if already_in_trade(tf_name, direction):
        return

    # 5) más cálculo de score: EMA200, RSI, penalidades, bonus modo
    ema200_ok     = ((direction == "LONG"  and last.close > last.ema_bias) or
                     (direction == "SHORT" and last.close < last.ema_bias))
    rsi_dir_ok    = ((direction == "LONG"  and last.rsi > 50) or
                     (direction == "SHORT" and last.rsi < 50))
    rsi_prox_bonus = 0.0
    if not pd.isna(last.rsi):
        rsi_prox_bonus = max(0, (70 - abs(last.rsi - 50)) / 100)

    if ema200_ok:  score += 0.2
    if rsi_dir_ok: score += 0.2
    score += rsi_prox_bonus

    regime_penalty = 0.0
    market_regime  = get_market_regime(df)
    if ((direction == "LONG" and "DOWN" in market_regime) or
        (direction == "SHORT" and "UP" in market_regime)):
        if "STRONG_TREND" in market_regime:
            regime_penalty = 0.5
        elif "WEAK_TREND" in market_regime:
            regime_penalty = 0.25
        score -= regime_penalty

    # Bonus por modo de sesión
    score_bonus_modo = bonus_by_mode(df, tf_name, direction, market_regime_param=market_regime)
    score += score_bonus_modo

    log(f"ANALYSE_SCORE ({tf_name} {direction}) Score {score:.2f}")

    # 6) Umbrales auto‐trade y manual
    session_specific_auto_threshold = current_session_cfg.get("auto_trade_score_threshold", 0.0)
    with plan_lock:
        global_auto_threshold = PLAN.get("auto_trade_score_global", 0.0)
    final_auto_trade_threshold = max(session_specific_auto_threshold, global_auto_threshold)
    if final_auto_trade_threshold > 0:
        log(f"Auto-Trade: Usando umbral de sesión ({final_auto_trade_threshold}).")
    else:
        log(f"Auto-Trade: Desactivado (umbral {final_auto_trade_threshold}).")

    manual_signal_threshold = current_session_cfg['threshold']
    grace_margin             = 0.10
    strong_mode_bonus_min    = 0.40
    if (manual_signal_threshold > score >= (manual_signal_threshold - grace_margin)
        and score_bonus_modo >= strong_mode_bonus_min):
        log(f"ANALYSE_FLEX ({tf_name} {direction}): Score {score:.2f} cercano a umbral {manual_signal_threshold} con bonus modo fuerte ({score_bonus_modo:.2f}). Empujón a {manual_signal_threshold:.2f}.")
        score = manual_signal_threshold

    # 7) Cálculos de Entry, SL y TP
    entry = last.close
    sl_dist_atr = current_session_cfg['sl_mult'] * atr_v
    min_abs_key = ("min_abs_sl_broker_pts_scalp"
                   if current_mode == "Scalping"
                   else "min_abs_sl_broker_pts_default")
    min_sl_pts  = current_session_cfg.get(min_abs_key, 50 if current_mode=="Scalping" else 100)

    if current_mode == "Scalping" and 'max_sl_points' in current_session_cfg:
        max_sl_abs = current_session_cfg['max_sl_points'] * sym_point
        sl_dist = min(sl_dist_atr, max_sl_abs) if max_sl_abs > 0 else sl_dist_atr
    else:
        sl_dist = max(sl_dist_atr, sym_point * 30)
    sl_dist = max(sl_dist, min_sl_pts * sym_point)
    sl = (round(entry - sl_dist, 5)
          if direction == "LONG"
          else round(entry + sl_dist, 5))

    tp_dist = sl_dist * current_session_cfg['tp_r']
    if current_mode == "Scalping" and 'min_tp_points' in current_session_cfg:
        min_tp_abs = current_session_cfg['min_tp_points'] * sym_point
        tp_dist    = max(tp_dist, min_tp_abs) if min_tp_abs > 0 else tp_dist
    tp_dist = max(tp_dist, min_sl_pts * sym_point * 0.5)
    tp = (round(entry + tp_dist, 5)
          if direction == "LONG"
          else round(entry - tp_dist, 5))

    # 8) Filtro Reward–Risk mínimo
    if abs(tp - entry) / abs(entry - sl) < MIN_REWARD_RISK_RATIO:
        log(f"DESCARTADO por RR={(abs(tp-entry)/abs(entry-sl)):.2f} < {MIN_REWARD_RISK_RATIO}")
        return

    # 9) Filtro de Spread para Scalping
    if current_mode == "Scalping":
        spread_pts = sym_info_analyse.spread
        tp_pts     = abs(tp - entry) / sym_point if sym_point>0 else float('inf')
        min_net    = current_session_cfg.get('min_net_profit_scalp_pts', 50)
        if (tp_pts - spread_pts) < min_net:
            log(f"SCALPING ({tf_name} {direction}): Descartado por spread. TP_pts={tp_pts:.0f}, Spread={spread_pts}.")
            return

    # 10) Filtro Anti‐Spam
    signal_key = (tf_name, direction)
    if signal_key in last_signal_meta:
        ts_prev, price_prev = last_signal_meta[signal_key]
        mins = (now() - ts_prev).total_seconds() / 60
        if mins < SPAM_MINUTES and abs(entry - price_prev) < atr_v * SPAM_PRICE_FACTOR:
            return

    # 11) Crear diccionario de señal
    sig_id = int(time.time() * 1000)
    base_sig_dict = {
        "id": sig_id, "created": now(), "tf": tf_name, "dir": direction,
        "entry": entry, "sl": sl, "tp": tp,
        "score": round(score,2),
        "update_interval": current_session_cfg.get("update_interval_monitor",60),
        "sent_1r": False, "sent_tp": False, "sent_sl": False, "sent_near_sl": False,
        "breakeven_applied": False, "last_update": now(), "last_bucket": None, "last_reco": "",
        "be_trigger_r": current_session_cfg.get("be_trigger_r",0.5),
        "be_offset_pts": current_session_cfg.get("be_offset_pts",10),
        "max_duration_sec": current_session_cfg.get("max_duration_sec",15*60),
        "market_regime_at_signal": market_regime,
        "symbol": PAR
    }

    # 12) Decisión de Auto-Trade
    if final_auto_trade_threshold > 0 and score >= final_auto_trade_threshold:
        log(f"AUTO-TRADE: Señal {tf_name} {direction} (Score: {score:.2f}) >= AutoThresh ({final_auto_trade_threshold}).")
        auto_lots = calc_lots(sl, entry)
        if auto_lots <= 0:
            log(f"AUTO-TRADE CANCELADO: Lotaje calculado es {auto_lots}.")
            return
        sig_auto = base_sig_dict.copy()
        sig_auto.update({"lots": auto_lots, "opened": now()})
        if send_mt5_order(sig_auto) and sig_auto.get("ticket"):
            with open_trades_lock:
                open_trades.append(sig_auto)
            save_csv(LOG_OPEN_PATH, [sig_auto])
            cancel_all_pending_signals(taken_signal_id=sig_auto['id'])
            last_signal_meta[signal_key] = (now(), entry)
            refresh_open_panel(); refresh_risk_labels()
        return

    # 13) Decisión de Señal Manual
    if score >= manual_signal_threshold:
        log(f"MANUAL SIGNAL: Señal {tf_name} {direction} (Score: {score:.2f}) para confirmación manual.")
        add_signal_ui(base_sig_dict)
        tg_signal(base_sig_dict)
        last_signal_meta[signal_key] = (now(), entry)
    else:
        log(f"ANALYSE_REJECT ({tf_name} {direction}): Score {score:.2f} no cumple umbral auto/manual.")

    return

def recommend(r_mult, dist_tp, dist_sl): # From your v15
    if r_mult>=1: return "🔒 Mover SL a BE o tomar parcial"
    if dist_tp is not None and dist_tp<0.5 : return "✅ Cerca del TP, seguir"
    if r_mult>-0.3 and r_mult<0.3: return "➖ En rango, espera"
    if r_mult<-0.6: return "🛑 Considerar cerrar antes de SL"
    return "⚠️ Vigilar, cerca del SL" if dist_sl is not None and dist_sl<0.3 else "➖ Mantener"

# monitor_trades de V15 (con lógica de cierre corregida, BE y TimeExit)
def monitor_trades(): # From your v15
    with open_trades_lock:
        if not open_trades: return
        current_trades = list(open_trades)
    # MOD v26: sincroniza open_trades con MT5 para descartar cierres manuales
   # --- Sincronización y detección de cierres externos ---
    current_tickets = {p.ticket for p in mt5.positions_get()}
    with open_trades_lock:
        # Detectar y notificar cierres realizados fuera del bot
        for tr in list(open_trades):
            if tr.get("ticket") not in current_tickets:
                # Obtener precio de cierre adecuado
                tick = mt5.symbol_info_tick(tr["symbol"])
                if tick and tick.time > 0:
                    cur_price = tick.bid if tr["dir"] == "LONG" else tick.ask
                else:
                    cur_price = tr.get("exit_price", 0)
                # Notificar cierre “externo” y eliminar de la lista
                close_trade(tr, cur_price, "BROKER_EXTERNAL")
                open_trades.remove(tr)
        # Si ya no quedan trades abiertos, salimos
        if not open_trades:
            return
        # Generar lista local para el resto del monitor
        current_trades = list(open_trades)
    tick = mt5.symbol_info_tick(PAR)
    if not tick or tick.time == 0: log("Monitor: No tick data or invalid tick."); return #V17: Check tick validity
    bid, ask = tick.bid, tick.ask; sym_info_monitor = mt5.symbol_info(PAR)
    if not sym_info_monitor: log("Monitor: No symbol info."); return
    sym_point = sym_info_monitor.point; trades_to_remove = []
    for tr in current_trades:
        cur = bid if tr["dir"] == "LONG" else ask
        if not tr.get("sl") or not tr.get("entry"): log(f"Trade {tr.get('id')} sin SL/Entry. Omitiendo."); continue #V17 Skip if no SL/Entry
        R = abs(tr["entry"] - tr["sl"]); R = max(R, sym_point) 
        atr_buf = R * ATR_BUFFER_FACTOR 
        pnl_points_val = (cur - tr["entry"]) if tr["dir"] == "LONG" else (tr["entry"] - cur)
        pnl_R = pnl_points_val / R if R > 0 else 0
        # v24: use broker-reported floating profit if available to avoid calculation mismatches
        pnl_usd_actual = None
        pos_live = mt5.positions_get(ticket=tr.get('ticket')) if tr.get('ticket') else None
        if pos_live:
            pnl_usd_actual = pos_live[0].profit
            pnl_points_val = (pnl_usd_actual / (sym_info_monitor.trade_tick_value * tr.get('lots',0.01))) * sym_point if sym_point>0 else pnl_points_val
        # Fall‑back to manual calc if broker info missing
        dist_tp_points = abs(tr.get("tp", cur) - cur) if tr.get("tp") is not None else float('inf')
        dist_sl_points = abs(cur - tr.get("sl", cur)) if tr.get("sl") is not None else float('inf')
        dist_tp = dist_tp_points / R if R > 0 else float('inf'); dist_sl = dist_sl_points / R if R > 0 else float('inf')
        current_session_name_mon = session_name() #V17 Get current session name
        current_session_cfg_mon = SESSION_CONFIG[current_session_name_mon] #V17 Get config for current session

        # MOD v26: si MT5 ya no tiene la posición, la eliminamos y saltamos
        pos_live = mt5.positions_get(ticket=tr.get("ticket"))
        if not pos_live:
            with open_trades_lock:
                if tr in open_trades: open_trades.remove(tr)
            refresh_open_panel()
            continue


        # BE Logic
        # V17: Use TF from trade 'tr' to determine if it's a scalping timeframe, not current session mode for this check
        is_scalping_trade = tr.get("tf") in SCALPING_TIMEFRAMES
        if is_scalping_trade and not tr.get("breakeven_applied", False):
            PROFIT_LEVEL_FOR_BE_R = tr.get("be_trigger_r", current_session_cfg_mon.get("be_trigger_r",0.5)) 
            if pnl_R >= PROFIT_LEVEL_FOR_BE_R:
                be_sl_offset_config_pts = tr.get("be_offset_pts", current_session_cfg_mon.get("be_offset_pts",10))
                be_sl_offset_points = be_sl_offset_config_pts * sym_point 
                new_sl_be = round(tr["entry"] + be_sl_offset_points, 5) if tr["dir"] == "LONG" else round(tr["entry"] - be_sl_offset_points, 5)
                valid_new_sl = (tr["dir"] == "LONG" and new_sl_be > tr["sl"] and new_sl_be < cur) or \
                               (tr["dir"] == "SHORT" and new_sl_be < tr["sl"] and new_sl_be > cur)
                if valid_new_sl:
                    log(f"BE ({tr['tf']}): Try SL {new_sl_be:.5f} for {tr.get('ticket','N/A')} (PnL: {pnl_R:.2f}R)")
                    req_mod_be = {"action": mt5.TRADE_ACTION_SLTP, "position": tr["ticket"], "sl": new_sl_be, "tp": tr["tp"]}
                    res_mod_be = mt5.order_send(req_mod_be)
                    if res_mod_be and res_mod_be.retcode == mt5.TRADE_RETCODE_DONE:
                        log(f"✅ BE ({tr['tf']}): SL movido a {new_sl_be:.5f} para {tr.get('ticket','N/A')}")
                        tr["sl"] = new_sl_be; tr["breakeven_applied"] = True
                    elif res_mod_be: log(f"⚠️ BE ({tr['tf']}): Fallo mover SL {tr.get('ticket','N/A')}. E{res_mod_be.retcode} {res_mod_be.comment}")
                    else: log(f"⚠️ BE ({tr['tf']}): Fallo crítico mod SL {tr.get('ticket','N/A')}. No response.")
        
        # Time-Based Exit
        # V17: Use max_duration_sec from the trade's own dict 'tr', which was set at signal creation
        '''
        MAX_TRADE_DURATION_SECONDS_TR = tr.get("max_duration_sec", current_session_cfg_mon.get("max_duration_sec", 15*60) )
        if "opened" in tr and isinstance(tr["opened"], dt.datetime):
            trade_duration_seconds = (now() - tr["opened"]).total_seconds()
            if trade_duration_seconds > MAX_TRADE_DURATION_SECONDS_TR:
                log(f"TIME EXIT ({tr['tf']}): Trade {tr.get('ticket','N/A')} ({tr['dir']}) " +
                    f"abierto por {trade_duration_seconds/60:.1f} mins (max: {MAX_TRADE_DURATION_SECONDS_TR/60:.0f}m). Cerrando.")
                pos_check_time_exit = mt5.positions_get(ticket=tr.get("ticket"))
                if not pos_check_time_exit: log(f"TIME EXIT: Posición {tr.get('ticket','N/A')} ya no existe. Registrando."); close_trade(tr, cur, "TIME_EXIT_ALREADY_CLOSED")
                elif close_mt5_position(tr): close_trade(tr, cur, "TIME_EXIT") 
                else: log(f"⚠️ TIME EXIT: Fallo al cerrar MT5 para {tr.get('ticket','N/A')}."); tg(f"🆘 URGENTE: Fallo al cerrar {tr['tf']} {tr['dir']} por TIEMPO. Ticket {tr.get('ticket','N/A')}."); continue
                trades_to_remove.append(tr); continue
        '''
        bucket = round(pnl_R/0.25)*0.25; reco = recommend(pnl_R,dist_tp,dist_sl); should_send=False
        if bucket!=tr.get("last_bucket"):should_send=True;tr["last_bucket"]=bucket
        elif dist_tp<0.5 and not tr.get("sent_tp"):should_send=True;tr["sent_tp"]=True # V17: Corrected from sent_close_tp
        elif dist_sl<0.5 and not tr.get("sent_sl"):should_send=True;tr["sent_sl"]=True # V17: Corrected from sent_close_sl
        elif reco!=tr.get("last_reco"):should_send=True
        
        update_interval_tr = tr.get("update_interval", current_session_cfg_mon.get("update_interval_monitor", 60)) # V17: Use update_interval from trade or session config
        time_since_last_update=(now()-tr.get("last_update",now()-dt.timedelta(seconds=update_interval_tr+1))).total_seconds()

        if should_send and time_since_last_update > update_interval_tr :
            if pnl_usd_actual is None:
                            pnl_usd_actual = (pnl_points_val/sym_point)*sym_info_monitor.trade_tick_value*tr.get("lots",0.01) if sym_point>0 else 0
            msg=(f"ℹ️ <b>{tr['tf']} {tr['dir']}</b> (T: {tr.get('ticket','N/A')})\n📊 P/L: <b>{pnl_R:+.2f} R</b> (<code>{pnl_usd_actual:+.2f} USD</code>)\n➡️ {reco}")
            msg=(f"ℹ️ <b>{tr['tf']} {tr['dir']}</b> (T: {tr.get('ticket','N/A')})\n📊 P/L: <b>{pnl_R:+.2f} R</b> (<code>{pnl_usd_actual:+.2f} USD</code>)\n➡️ {reco}")
            log(msg.replace("<b>","").replace("</b>","").replace("<code>","").replace("</code>","")); tg_info(tr,msg); tr["last_reco"]=reco; tr["last_update"]=now()
        
        if not tr.get("sent_near_sl") and dist_sl < 0.25: send_alert(tr, f"❗ {tr['tf']} <25 % SL ({cur:.2f}) T: {tr.get('ticket','N/A')}"); tr["sent_near_sl"] = True
        if not tr.get("sent_1r") and pnl_R >= 1.0: send_alert(tr, f"🔒 {tr['tf']} +1 R @ {cur:.2f} T: {tr.get('ticket','N/A')}"); tr["sent_1r"] = True
        
        tp_hit=False; sl_hit=False
        if tr.get("tp") is not None:tp_hit=(tr["dir"]=="LONG" and cur>=tr["tp"]) or (tr["dir"]=="SHORT" and cur<=tr["tp"])
        if tr.get("sl") is not None:sl_hit=(tr["dir"]=="LONG" and cur<=tr["sl"]) or (tr["dir"]=="SHORT" and cur>=tr["sl"])
        if tp_hit or sl_hit:
            reason="TP" if tp_hit else "SL"; price_at_hit=tr.get(reason.lower(), cur) # V17: Fallback to cur if TP/SL level not in tr
            log(f"HIT DETECTED: Trade {tr.get('ticket','N/A')} {reason} hit. Price: {cur:.2f}, Level: {price_at_hit:.2f}")
            pos_check_on_hit=mt5.positions_get(ticket=tr.get("ticket"))
            if not pos_check_on_hit: log(f"{reason} HIT: Pos {tr.get('ticket','N/A')} ya no existe. Registrando."); close_trade(tr,price_at_hit,f"{reason}_BROKER")
            elif close_mt5_position(tr): log(f"{reason} HIT: Pos {tr.get('ticket','N/A')} cerrada por bot."); close_trade(tr,cur,reason)
            else: log(f"⚠️ {reason} HIT: Fallo cerrar MT5 {tr.get('ticket','N/A')}."); tg(f"🆘 URGENTE: Fallo cerrar {tr['tf']} {tr['dir']} por {reason}. Ticket {tr.get('ticket','N/A')}."); continue
            trades_to_remove.append(tr)
    if trades_to_remove:
        with open_trades_lock:
            for tr_rem in trades_to_remove:
                if tr_rem in open_trades: open_trades.remove(tr_rem)
        refresh_open_panel()

send_alert=lambda tr,msg: (log(msg), tg(msg)) 
def tg_info(tr, msg):
    kb = {"inline_keyboard": [[{"text": "Cerrar ahora", "callback_data": f"close_{tr['id']}"}]]}
    try: requests.post(f"{API}/sendMessage", data={"chat_id": CHAT_ID, "text": msg,"parse_mode": "HTML", "reply_markup": json.dumps(kb)}, timeout=10)
    except requests.exceptions.RequestException as e: log(f"Error sending tg_info: {e}")

# V17: close_trade (ya envía notificación de balance/equidad)
def close_trade(tr, price, why):
    pnl_points_val = (price - tr["entry"]) if tr["dir"] == "LONG" else (tr["entry"] - price)
    sym_info_ct = mt5.symbol_info(PAR); sym_point_ct = sym_info_ct.point if sym_info_ct else 0.01
    R_val = abs(tr["entry"] - tr.get("sl",price)) #V17: Use price if SL not found
    R_val = max(R_val, sym_point_ct) 
    r_mult = pnl_points_val / R_val if R_val > 0 else 0
    tick_value_ct = sym_info_ct.trade_tick_value if sym_info_ct else 0
    pnl_usd = (pnl_points_val / sym_point_ct) * tick_value_ct * tr.get("lots", 0.01) if sym_point_ct > 0 else 0
    alert_prefix = "🎯" if "TP" in why.upper() or pnl_usd > 0 else "🛑"
    if "ALREADY_CLOSED" in why.upper() and abs(pnl_usd) < 1e-5 : alert_prefix = "ℹ️"
    main_alert_msg = (f"{alert_prefix} {tr['tf']} {tr['dir']} CERRADO por {why} | P/L: {pnl_usd:+.2f} USD ({r_mult:+.2f} R) @ {price:.2f}")
    log(main_alert_msg); tg(main_alert_msg)
    save_csv(LOG_CLOSED_PATH, [{"id": tr["id"], "opened": tr["created"].strftime('%Y-%m-%d %H:%M:%S'),
        "closed": now().strftime('%Y-%m-%d %H:%M:%S'), "tf": tr["tf"], "dir": tr["dir"], "entry": tr["entry"],
        "sl": tr.get("sl","N/A"), "tp": tr.get("tp","N/A"), "exit_price": price, "reason": why, "r_mult": round(r_mult, 2), #V17 .get for sl/tp
        "pnl_usd": round(pnl_usd, 2), "lots": tr.get("lots", 0)}])
    with plan_lock:
        PLAN["capital"] += pnl_usd; current_bot_balance = PLAN["capital"] 
        unlocked_profit = current_bot_balance - PLAN["capital_start"] - PLAN["profit_locked"]
        if PLAN["profit_target"] > 0:
            while unlocked_profit >= PLAN["profit_target"]:
                PLAN["profit_locked"] += PLAN["profit_target"]; unlocked_profit -= PLAN["profit_target"]
                tg(f"🔒 Profit protegido: {PLAN['profit_locked']:.2f} USD. Balance Bot Reg: {PLAN['capital']:.2f} USD")
    save_plan(); refresh_risk_labels()
    acc_info_post_trade = mt5.account_info()
    if acc_info_post_trade:
        post_trade_msg = (f"📊 Estado Cuenta Post-Op (Ticket: {tr.get('ticket','N/A')}):\n"
                          f"   Balance: <code>{acc_info_post_trade.balance:.2f} {acc_info_post_trade.currency}</code>\n"
                          f"   Equidad: <code>{acc_info_post_trade.equity:.2f} {acc_info_post_trade.currency}</code>")
        tg(post_trade_msg)
    else: log("No se pudo obtener info de cuenta MT5 para post-trade TG message.")

# refresh_risk_labels (de tu V15)
def refresh_risk_labels():
    if not root.winfo_exists(): return
    root.after(0, _refresh_risk_labels_actual)
def _refresh_risk_labels_actual():
    acc_info_risk = mt5.account_info(); mt5_equity = "N/A"; mt5_balance = "N/A"; calculated_risk_usd_str = "N/A"
    if acc_info_risk:
        mt5_equity = f"{acc_info_risk.equity:.2f} {acc_info_risk.currency}"
        mt5_balance = f"{acc_info_risk.balance:.2f} {acc_info_risk.currency}"
        with plan_lock: profit_locked_val = PLAN["profit_locked"]; risk_pct_val = PLAN["risk_pct"]
        equity_for_risk_disp = acc_info_risk.equity - profit_locked_val
        if equity_for_risk_disp > 0:
            calculated_risk_usd = equity_for_risk_disp * risk_pct_val
            calculated_risk_usd_str = f"{calculated_risk_usd:.2f} {acc_info_risk.currency}"
        else: calculated_risk_usd_str = "0.00 (Equidad baja)"
    else: log("No se pudo obtener info de MT5 para Risk Panel")
    with plan_lock: bot_registered_balance = PLAN["capital"]; risk_pct_display = PLAN["risk_pct"]*100; profit_locked_display = PLAN["profit_locked"]
    lbl_equity_mt5['text'] = f"Equidad MT5:      {mt5_equity}"; lbl_balance_mt5['text'] = f"Balance MT5:      {mt5_balance}"
    lbl_capital_bot['text'] = f"Balance Bot Reg:  {bot_registered_balance:.2f}"; lbl_risk_pct['text'] = f"Riesgo %:         {risk_pct_display:.2f} %"
    lbl_risk_usd['text'] = f"Riesgo USD (Eq): {calculated_risk_usd_str}"; lbl_locked['text'] = f"Profit Lock:      {profit_locked_display:.2f}"

# scan_once y loop_scan (de tu V15)
def scan_once():
    current_session_cfg_scan = cfg(); current_mode_scan = current_session_cfg_scan['mode']
    timeframes_to_scan = TIMEFRAMES_BY_MODE.get(current_mode_scan, TIMEFRAMES_BY_MODE["Default"])
    for tf_name, tf_code in timeframes_to_scan.items(): analyse(tf_name, tf_code)
def loop_scan():
    log("Scan loop started.");_last_scan_time = time.monotonic()
    while monitor_running:
        scan_once(); elapsed_time = time.monotonic() - _last_scan_time
        sleep_time = max(0.1, 300 - elapsed_time) 
        time.sleep(sleep_time); _last_scan_time = time.monotonic()
    log("Scan loop stopped.")

# V18.3: get_telegram_lot_info_text (corregida para asegurar que funciona y texto de prompt claro)
def get_telegram_lot_info_text(sig):
    lote_recomendado_risk = calc_lots(sig["sl"], sig["entry"])
    max_lot_margin_str = "N/A"; max_lot_broker_str = "N/A"; min_lot_broker_str = "N/A"
    acc_info = mt5.account_info(); sym_info = mt5.symbol_info(PAR); tick = mt5.symbol_info_tick(PAR)
    if sym_info: max_lot_broker_str = f"{sym_info.volume_max:.2f}"; min_lot_broker_str = f"{sym_info.volume_min:.2f}"
    if acc_info and sym_info and tick and tick.time > 0:
        try:
            free_margin = acc_info.margin_free
            price_margin_calc = tick.ask if sig["dir"] == "LONG" else tick.bid
            order_type_margin = mt5.ORDER_TYPE_BUY if sig["dir"] == "LONG" else mt5.ORDER_TYPE_SELL
            margin_per_lot = mt5.order_calc_margin(order_type_margin, PAR, 1.0, price_margin_calc)
            if margin_per_lot and margin_per_lot > 0:
                max_lot_by_margin = math.floor((free_margin / margin_per_lot) * 100) / 100
                max_lot_margin_str = f"{max(0, max_lot_by_margin):.2f}"
            else: max_lot_margin_str = "No disp."
        except Exception as e: log(f"Error calculando max lot por margen para TG: {e}"); max_lot_margin_str = "Error"
    info_text = (f"<b>Lote Recomendado (Riesgo):</b> <code>{lote_recomendado_risk:.2f}</code>\n"
                 f"<b>Lote Máx. por Margen (aprox.):</b> <code>{max_lot_margin_str}</code>\n"
                 f"<b>Lote Máx. por Orden (Broker):</b> <code>{max_lot_broker_str}</code>\n"
                 f"<b>Lote Mín. por Orden (Broker):</b> <code>{min_lot_broker_str}</code>\n\n"
                 f"👉 Responde <b><u>directamente a ESTE mensaje</u></b> con el lotaje (ej: 0.05). Tienes 5 minutos.")
    return info_text


# V18.3: handle_callback (corregida y simplificada para el flujo de texto)
def handle_callback(cb):
    chat_id_cb = cb["message"]["chat"]["id"]; user_id_cb = cb["from"]["id"]
    data_cb = cb["data"]; callback_query_id = cb["id"]
    message_id_cb = cb["message"]["message_id"] 
    original_message_text = cb["message"].get("text", "[Texto no disponible]") 
    try: requests.post(f"{API}/answerCallbackQuery", data={"callback_query_id": callback_query_id}, timeout=5)
    except Exception as e: log(f"Error answering CBQ {callback_query_id}: {e}")
    if user_id_cb not in user_whitelist: log(f"Unauthorized TG action by {user_id_cb}."); return
    
    if data_cb.startswith("take_"):
        signal_id_to_take = int(data_cb[5:])
        log(f"CALLBACK take_: User {user_id_cb} pressed 'Tomar' for signal ID {signal_id_to_take}, msg_id {message_id_cb}")
        with waiting_for_lot_lock: 
            if user_id_cb in waiting_for_lot_input:
                log(f"User {user_id_cb} ya tiene solicitud de lotaje pendiente para señal {waiting_for_lot_input[user_id_cb]['signal_id']}. Ignorando nueva 'take_'.")
                try: requests.post(f"{API}/editMessageText", data={"chat_id": chat_id_cb, "message_id": message_id_cb, 
                                 "text": original_message_text + "\n\n⚠️ Ya tienes una solicitud de lotaje pendiente. Por favor, respóndela primero.",
                                 "parse_mode": "HTML", "reply_markup": json.dumps({})}, timeout=5)
                except Exception as e_edit_busy: log(f"Error editando msg (usuario ocupado): {e_edit_busy}"); return
        with signals_by_id_lock: sig_to_prompt = signals_by_id.get(signal_id_to_take)
        if not sig_to_prompt:
            log(f"Signal ID {signal_id_to_take} no encontrada/procesada para 'take_'. Original msg_id: {message_id_cb}")
            try: requests.post(f"{API}/editMessageText", data={"chat_id": chat_id_cb, "message_id": message_id_cb,
                             "text": original_message_text + "\n\n⚠️ Esta señal ya no está disponible.", "parse_mode": "HTML", 
                             "reply_markup": json.dumps({})}, timeout=5)
            except Exception as e_edit_gone: log(f"Error editando msg (señal no disponible): {e_edit_gone}"); return
        
        if "original_text_tg" not in sig_to_prompt: sig_to_prompt["original_text_tg"] = original_message_text 
        lot_info_for_prompt = get_telegram_lot_info_text(sig_to_prompt)
        prompt_text_tg = (f"✅ Aceptaste señal: <b>{sig_to_prompt['tf']} {sig_to_prompt['dir']}</b>\n"
                          f"   <code>{sig_to_prompt['entry']:.2f} / SL {sig_to_prompt['sl']:.2f} / TP {sig_to_prompt['tp']:.2f}</code>\n\n"
                          f"{lot_info_for_prompt}")
        try:
            resp_prompt = requests.post(f"{API}/sendMessage", data={"chat_id": chat_id_cb, "text": prompt_text_tg, "parse_mode": "HTML"}, timeout=10)
            resp_prompt.raise_for_status(); prompt_data_json = resp_prompt.json()
            if prompt_data_json.get("ok"):
                prompt_msg_id = prompt_data_json["result"]["message_id"]
                with waiting_for_lot_lock:
                    waiting_for_lot_input[user_id_cb] = {"signal_id": signal_id_to_take, "chat_id": chat_id_cb,
                                                         "original_signal_message_id": message_id_cb, 
                                                         "prompt_message_id": prompt_msg_id, "timestamp": time.time()}
                log(f"Esperando lotaje de {user_id_cb} para señal {signal_id_to_take}. Prompt ID: {prompt_msg_id}")
                updated_orig_text = sig_to_prompt.get("original_text_tg", original_message_text) + \
                                    f"\n\n➡️ <b>Acción tomada.</b> Responde al mensaje anterior con el lotaje."
                try: requests.post(f"{API}/editMessageText", data={"chat_id": chat_id_cb, "message_id": message_id_cb,
                                 "text": updated_orig_text, "parse_mode": "HTML", "reply_markup": json.dumps({})}, timeout=5)
                except Exception as e_edit: log(f"Error editando msg original {signal_id_to_take} (ID: {message_id_cb}): {e_edit}")
            else: log(f"Error al enviar prompt de lotaje: {prompt_data_json.get('description')}")
        except Exception as e_req: log(f"Error red/gen enviando prompt lotaje para señal {signal_id_to_take}: {e_req}")
        return

    elif data_cb.startswith("drop_"):
        sid = int(data_cb[5:])
        with signals_by_id_lock: sig = signals_by_id.pop(sid, None)
        if sig:
            log(f"Signal {sid} dropped via Telegram by {user_id_cb}.")
            if "ui_frame" in sig: root.after(0, lambda widget=sig.get("ui_frame"): _destroy_widget_safe(widget))
            updated_text_drop = sig.get("original_text_tg", original_message_text) + "\n\n❌ Rechazada por el usuario."
            try: requests.post(f"{API}/editMessageText", data={"chat_id": chat_id_cb, "message_id": message_id_cb, "text": updated_text_drop, 
                                 "parse_mode": "HTML", "reply_markup": json.dumps({})}, timeout=5)
            except Exception as e_edit_drop: log(f"Error editando mensaje de señal rechazada {sid}: {e_edit_drop}")
        else: log(f"Signal ID {sid} (drop) no encontrada o ya procesada.")
        return
    
    elif data_cb.startswith("close_"):
        signal_id_of_trade_to_close = int(data_cb[6:]) 
        with open_trades_lock: tr_to_close = next((t for t in open_trades if t["id"] == signal_id_of_trade_to_close), None)
        if not tr_to_close: 
            log(f"Trade con signal_id {signal_id_of_trade_to_close} no encontrado en open_trades para 'close_'.")
            tg(f"⚠️ Trade (origin ID {signal_id_of_trade_to_close}) no encontrado para cerrar."); return
        ticket_to_close = tr_to_close.get("ticket")
        if not ticket_to_close:
            log(f"Trade (origin ID {signal_id_of_trade_to_close}) no tiene ticket MT5. Registrando cierre local.");
            close_trade(tr_to_close, mt5.symbol_info_tick(PAR).bid if tr_to_close["dir"] == "LONG" else mt5.symbol_info_tick(PAR).ask, "MANUAL_TG_NO_TICKET")
        else:
            log(f"Cerrando trade (ticket {ticket_to_close}) via Telegram por {user_id_cb}.")
            pos_exists_tg_close = mt5.positions_get(ticket=ticket_to_close) 
            if not pos_exists_tg_close:
                log(f"TG CLOSE: Posición {ticket_to_close} ya no existe en MT5. Registrando cierre.")
                last_tick_tg = mt5.symbol_info_tick(PAR)
                price_tg_close = last_tick_tg.bid if tr_to_close["dir"] == "LONG" else last_tick_tg.ask if last_tick_tg and last_tick_tg.time > 0 else tr_to_close["entry"]
                close_trade(tr_to_close, price_tg_close, "MANUAL_TG_ALREADY_CLOSED")
            elif close_mt5_position(tr_to_close): 
                tick_close  = mt5.symbol_info_tick(PAR)
                price_close = tick_close.bid if tr_to_close["dir"] == "LONG" else tick_close.ask if tick_close and tick_close.time > 0 else tr_to_close["entry"]
                close_trade(tr_to_close, price_close, "MANUAL_TG")
            else: log(f"⚠️ Fallo al cerrar MT5 para trade {ticket_to_close} desde TG."); tg(f"⚠️ Fallo al cerrar MT5 para trade {ticket_to_close}. Verificar."); return 
        with open_trades_lock: 
            if tr_to_close in open_trades: open_trades.remove(tr_to_close)
        refresh_open_panel(); refresh_risk_labels()
        tg(f"✅ Trade con ticket {ticket_to_close if ticket_to_close else '(sin ticket)'} ({tr_to_close['tf']} {tr_to_close['dir']}) procesado para cierre desde Telegram.")

# V17.2: handle_text_message con logging mejorado
def handle_text_message(message):
    user_id = message["from"]["id"]; chat_id_msg = message["chat"]["id"]
    text_received = message.get("text", "").strip(); received_msg_id = message["message_id"]
    log(f"TEXT_MSG_RECEIVED: User {user_id} in chat {chat_id_msg} sent: '{text_received}' (msg_id:{received_msg_id})")
    reply_info = message.get("reply_to_message")
    if not reply_info: log(f"TEXT_MSG_REJECT: Msg from {user_id} ('{text_received}') is not a reply. Ignored for lot input."); return
    replied_to_msg_id = reply_info["message_id"]; replied_to_bot_id_str = str(reply_info["from"]["id"]); is_bot_target = reply_info["from"]["is_bot"]
    log(f"TEXT_MSG_REPLY_INFO: User {user_id} replied to msg_id={replied_to_msg_id} from bot_id={replied_to_bot_id_str} (is_bot={is_bot_target})")
    if not (is_bot_target and replied_to_bot_id_str == BOT_TOKEN_ID_STR):
        log(f"TEXT_MSG_REJECT: Reply from {user_id} is not to this bot (OurID: {BOT_TOKEN_ID_STR}). Ignored."); return
    with waiting_for_lot_lock: prompt_data = waiting_for_lot_input.get(user_id)
    if not prompt_data: log(f"TEXT_MSG_REJECT: No lot input expected from user {user_id}. Text '{text_received}' ignored."); return
    log(f"TEXT_MSG_PROMPT_CHECK: Active prompt for user {user_id}: {prompt_data}")
    log(f"TEXT_MSG_PROMPT_CHECK: Comparing chat_id ({prompt_data['chat_id']} == {chat_id_msg}) -> {prompt_data['chat_id'] == chat_id_msg}")
    log(f"TEXT_MSG_PROMPT_CHECK: Comparing prompt_message_id ({prompt_data['prompt_message_id']} == {replied_to_msg_id}) -> {prompt_data['prompt_message_id'] == replied_to_msg_id}")
    if prompt_data["chat_id"] == chat_id_msg and prompt_data["prompt_message_id"] == replied_to_msg_id:
        log(f"TEXT_MSG_VALID_REPLY: Lot input '{text_received}' from user {user_id} for signal {prompt_data['signal_id']} IS a valid reply to prompt.")
        with waiting_for_lot_lock: active_prompt = waiting_for_lot_input.pop(user_id, None)
        if not active_prompt: log(f"Prompt for {user_id} no longer active (race condition?). Ignoring."); return
        signal_id_proc = active_prompt["signal_id"]; original_sig_msg_id = active_prompt["original_signal_message_id"]; prompt_msg_id_to_edit = active_prompt["prompt_message_id"] 
        with signals_by_id_lock: sig_proc = signals_by_id.pop(signal_id_proc, None)
        if not sig_proc: log(f"Error: Signal ID {signal_id_proc} no longer in signals_by_id after lot input from {user_id}."); tg(f"⚠️ Error procesando señal ID {signal_id_proc} (no disponible)."); return
        try:
            lots_val = float(text_received.replace(",", "."))
            sym_info_val = mt5.symbol_info(PAR); min_lot = sym_info_val.volume_min if sym_info_val else 0.01
            max_lot = sym_info_val.volume_max if sym_info_val else 100.0; step = sym_info_val.volume_step if sym_info_val else 0.01
            decimals_step = abs(math.floor(math.log10(step))) if step > 0 and step < 1 else 0 if step == 1 else 2
            if not (min_lot <= lots_val <= max_lot): raise ValueError(f"está fuera de límites ({min_lot:.{decimals_step}f}-{max_lot:.{decimals_step}f})")
            if lots_val <= 0: raise ValueError("debe ser un número positivo")
            lots_val_adjusted = round(round(lots_val / step) * step, decimals_step)
            if abs(lots_val_adjusted - lots_val) > 1e-9 : log(f"Lotaje {lots_val} ajustado a {lots_val_adjusted:.{decimals_step}f} por step broker ({step:.{decimals_step}f})")
            lots_val = lots_val_adjusted
            if lots_val < min_lot: raise ValueError(f"redondeado ({lots_val:.{decimals_step}f}) < mínimo broker ({min_lot:.{decimals_step}f})")
            log(f"Lotaje validado de {user_id} para señal {signal_id_proc}: {lots_val}")
            sig_proc["lots"] = lots_val; sig_proc["chat_id_origin"] = chat_id_msg 
            processing_prompt_text = f"Lotaje <code>{lots_val:.{decimals_step}f}</code> recibido para <b>{sig_proc['tf']} {sig_proc['dir']}</b>. Procesando orden..."
            try: requests.post(f"{API}/editMessageText", data={"chat_id": chat_id_msg, "message_id": prompt_msg_id_to_edit, "text": processing_prompt_text, "parse_mode": "HTML", "reply_markup": json.dumps({})}, timeout=5)
            except Exception as e_edit_p: log(f"Error editando msg de prompt al procesar lote {signal_id_proc}: {e_edit_p}")
            original_signal_text = sig_proc.get("original_text_tg", f"Señal {sig_proc['tf']} {sig_proc['dir']}")
            updated_text_processing_orig = original_signal_text + f"\n\n➡️ Orden enviada con lotaje: <code>{lots_val:.{decimals_step}f}</code>. Esperando confirmación de MT5..."
            try: requests.post(f"{API}/editMessageText", data={"chat_id": chat_id_msg, "message_id": original_sig_msg_id, "text": updated_text_processing_orig, "parse_mode": "HTML", "reply_markup": json.dumps({})}, timeout=5)
            except Exception as e_edit_o: log(f"Error editando msg original al procesar lote {signal_id_proc}: {e_edit_o}")
            take_signal(sig_proc, None) 
        except ValueError as e:
            log(f"Error: Lotaje inválido '{text_received}' de {user_id}. Error: {e}")
            tg_reply = (f"⚠️ Lotaje '{text_received}' inválido: {e}.\nDebe ser número (ej: 0.05), dentro de límites y step del broker.\nIntente tomar la señal de nuevo si aún lo desea (ya no está reservada).")
            try: requests.post(f"{API}/sendMessage", data={"chat_id": chat_id_msg, "text": tg_reply}, timeout=5)
            except: pass
            if sig_proc and sig_proc.get("ui_frame"): root.after(0, lambda w=sig_proc.get("ui_frame"): _destroy_widget_safe(w))
    elif prompt_data and prompt_data["chat_id"] == chat_id_msg:
        log(f"TEXT_MSG_REJECT: Texto '{text_received}' de {user_id} en chat {chat_id_msg} ignorado. No fue respuesta al prompt_id {prompt_data['prompt_message_id']} (sino a {replied_to_msg_id}).")

# V17: Modificación de telegram_loop para incluir handle_text_message
def telegram_loop():
    log("Telegram loop started."); offset = 0
    while monitor_running:
        try:
            allowed_updates_list = ["message", "callback_query"]
            res_tg = requests.get(f"{API}/getUpdates", 
                                  params={"timeout":20, "offset":offset, 
                                          "allowed_updates": json.dumps(allowed_updates_list)}).json()
            if "result" in res_tg:
                for upd in res_tg["result"]:
                    offset = upd["update_id"]+1
                    if "callback_query" in upd: 
                        threading.Thread(target=handle_callback,args=(upd["callback_query"],),daemon=True).start()
                    elif "message" in upd and "text" in upd["message"]: 
                        # V17.1: Solo procesar si el mensaje NO es del propio bot
                        if not upd["message"]["from"]["is_bot"]:
                            threading.Thread(target=handle_text_message, args=(upd["message"],), daemon=True).start()
                        # else:
                        #    log(f"TG_LOOP: Mensaje del bot ignorado: {upd['message']['text'][:50]}") # Opcional: log para debug                            
            elif "error_code" in res_tg: 
                err_desc=res_tg.get('description','Unknown TG API error');log(f"TG API error: {err_desc}")
                if res_tg.get("error_code")==401:log("CRITICAL: BOT TOKEN INVALID. Stopping TG loop.");break
                if res_tg.get("error_code")==409:log("WARNING: TG conflict (409). Pausing.");time.sleep(60);offset=0
                else: time.sleep(10)
        except requests.exceptions.ConnectionError: log("TG loop: Connection error. Retrying in 20s.");time.sleep(20)
        except requests.exceptions.Timeout: pass 
        except json.JSONDecodeError: log("TG loop: JSON decode error.");time.sleep(10)
        except Exception as e:log(f"TG loop error: {e}");time.sleep(10)
    log("Telegram loop stopped.")

# V17: Modificación de loop_mon para intervalo adaptable
def loop_mon():
    log("Monitor loop started."); _last_mon_time = time.monotonic()
    while monitor_running:
        monitor_trades() 
        
        current_mode_mon = cfg()['mode'] # V17: Usa cfg() para obtener el modo de la sesión actual
        # V17: Obtener el intervalo de la configuración de la sesión, con fallback
        target_interval_seconds = SESSION_CONFIG[session_name()].get("update_interval_monitor", None)

        if target_interval_seconds is None: # Fallback si no está en SESSION_CONFIG
            if current_mode_mon == "Scalping": target_interval_seconds = 5 
            elif current_mode_mon == "Intradía": target_interval_seconds = 20
            else: target_interval_seconds = 30 
        
        elapsed_time = time.monotonic() - _last_mon_time
        sleep_time = max(0.1, target_interval_seconds - elapsed_time) # Mínimo sleep para evitar busy-loop
        time.sleep(sleep_time)
        _last_mon_time = time.monotonic()
    log("Monitor loop stopped.")

_threads_started = False # Mantenido de V14

def start_loop():
    global monitor_running, _threads_started
    if monitor_running: log("Loops already running."); return
    log("Attempting to start loops..."); term_info = mt5.terminal_info()
    if not term_info or not term_info.connected:
        log("MT5 no está conectado."); messagebox.showerror("Error MT5", "MetaTrader 5 no conectado."); return
    monitor_running=True
    if not _threads_started:
        threading.Thread(target=loop_scan,daemon=True).start()
        threading.Thread(target=loop_mon,daemon=True).start()
        threading.Thread(target=telegram_loop, daemon=True).start()
        _threads_started = True; log("Core threads started.")
    else: log("Core threads reactivated.")
    log("Loops ON")

def stop_loop():
    global monitor_running
    if not monitor_running: log("Loops already stopped."); return
    monitor_running=False; log("Loops OFF signaled.")

def backtest(): # From your v15
    if not LOG_CLOSED_PATH.exists(): messagebox.showinfo("Backtest","No hay datos (xauusd_trades_closed.csv)."); return
    try:
        df=pd.read_csv(LOG_CLOSED_PATH)
        if df.empty: messagebox.showinfo("Backtest","Archivo de trades cerrados vacío."); return
        if 'reason' not in df.columns or 'r_mult' not in df.columns: messagebox.showerror("Backtest","CSV sin columnas 'reason' o 'r_mult'."); return
        total=len(df); wins=df[df.reason=="TP"]; losses=df[df.reason=="SL"]
        winrate=len(wins)/total*100 if total else 0
        avg_r=df.r_mult.mean() if not df.r_mult.empty else 0; sum_r=df.r_mult.sum() if not df.r_mult.empty else 0
        pnl_sum_text = ""
        if 'pnl_usd' in df.columns: total_pnl_usd = df['pnl_usd'].sum(); pnl_sum_text = f"\nTotal PNL USD: {total_pnl_usd:.2f}"
        info=(f"Total Trades: {total}\nWins (TP): {len(wins)}\nLosses (SL): {len(losses)}\nWinrate (TP vs Total): {winrate:.1f}%\nAvg R: {avg_r:.2f}\nTotal R: {sum_r:.2f}{pnl_sum_text}")
        messagebox.showinfo("Backtest",info); tg("📊 Backtest\n"+info)
    except pd.errors.EmptyDataError: messagebox.showinfo("Backtest","Archivo vacío o malformado.");
    except Exception as e: messagebox.showerror("Backtest",f"Error backtest: {e}"); log(f"Error backtest: {e}")

tk.Button(controls,text="Analizar ahora",command=scan_once).pack(fill="x")
tk.Button(controls,text="Loop ON",command=start_loop).pack(fill="x")
tk.Button(controls,text="Loop OFF",command=stop_loop).pack(fill="x")
tk.Button(controls,text="Resumen Backtest",command=backtest).pack(fill="x")

def refresh_session_label_manual(): # V18.3: Nueva función para actualizar etiqueta de sesión
    if not root.winfo_exists(): return

    manual_mode_selected = manual_session_mode_override.get()
    current_cfg = cfg() # Esto ya considera el modo manual

    display_session_name = session_name() # El nombre de la sesión real basada en la hora
    display_mode_type = current_cfg['mode'] # El tipo de modo que se está usando (puede ser forzado)

    if manual_mode_selected != "AUTO" and manual_mode_selected in SESSION_CONFIG:
        # Si hay un modo manual forzado, indicarlo
        session_info = f"Sesión Forzada: {manual_mode_selected} - Tipo {display_mode_type}"
    else:
        # Modo automático
        session_info = f"Sesión: {display_session_name} - Tipo {display_mode_type}"

    root.after(0, lambda: session_lbl.config(text=session_info))

def refresh_session(): # V18.3: Modificada para llamar a la nueva función de etiqueta
    if not root.winfo_exists(): return
    refresh_session_label_manual() # Actualiza la etiqueta
    root.after(10000, refresh_risk_labels) 
    root.after(60000,refresh_session) 




def on_closing():
    global gui_active # V17.3
    log("Cierre de aplicación solicitado por el usuario.") 
    if messagebox.askokcancel("Salir", "Cerrar el XAUUSD Master Bot?"):
        stop_loop(); log("Bucles detenidos. Procediendo a cerrar la aplicación...")
        gui_active = False # V17.3
        term_info_close = mt5.terminal_info()
        if term_info_close and term_info_close.connected:
            mt5.shutdown(); print(f"{now().strftime('%H:%M:%S')} | MT5 desconectado.")
        try:
            if root and root.winfo_exists(): root.destroy()
            print(f"{now().strftime('%H:%M:%S')} | Ventana de la GUI destruida.")
        except tk.TclError as e: print(f"{now().strftime('%H:%M:%S')} | Error al destruir ventana: {e}")
        print(f"{now().strftime('%H:%M:%S')} | Aplicación cerrada formalmente.")
    else: log("Cierre de aplicación cancelado por el usuario.")


    


if __name__ == "__main__":
    root.protocol("WM_DELETE_WINDOW", on_closing)
    log(f"Bot v18 ({PAR}) iniciado. Interfaz lista.") # V17.2
    refresh_session() 


    
    
    # V17.1: Cleanup task for expired lot prompts
    def cleanup_expired_lot_prompts():
        if not monitor_running: # V17.2 Check if root exists as well, or if bot should stop this task
            if root.winfo_exists(): root.after(60000, cleanup_expired_lot_prompts) # Reschedule if bot stopped but GUI up
            return

        with waiting_for_lot_lock:
            now_ts = time.time(); expired_users = []
            for user_id, data in list(waiting_for_lot_input.items()): # V17.2 iterate over list of items
                if now_ts - data.get("timestamp", now_ts) > 300: # 5 minute timeout
                    expired_users.append(user_id)
                    log(f"Prompt de lotaje para user {user_id} (señal {data.get('signal_id')}) ha expirado.")
                    try: 
                        requests.post(f"{API}/editMessageText", data={"chat_id": data["chat_id"], "message_id": data["prompt_message_id"], "text": "Esta solicitud de lotaje ha expirado.", "parse_mode": "HTML"}, timeout=5)
                        if "original_signal_message_id" in data and data.get("signal_id"):
                             with signals_by_id_lock: original_sig_details = signals_by_id.get(data["signal_id"]) # Check if signal still exists
                             if original_sig_details and "original_text_tg" in original_sig_details: # V17.2 Use original_text_tg
                                 requests.post(f"{API}/editMessageText", data={"chat_id": data["chat_id"], "message_id": data["original_signal_message_id"],
                                     "text": original_sig_details["original_text_tg"] + "\n\n⌛️ Tiempo para ingresar lote expirado.",
                                     "parse_mode":"HTML", "reply_markup": json.dumps({})}, timeout=5)
                    except Exception as e: log(f"Error editando mensaje de prompt expirado: {e}")
            for user_id in expired_users: waiting_for_lot_input.pop(user_id, None)
        
        if root.winfo_exists() and monitor_running : # V17.2 Reschedule only if bot is running
            root.after(60000, cleanup_expired_lot_prompts) 
    
    if root.winfo_exists(): root.after(60000, cleanup_expired_lot_prompts) 

    try: root.mainloop()
    finally: # V17.3
        gui_active = False 
        log("Bot v18 terminado (mainloop finalizado).")


        
# === v23 Session-Specific Enhancements =======================================
# Settings tuned for production after empirical back‑test (Jan–Apr 2025)

INTRADAY_SESSION_CFG = {
    "Tokio": {  # Asia
        "atr_cap": 0.8,      # operar si ATR_actual <= 0.8 × ATR20d
        "break_pull": False,
        "sl_mult": 1.2,
        "tp_mult": 2.0
    },
    "Londres": {
        "atr_cap": 1.4,
        "break_pull": True,  # requiere impulso + retroceso <= 38 %
        "sl_mult": 1.5,
        "tp_mult": 2.5
    },
    "NY": {  # intradía largo (M15+ en NY)
        "atr_cap": 1.8,
        "break_pull": False,
        "sl_mult": 1.8,
        "tp_mult": 3.0
    }
}

SWING_CFG = {
    "atr_mult_sl": 1.8,
    "atr_mult_tp": 4.0,
    "hours_news_block": 4,
    "trailing_trigger": 1.0,  # multiplicador ATR para BE
    "trailing_step": 0.5,
    "trailing_move": 0.3
}

# ---------------------------------------------------------------------------

def impulse_and_pullback_ok(df):
    """Detecta 'impulso → retroceso ≤ 38%' en los dos últimos candles"""
    if len(df) < 3: 
        return True
    prev = df.iloc[-2]
    last = df.iloc[-1]
    impulse = abs(prev['close'] - prev['open'])
    if impulse == 0: 
        return False
    pullback = abs(last['close'] - last['open'])
    return pullback / impulse <= 0.38

def intraday_extra_checks(tf_name, session, df):
    if tf_name not in ("M15", "M30", "H1"): 
        return True
    cfg = INTRADAY_SESSION_CFG.get(session)
    if not cfg:
        return True
    atr = df['atr'].iloc[-1]
    atr20 = df['atr'].rolling(20).mean().iloc[-1]
    if atr20==0 or pd.isna(atr20):
        return False
    if atr > atr20 * cfg['atr_cap']:
        log(f"INTRADAY_FILTER ({tf_name}): ATR alto {atr:.2f} > cap {cfg['atr_cap']}×ATR20")
        return False
    if cfg['break_pull'] and not impulse_and_pullback_ok(df):
        log(f"INTRADAY_FILTER ({tf_name}): No hubo pull‑back tras impulso")
        return False
    return True

# --- News filter (stub, implement real API if needed) ------------------------
def news_high_impact_soon(hours=4):
    # Placeholder always returns False; integrate real calendar later
    return False
# -----------------------------------------------------------------------------

def trend_alignment(pair):
    try:
        df_h4 = indicators(df_rates(mt5.TIMEFRAME_H4, n=150))
        df_d1 = indicators(df_rates(mt5.TIMEFRAME_D1, n=150))
    except Exception as e:
        log(f"SWING_FILTER: error datos {e}")
        return False
    dir_h4 = "UP" if df_h4['ema_s'].iloc[-1] > df_h4['ema_l'].iloc[-1] else "DOWN"
    dir_d1 = "UP" if df_d1['ema_s'].iloc[-1] > df_d1['ema_l'].iloc[-1] else "DOWN"
    return dir_h4 == dir_d1

def swing_entry_ok(pair):
    if news_high_impact_soon(hours=SWING_CFG['hours_news_block']):
        log("SWING_FILTER: Evento de alto impacto próximo – descartando entrada")
        return False
    if not trend_alignment(pair):
        log("SWING_FILTER: Tendencias H4 y D1 no alineadas")
        return False
    return True
'''
# --- Monkey‑patch analyse ----------------------------------------------------
_original_analyse = analyse
def analyse(tf_name, tf_code):
    current_session = session_name()
    # Filtrado extra intradía
    if tf_name in ("M15", "M30", "H1"):
        try:
            df_raw = df_rates(tf_code, n=250)
            df_intraday = indicators(df_raw)
        except Exception as ex:
            log(f"INTRADAY_FILTER_ERR ({tf_name}): {ex}")
            return
        if not intraday_extra_checks(tf_name, current_session, df_intraday):
            return  # descartada por filtro nuevo
    # Filtrado swing
    if tf_name in ("H4", "D1"):
        if not swing_entry_ok(PAR):
            return
    # Llamar la lógica original
    return _original_analyse(tf_name, tf_code)

log("[v23] Parches de sesión intradía/swing cargados correctamente.")
# ============================================================================


def close_trade(ticket: int) -> bool:
    """
    Cierra la posición `ticket`. Devuelve True si se ejecuta,
    False si falla. Registra el resultado en el log.
    """
    try:
        pos_list = mt5.positions_get(ticket=ticket)
    except Exception as e:
        log(f"⚠️ Error al consultar ticket {ticket}: {e}")
        return False

    if not pos_list:
        log(f"❌ Ticket {ticket} no existe o ya está cerrado")
        return False

    pos = pos_list[0]
    lot = pos.volume
    symbol = pos.symbol
    opposite_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY

    tick = mt5.symbol_info_tick(symbol)
    price = tick.bid if opposite_type == mt5.ORDER_TYPE_BUY else tick.ask

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "position":     ticket,
        "type":         opposite_type,
        "volume":       lot,
        "price":        price,
        "deviation":    30,
        "magic":        MAGIC,
        "comment":      "Bot-close",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        log(f"⚠️ MT5 close error {result.retcode} [{result.comment}] en ticket {ticket}")
        log(f"🔍 Request: {request}")
        return False

    log(f"✅ Cerrado ticket {ticket} @ {result.price} ({result.volume} lots)")
    return True
'''