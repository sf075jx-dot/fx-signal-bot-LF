import os
from datetime import datetime
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ReportLabによるPDF生成用ライブラリ
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import urllib.request

# ==========================================
# 定数・基本設定
# ==========================================
SYMBOL = "JPY=X"        # USD/JPY
PERIOD = "5y"           # AI学習・バックテスト用は過去5年分を維持
INTERVAL = "1d"         # 日足

US10Y_SYMBOL = "^TNX"   # 米10年国債利回り
SP500_SYMBOL = "^GSPC"  # S&P500指数
VIX_SYMBOL = "^VIX"     # VIX恐怖指数
EURJPY_SYMBOL = "EURJPY=X" # ユーロ円

INITIAL_CAPITAL = 1_000_000  
LOT_SIZE = 10_000            
TP_MULT = 1.5                
SL_MULT = 1.0                

LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
LINE_USER_ID = os.environ.get("LINE_USER_ID")


# PDF用の日本語フォント設定（主軸コード）
def setup_pdf_japanese_font():
    # Ubuntuでapt-get installしたIPAゴシックのパス
    font_path = "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"
    
    # ローカル開発環境（Windows等）を考慮してフォントファイルが存在するかフォールバックを持たせる場合
    if not os.path.exists(font_path):
        font_path = "ipaexg.ttf" # 必要に応じてローカル用のパス
        if not os.path.exists(font_path):
            font_url = "https://github.com/google/fonts/raw/main/ofl/ipaexg/IPAexGothic%5B%5D.ttf"
            try:
                urllib.request.urlretrieve(font_url, font_path)
            except Exception:
                pass

    if os.path.exists(font_path):
        pdfmetrics.registerFont(TTFont('IPAexGothic', font_path))
        return 'IPAexGothic'
    
    return 'Helvetica'


def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))


def fetch_and_prepare_data():
    print("1/4 市場データの取得・計算中...")
    df = yf.download(SYMBOL, period=PERIOD, interval=INTERVAL)
    df_us10y = yf.download(US10Y_SYMBOL, period=PERIOD, interval=INTERVAL)
    df_sp500 = yf.download(SP500_SYMBOL, period=PERIOD, interval=INTERVAL)
    df_vix = yf.download(VIX_SYMBOL, period=PERIOD, interval=INTERVAL)
    df_eurjpy = yf.download(EURJPY_SYMBOL, period=PERIOD, interval=INTERVAL)

    if df.empty or df_us10y.empty or df_sp500.empty or df_vix.empty or df_eurjpy.empty:
        raise ValueError("市場データの取得に失敗しました。")

    for d in [df, df_us10y, df_sp500, df_vix, df_eurjpy]:
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)

    df["US10Y_Close"] = df_us10y["Close"]
    df["SP500_Close"] = df_sp500["Close"]
    df["VIX_Close"] = df_vix["Close"]
    df["EURJPY_Close"] = df_eurjpy["Close"]
    df.ffill(inplace=True)

    df["Close_Smooth"] = df["Close"].ewm(span=3, adjust=False).mean()
    df["Returns"] = df["Close_Smooth"].pct_change()
    df["Return_1d"] = df["Close_Smooth"].pct_change(1)
    df["Return_3d"] = df["Close_Smooth"].pct_change(3)
    df["Return_5d"] = df["Close_Smooth"].pct_change(5)
    df["Return_10d"] = df["Close_Smooth"].pct_change(10)

    df["US10Y_Return"] = df["US10Y_Close"].pct_change()
    df["US10Y_Return_5d"] = df["US10Y_Close"].pct_change(5)
    df["SP500_Return"] = df["SP500_Close"].pct_change()
    df["SP500_Return_5d"] = df["SP500_Close"].pct_change(5)
    df["US10Y_USDJPY_Diff"] = df["US10Y_Return"] - df["Returns"]

    df["VIX_Return"] = df["VIX_Close"].pct_change()
    df["EURJPY_Return"] = df["EURJPY_Close"].pct_change()
    df["EURJPY_EURJPY_Return_5d"] = df["EURJPY_Close"].pct_change(5)

    body = np.abs(df["Close"] - df["Open"])
    full_range = df["High"] - df["Low"] + 1e-8
    df["Body_Ratio"] = body / full_range
    df["Upper_Shadow"] = (df["High"] - np.maximum(df["Close"], df["Open"])) / full_range
    df["Lower_Shadow"] = (np.minimum(df["Close"], df["Open"]) - df["Low"]) / full_range

    df["SMA200"] = df["Close"].rolling(window=200).mean()
    df["SMA50"] = df["Close"].rolling(window=50).mean()
    df["SMA20"] = df["Close"].rolling(window=20).mean()
    
    df["SMA_Ratio_20"] = df["Close"] / df["SMA20"] - 1.0
    df["SMA_Ratio_50"] = df["Close"] / df["SMA50"] - 1.0
    df["SMA_Ratio_200"] = df["Close"] / df["SMA200"] - 1.0

    std20 = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["SMA20"] + (std20 * 2)
    df["BB_Lower"] = df["SMA20"] - (std20 * 2)
    df["BB_Pos"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"] + 1e-8)

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    df["MACD_Hist_Slope"] = df["MACD_Hist"] - df["MACD_Hist"].shift(2)

    df["RSI"] = compute_rsi(df["Close"], window=14)
    df["RSI_Slope"] = df["RSI"] - df["RSI"].shift(3)

    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(window=14).mean()
    df["ATR_Ratio"] = df["ATR"] / df["ATR"].rolling(window=50).mean()
    df["Volatility"] = df["Returns"].rolling(window=10).std()

    df["DayOfWeek"] = df.index.dayofweek
    df["Target_Up"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    df.dropna(inplace=True)
    return df


def train_and_predict(df):
    print("2/4 アンサンブルAI学習＆予測中...")

    features = [
        "Returns", "Return_1d", "Return_3d", "Return_5d", "Return_10d",
        "US10Y_Return", "US10Y_Return_5d", "SP500_Return", "SP500_Return_5d",
        "US10Y_USDJPY_Diff", "VIX_Close", "VIX_Return",
        "EURJPY_Return", "EURJPY_EURJPY_Return_5d",
        "Body_Ratio", "Upper_Shadow", "Lower_Shadow",
        "Volatility", "RSI", "RSI_Slope",
        "SMA_Ratio_20", "SMA_Ratio_50", "SMA_Ratio_200",
        "BB_Pos", "MACD", "MACD_Hist", "MACD_Hist_Slope", "ATR_Ratio", "DayOfWeek"
    ]

    train_df = df.iloc[:-1].reset_index(drop=True)
    latest_data = df.iloc[[-1]]

    X_train = train_df[features]
    y_train = train_df["Target_Up"]
    pos_weight = (len(y_train) - sum(y_train)) / (sum(y_train) + 1e-8)

    # 1. LightGBM
    lgb_params = {
        "objective": "binary", "metric": "binary_logloss", "learning_rate": 0.01,
        "max_depth": 3, "num_leaves": 6, "feature_fraction": 0.6,
        "bagging_fraction": 0.7, "bagging_freq": 1, "scale_pos_weight": pos_weight, "verbose": -1, "seed": 42
    }
    dtrain_lgb = lgb.Dataset(X_train, label=y_train)
    model_lgb = lgb.train(lgb_params, dtrain_lgb, num_boost_round=120)
    p_lgb = float(model_lgb.predict(latest_data[features])[0])

    # 2. XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=120, learning_rate=0.01, max_depth=3,
        scale_pos_weight=pos_weight, random_state=42, verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    p_xgb = float(xgb_model.predict_proba(latest_data[features])[0][1])

    # 3. CatBoost
    cat_model = CatBoostClassifier(
        iterations=120, learning_rate=0.01, depth=3,
        auto_class_weights='Balanced', random_seed=42, verbose=0
    )
    cat_model.fit(X_train, y_train)
    p_cat = float(cat_model.predict_proba(latest_data[features])[0][1])

    prob_up = float((p_lgb + p_xgb + p_cat) / 3.0)
    prob_down = 1.0 - prob_up

    latest_close = float(latest_data["Close"].values[0])
    latest_sma200 = float(latest_data["SMA200"].values[0])
    latest_atr = float(latest_data["ATR"].values[0])
    latest_atr_ratio = float(latest_data["ATR_Ratio"].values[0])
    latest_vix = float(latest_data["VIX_Close"].values[0])
    latest_rsi = float(latest_data["RSI"].values[0])

    is_uptrend = latest_close > latest_sma200
    threshold = 0.67 if latest_atr_ratio > 1.25 else 0.62

    if prob_up >= threshold and is_uptrend:
        signal = "BUY"
    elif prob_down >= threshold and not is_uptrend:
        signal = "SELL"
    else:
        signal = "HOLD"

    return signal, prob_up, prob_down, p_lgb, p_xgb, p_cat, latest_close, latest_sma200, latest_atr, latest_atr_ratio, latest_vix, latest_rsi, threshold


def generate_backtest_chart_and_pdf(
    df, signal, prob_up, prob_down, p_lgb, p_xgb, p_cat, 
    current_price, sma200, atr, atr_ratio, vix, rsi, threshold
):
    print("3/4 バックテストチャートおよびPDFレポート生成中...")
    
    features = [
        "Returns", "Return_1d", "Return_3d", "Return_5d", "Return_10d",
        "US10Y_Return", "US10Y_Return_5d", "SP500_Return", "SP500_Return_5d",
        "US10Y_USDJPY_Diff", "VIX_Close", "VIX_Return",
        "EURJPY_Return", "EURJPY_EURJPY_Return_5d",
        "Body_Ratio", "Upper_Shadow", "Lower_Shadow",
        "Volatility", "RSI", "RSI_Slope",
        "SMA_Ratio_20", "SMA_Ratio_50", "SMA_Ratio_200",
        "BB_Pos", "MACD", "MACD_Hist", "MACD_Hist_Slope", "ATR_Ratio", "DayOfWeek"
    ]

    min_train_size = 250
    results = []

    for i in range(min_train_size, len(df) - 1):
        train_df = df.iloc[:i]
        test_row = df.iloc[[i]]

        X_train = train_df[features]
        y_train = train_df["Target_Up"]
        pos_weight = (len(y_train) - sum(y_train)) / (sum(y_train) + 1e-8)

        dtrain = lgb.Dataset(X_train, label=y_train)
        params = {"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.02, "max_depth": 3, "num_leaves": 6, "scale_pos_weight": pos_weight, "verbose": -1, "seed": 42}
        gbm = lgb.train(params, dtrain, num_boost_round=60)
        
        xgb_m = xgb.XGBClassifier(n_estimators=60, learning_rate=0.02, max_depth=3, scale_pos_weight=pos_weight, random_state=42, verbosity=0)
        xgb_m.fit(X_train, y_train)

        p1 = float(gbm.predict(test_row[features])[0])
        p2 = float(xgb_m.predict_proba(test_row[features])[0][1])
        p_avg = float((p1 + p2) / 2.0)
        p_down = 1.0 - p_avg

        curr_close = float(test_row["Close"].values[0])
        next_high = float(df.iloc[i + 1]["High"])
        next_low = float(df.iloc[i + 1]["Low"])
        next_close = float(df.iloc[i + 1]["Close"])
        s200 = float(test_row["SMA200"].values[0])
        t_atr = float(test_row["ATR"].values[0])
        t_atr_ratio = float(test_row["ATR_Ratio"].values[0])

        is_up = curr_close > s200
        th = 0.67 if t_atr_ratio > 1.25 else 0.62

        sig = "HOLD"
        if p_avg >= th and is_up:
            sig = "BUY"
        elif p_down >= th and not is_up:
            sig = "SELL"

        pnl = 0.0
        if sig == "BUY":
            tp = curr_close + (t_atr * TP_MULT)
            sl = curr_close - (t_atr * SL_MULT)
            if next_high >= tp:
                pnl = (tp - curr_close) * LOT_SIZE
            elif next_low <= sl:
                pnl = (sl - curr_close) * LOT_SIZE
            else:
                pnl = (next_close - curr_close) * LOT_SIZE
        elif sig == "SELL":
            tp = curr_close - (t_atr * TP_MULT)
            sl = curr_close + (t_atr * SL_MULT)
            if next_low <= tp:
                pnl = (curr_close - tp) * LOT_SIZE
            elif next_high >= sl:
                pnl = (curr_close - sl) * LOT_SIZE
            else:
                pnl = (curr_close - next_close) * LOT_SIZE

        results.append({"Date": df.index[i + 1], "PnL": pnl})

    res_df = pd.DataFrame(results)
    res_df["Equity"] = INITIAL_CAPITAL + res_df["PnL"].cumsum()

    # グラフ描画（英語表記にして文字化け回避）
    chart_path = "backtest_result.png"
    plt.figure(figsize=(7, 3.5), dpi=100)
    plt.plot(res_df["Date"], res_df["Equity"], label="AI Ensemble Equity", color="#2ca02c", linewidth=2)
    plt.axhline(INITIAL_CAPITAL, color="gray", linestyle="--", label="Initial Capital")
    
    plt.title("USD/JPY AI Ensemble Backtest Equity", fontsize=11)
    plt.xlabel("Date", fontsize=9)
    plt.ylabel("Equity (JPY)", fontsize=9)
    plt.legend(loc="upper left", fontsize=8)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(chart_path, format="png", dpi=100)
    plt.close()

    # ==========================================
    # 3. 直近期間（約90営業日）のローソク足＆SMAチャートの生成
    # ==========================================
    chart_sma_path = "sma_chart.png"
    plt.figure(figsize=(7, 3.5), dpi=100)
    
    # 直近90日分にデータを絞り込む
    df_recent = df.tail(90).copy()
    
    # 簡易ローソク足描画（ヒゲと実体）
    # 上昇: 緑/青系(例: #2ca02c), 下落: 赤系(例: #d62728)
    for idx, row in df_recent.iterrows():
        o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]
        color = "#2ca02c" if c >= o else "#d62728"
        # ヒゲ（HighからLowまで）
        plt.vlines(idx, l, h, color=color, linewidth=1, alpha=0.8)
        # 実体（OpenからCloseまで）
        body_bottom = min(o, c)
        body_height = max(abs(c - o), 1e-8)
        plt.bar(idx, body_height, bottom=body_bottom, width=0.6, color=color, alpha=0.9)

    # 移動平均線の描画（直近部分）
    plt.plot(df_recent.index, df_recent["SMA20"], label="SMA 20", color="#ff7f0e", linewidth=1.2)
    plt.plot(df_recent.index, df_recent["SMA50"], label="SMA 50", color="#1f77b4", linewidth=1.2)
    plt.plot(df_recent.index, df_recent["SMA200"], label="SMA 200", color="#9467bd", linewidth=1.2)

    plt.title("USD/JPY Candlestick & Moving Averages (Recent 90 Days)", fontsize=11)
    plt.xlabel("Date", fontsize=9)
    plt.ylabel("Price (JPY)", fontsize=9)
    plt.legend(loc="upper left", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(chart_sma_path, format="png", dpi=100)
    plt.close()

    pdf_path = "ai_report.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []

    pdf_font = setup_pdf_japanese_font()

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontName=pdf_font, fontSize=16, textColor=colors.HexColor('#1f77b4'), spaceAfter=15)
    heading_style = ParagraphStyle('HeadingStyle', parent=styles['Heading2'], fontName=pdf_font, fontSize=12, textColor=colors.HexColor('#333333'), spaceBefore=10, spaceAfter=6)
    normal_style = ParagraphStyle('NormalStyle', parent=styles['Normal'], fontName=pdf_font, fontSize=10, textColor=colors.HexColor('#444444'), leading=14)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    story.append(Paragraph("ドル円（USD/JPY）自動AI取引 レポート", title_style))
    story.append(Paragraph(f"生成日時: {now_str} JST", normal_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("1. 本日のAI診断サマリー（3大モデル個別内訳付き）", heading_style))
    
    if signal == "BUY":
        sig_text = "買い (BUY - 上昇予測)"
    elif signal == "SELL":
        sig_text = "売り (SELL - 下落予測)"
    else:
        sig_text = "様子見 (HOLD - 待機推奨)"
    
    data_summary = [
        [Paragraph("<b>AI推奨判定</b>", normal_style), Paragraph(f"<b>{sig_text}</b>", normal_style)],
        [Paragraph("アンサンブル平均 (上昇)", normal_style), Paragraph(f"{prob_up * 100:.1f}%", normal_style)],
        [Paragraph("アンサンブル平均 (下落)", normal_style), Paragraph(f"{prob_down * 100:.1f}%", normal_style)],
        [Paragraph("  - LightGBM 上昇確率", normal_style), Paragraph(f"{p_lgb * 100:.1f}%", normal_style)],
        [Paragraph("  - XGBoost 上昇確率", normal_style), Paragraph(f"{p_xgb * 100:.1f}%", normal_style)],
        [Paragraph("  - CatBoost 上昇確率", normal_style), Paragraph(f"{p_cat * 100:.1f}%", normal_style)],
        [Paragraph("現在価格", normal_style), Paragraph(f"{current_price:.2f} 円", normal_style)],
        [Paragraph("トレンド判定 (SMA200)", normal_style), Paragraph(f"{'上昇トレンド' if current_price > sma200 else '下落トレンド'} ({sma200:.2f}円)", normal_style)],
        [Paragraph("市場ボラティリティ", normal_style), Paragraph(f"VIX: {vix:.2f} | ATR比率: {atr_ratio:.2f}", normal_style)],
        [Paragraph("RSI (14)", normal_style), Paragraph(f"{rsi:.1f}", normal_style)]
    ]

    t = Table(data_summary, colWidths=[180, 320])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f9f9f9')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#dddddd')),
        ('PADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t)
    story.append(Spacer(1, 15))

    story.append(Paragraph("2. 過去5年間のバックテスト資産曲線", heading_style))
    story.append(RLImage(chart_path, width=450, height=225))
    story.append(Spacer(1, 15))

    # 3. ローソク足＆移動平均線チャートのPDF組み込み
    story.append(Paragraph("3. 直近チャート (ローソク足 & 移動平均線)", heading_style))
    story.append(RLImage(chart_sma_path, width=450, height=225))

    doc.build(story)
    print("-> PDFレポートおよびチャートの生成が完了しました。")
    return pdf_path, chart_path


def send_line_notification(text):
    if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_USER_ID:
        print("Error: LINE環境変数が設定されていません。")
        return

    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"
    }

    messages = [{"type": "text", "text": text}]
    payload = {"to": LINE_USER_ID, "messages": messages}
    
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        print("-> LINEへメッセージを送信しました!")
    else:
        print(f"LINE送信失敗: {response.status_code}, {response.text}")


def main():
    try:
        df = fetch_and_prepare_data()
        signal, prob_up, prob_down, p_lgb, p_xgb, p_cat, current_price, sma200, atr, atr_ratio, vix, rsi, threshold = train_and_predict(df)
        generate_backtest_chart_and_pdf(df, signal, prob_up, prob_down, p_lgb, p_xgb, p_cat, current_price, sma200, atr, atr_ratio, vix, rsi, threshold)

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

        if current_price > sma200:
            trend_str = "上向き（上昇トレンド）📈"
        else:
            trend_str = "下向き（下落トレンド）📉"

        is_high_volatility = atr_ratio > 1.4 or vix > 25.0
        
        if is_high_volatility:
            alert_header = "⚠️ 【注意：相場が激しく荒れています】\n値動きが非常に大きくなっています。無理な取引は控えて様子を見ましょう。\n───────────────────\n\n"
        else:
            alert_header = ""

        tp_mult = 1.5
        sl_mult = 1.0

        if prob_up > prob_down:
            status_str = "「上昇を予測」"
            status_note = "※3つのAIによる合議制分析の結果、上方向の確率が優勢です。"
        else:
            status_str = "「下落を予測」"
            status_note = "※3つのAIによる合議制分析の結果、下方向の確率が優勢です。"

        # ロング目線・ショート目線それぞれの損益ライン計算
        long_tp_price = current_price + (atr * tp_mult)
        long_sl_price = current_price - (atr * sl_mult)

        short_tp_price = current_price - (atr * tp_mult)
        short_sl_price = current_price + (atr * sl_mult)

        repo_owner = os.environ.get("GITHUB_REPOSITORY_OWNER", "")
        repo_full = os.environ.get("GITHUB_REPOSITORY", "")
        repo_name = repo_full.split("/")[-1] if repo_full else ""
        
        if repo_owner and repo_name:
            pdf_url = f"https://github.com/{repo_owner}/{repo_name}/blob/main/ai_report.pdf"
            report_section = f"📁 本日の詳細レポート（PDF）:\n{pdf_url}"
        else:
            report_section = "📁 詳細レポートはGitHubリポジトリをご確認ください。"

        message_text = f"""{alert_header}📊 ドル/円（USD/JPY）AI予測 📊
───────────────────
【本日のAI診断】
{status_str}
{status_note}

【AIの予測平均】
・上がりそう : {prob_up * 100:.1f}%
・下がりそう : {prob_down * 100:.1f}%

【現在の状況】
・いまの価格 : {current_price:.2f} 円
・全体の流れ : {trend_str}
・市場の荒れ具合 : {'激しく荒れています⚠' if is_high_volatility else '落ち着いています✨'}

【目安の価格】
■ ロング（買い）目線の場合
・利益確定目標 : 🎯 {long_tp_price:.2f} 円
・損切り撤退ライン : 🛡️ {long_sl_price:.2f} 円

■ ショート（売り）目線の場合
・エントリー目安 : {current_price:.2f} 円付近
・利益確定目標 : 🎯 {short_tp_price:.2f} 円
・損切り撤退ライン : 🛡️ {short_sl_price:.2f} 円

───────────────────
{report_section}

※この予報は複数AIによる合議制分析の参考情報です。売買はご自身の判断で行ってください。
配信時間: {now_str}"""

        print("4/4 LINE送信処理中...")
        send_line_notification(message_text)

    except Exception as e:
        print(f"エラーが発生しました: {e}")


if __name__ == "__main__":
    main()
