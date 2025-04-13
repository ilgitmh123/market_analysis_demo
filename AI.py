import time
import requests
import pandas as pd
import talib as ta
import logging
import numpy as np
from sklearn.linear_model import LinearRegression
from telethon import TelegramClient, events
import re
# Cấu hình logging
logging.basicConfig(level=logging.INFO)
# Cấu hình API Binance
APIURL = "https://api.binance.com/api/v3/klines"
ORDER_BOOK_URL = "https://api.binance.com/api/v3/depth"
FUTURES_API_URL = "https://fapi.binance.com/fapi/v1/globalLongShortAccountRatio"
# Telegram Bot
api_id = 
api_hash = 
phone_number = 
client = TelegramClient('session_file', api_id, api_hash)
# Lấy dữ liệu nến từ Binance
def get_binance_candles(symbol, interval='5m', limit=618):
    url = f"{APIURL}?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    data = response.json()

    # Kiểm tra nếu dữ liệu trả về hợp lệ
    if isinstance(data, list) and len(data) > 0:
        candles = []
        for candle in data:
            try:
                # Kiểm tra và chuyển đổi các giá trị thành float
                timestamp = float(candle[0])
                open_price = float(candle[1])
                high_price = float(candle[2])
                low_price = float(candle[3])
                close_price = float(candle[4])
                volume = float(candle[5])

                candles.append({
                    'timestamp': timestamp,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume
                })
            except ValueError as e:
                logging.error(f"Error converting data to float: {e}")
                continue  # Bỏ qua các dữ liệu không hợp lệ
        
        # Chuyển đổi danh sách thành DataFrame của pandas
        df = pd.DataFrame(candles)
        return df
    else:
        logging.error("Error: Invalid data received from Binance API.")
        return pd.DataFrame()  # Trả về DataFrame rỗng nếu dữ liệu không hợp lệ

# Lấy Order Book từ Binance
def get_order_book(symbol):
    url = f"{ORDER_BOOK_URL}?symbol={symbol}&limit=100"
    response = requests.get(url).json()
    buy_orders = sum([float(order[1]) for order in response['bids']])
    sell_orders = sum([float(order[1]) for order in response['asks']])
    return buy_orders, sell_orders



# Lấy thông tin tâm lý thị trường từ Binance (Sử dụng /api/v3/ticker/24hr)
def get_market_sentiment(symbol):
    url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Kiểm tra nếu có lỗi HTTP
        data = response.json()

        # Kiểm tra nếu dữ liệu trả về hợp lệ
        if 'priceChangePercent' in data:
            price_change_percent = float(data['priceChangePercent'])

            sentiment = "Tâm lý thị trường trung lập ⚪️"
            if price_change_percent > 0:
                sentiment = "Tâm lý thị trường nghiêng về phe mua (Bullish) 🟢"
            elif price_change_percent < 0:
                sentiment = "Tâm lý thị trường nghiêng về phe bán (Bearish) 🔴"

            return sentiment, price_change_percent
        else:
            logging.error(f"Error: Missing 'priceChangePercent' in response for {symbol}. Response: {data}")
            return "Tâm lý thị trường không xác định", 0  # Trả về giá trị mặc định nếu không có 'priceChangePercent'

    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP Error: {e}")
        return "Tâm lý thị trường không xác định", 0  # Trả về giá trị mặc định nếu có lỗi kết nối

    except ValueError as e:
        logging.error(f"Error processing JSON response: {e}")
        return "Tâm lý thị trường không xác định", 0  # Trả về giá trị mặc định nếu có lỗi khi xử lý dữ liệu

# Tính toán các chỉ báo kỹ thuật
def calculate_indicators(df):
    if df.empty:
        logging.error("DataFrame is empty. No data available to calculate indicators.")
        return df
    df['rsi'] = ta.RSI(df['close'], timeperiod=9)
    df['macd'], df['macd_signal'], _ = ta.MACD(df['close'])
    df['upper_band'], df['middle_band'], df['lower_band'] = ta.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
    df['psar'] = ta.SAR(df['high'], df['low'], acceleration=0.02, maximum=0.2)
    df['stochastic_k'], df['stochastic_d'] = ta.STOCH(df['high'], df['low'], df['close'], fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    return calculate_ichimoku(df)

# Tính toán Ichimoku Kinko Hyo
def calculate_ichimoku(df):
    df['tenkan_sen'] = (df['high'].rolling(window=9).max() + df['low'].rolling(window=9).min()) / 2
    df['kijun_sen'] = (df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
    df['senkou_span_b'] = (df['high'].rolling(window=52).max() + df['low'].rolling(window=52).min()) / 2
    df['senkou_span_b'] = df['senkou_span_b'].shift(26)
    df['chikou_span'] = df['close'].shift(-26)
    return df

# Phát hiện kháng cự & hỗ trợ gần nhất
def find_support_resistance(df):
    support = df['low'].min()
    resistance = df['high'].max()
    return support, resistance

# Dự báo giá tương lai bằng Linear Regression
def predict_future_prices(df, future_steps=[5, 10, 15, 30, 60]):
    X = np.array(range(len(df))).reshape(-1, 1)
    y = df['close'].values

    model = LinearRegression()
    model.fit(X, y)

    future_X = np.array([len(df) + step for step in future_steps]).reshape(-1, 1)
    future_prices = model.predict(future_X)

    return {future_steps[i]: future_prices[i] for i in range(len(future_steps))}

# Tính toán tỷ lệ thắng cho từng target
def calculate_win_percentage(df, entry_price, order_type):
    win_percentage = 0

    # Tính các chỉ báo kỹ thuật
    if order_type == 'LONG':
        if df['tenkan_sen'].iloc[-1] > df['kijun_sen'].iloc[-1] and df['close'].iloc[-1] > df['senkou_span_a'].iloc[-1] and df['close'].iloc[-1] > df['senkou_span_b'].iloc[-1]:
            win_percentage += 20
        if entry_price > df['upper_band'].iloc[-1]:  
            win_percentage += 15
        if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1]:  
            win_percentage += 20
        if df['psar'].iloc[-1] < df['close'].iloc[-1]:  
            win_percentage += 10
        if df['rsi'].iloc[-1] < 30:
            win_percentage += 15
    elif order_type == 'SHORT':
        if df['tenkan_sen'].iloc[-1] < df['kijun_sen'].iloc[-1] and df['close'].iloc[-1] < df['senkou_span_a'].iloc[-1] and df['close'].iloc[-1] < df['senkou_span_b'].iloc[-1]:
            win_percentage += 20
        if entry_price < df['lower_band'].iloc[-1]:  
            win_percentage += 15
        if df['macd'].iloc[-1] < df['macd_signal'].iloc[-1]: 
            win_percentage += 20
        if df['psar'].iloc[-1] > df['close'].iloc[-1]:  
            win_percentage += 10
        if df['rsi'].iloc[-1] > 70:
            win_percentage += 15

    return win_percentage

# Phân tích xu hướng tổng quan
def analyze_trend(df):
    df = calculate_indicators(df)
    trend = "Xu hướng không rõ ràng ⚪️"
    signals = 0

    if df['rsi'].iloc[-1] > 50:
        signals += 1
    if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1]:
        signals += 1
    if df['psar'].iloc[-1] < df['close'].iloc[-1]:
        signals += 1

    if signals >= 2:
        trend = "Xu hướng chính: Tăng 🟢"
    elif signals == 1:
        trend = "Xu hướng trung lập ⚪️"
    else:
        trend = "Xu hướng chính: Giảm 🔴"
    return trend
def get_current_market_price(symbol):
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    response = requests.get(url).json()
    if 'price' in response:
        return float(response['price'])  # Trả về giá thị trường hiện tại
    else:
        logging.error(f"Error: Could not get current market price for {symbol}.")
        return None  # Nếu không lấy được giá, trả về None
async def send_analysis(event, message):
    # Sử dụng hàm process_message để phân tích thông tin từ tin nhắn
    symbol, entry_price, targets, stop_loss, leverage = process_message(message)
    
    if symbol is None:  # Nếu symbol không hợp lệ, bot sẽ không tiếp tục xử lý
        await event.respond("⚠️ Cặp giao dịch không hợp lệ. Vui lòng thử lại.")
        return
    if entry_price is None:
        entry_price = get_current_market_price(symbol)
        if entry_price is None:  # Nếu không lấy được giá thị trường, dừng lại
            await event.respond("⚠️ Không thể lấy giá thị trường. Vui lòng thử lại sau.")
            return
    # Lấy thông tin về lệnh giao dịch từ tin nhắn (LONG hoặc SHORT)
    order_type = 'LONG' if 'LONG' in message else 'SHORT'  # Nếu tin nhắn chứa 'LONG', dùng 'LONG' nếu không thì dùng 'SHORT'
    
    # Lấy dữ liệu từ Binance API
    df = get_binance_candles(symbol)
    buy_orders, sell_orders = get_order_book(symbol)
    sentiment, long_short_ratio = get_market_sentiment(symbol)
    trend = analyze_trend(df)
    support, resistance = find_support_resistance(df)
    future_prices = predict_future_prices(df)

    # Tính tỷ lệ thắng cho từng target
    win_percentages = [calculate_win_percentage(df, entry_price, order_type) for target in targets]

    order_book_analysis = "🔹 Order Book: "
    if buy_orders > sell_orders:
        order_book_analysis += f"Nhiều lệnh mua hơn (Bullish) 🟢 ({buy_orders:.10f} vs {sell_orders:.10f})"
    else:
        order_book_analysis += f"Nhiều lệnh bán hơn (Bearish) 🔴 ({sell_orders:.10f} vs {buy_orders:.10f})"

    future_price_message = "\n📈 **Dự báo giá tương lai:**\n"
    for minutes, price in future_prices.items():
        future_price_message += f"⏳ {minutes} phút tới: **{price:.10f}** USDT\n"

    # Tạo thông báo chi tiết về tỷ lệ thắng cho từng target
    targets_message = ""
    for i, win_percentage in enumerate(win_percentages):
        targets_message += f"🎯 **Target {i + 1}:** {targets[i]} USDT - Tỷ lệ đạt đến là : {win_percentage}%\n"

    # Tạo thông báo cuối cùng
    # Lấy các chỉ số cần thiết từ DataFrame
    rsi_value = df['rsi'].iloc[-1]  # RSI hiện tại
    upper_bollinger_band = df['upper_band'].iloc[-1]  # Mức trên của Bollinger Band
    lower_bollinger_band = df['lower_band'].iloc[-1]  # Mức dưới của Bollinger Band
    ichimoku_span_a = df['senkou_span_a'].iloc[-1]  # Mây Ichimoku Span A
    ichimoku_span_b = df['senkou_span_b'].iloc[-1]  # Mây Ichimoku Span B

    # Tạo thông báo
    message = f"""
    📊 **Phân tích {symbol}**
    - {trend}
    - {sentiment} (Tỷ lệ Long/Short: {long_short_ratio:.10f})
    - {order_book_analysis}
    - **Hỗ trợ gần nhất:** {support:.10f}
    - **Kháng cự gần nhất:** {resistance:.10f}
    - **RSI hiện tại:** {rsi_value:.10f}
    - **Mức trên của Bollinger Band:** {upper_bollinger_band:.10f}
    - **Mức dưới của Bollinger Band:** {lower_bollinger_band:.10f}
    - **Mây Ichimoku (Span A):** {ichimoku_span_a:.10f}
    - **Mây Ichimoku (Span B):** {ichimoku_span_b:.10f}
    {future_price_message}
    {targets_message}
    """


def clean_part(part):
    # Biểu thức chính quy để chỉ giữ lại chữ cái và số
    cleaned_part = re.sub(r'[^a-zA-Z0-9]', '', part)  # Loại bỏ tất cả ký tự không phải chữ cái và số
    return cleaned_part

    # Gửi thông báo đến người dùng
    await event.respond(message)
    logging.info(f"Sent message: {message}")

def calculate_fibonacci(entry_min, entry_max, fibonacci_ratio=0.618):
    return (entry_max - entry_min) * fibonacci_ratio + entry_min
def is_valid_symbol(symbol):
    url = "https://api.binance.com/api/v3/exchangeInfo"
    response = requests.get(url).json()

    # Kiểm tra xem symbol có tồn tại trong danh sách tradingPairs
    for symbol_info in response['symbols']:
        if symbol_info['symbol'] == symbol.upper():  # Kiểm tra symbol, dùng upper() để chuyển chữ hoa
            return True
    return False
def process_message(message):
    lines = message.split("\n")
    logging.info(f"Lines parsed: {lines}")
    order_type = None
    for line in lines:
        line_lower = line.strip().lower()
        if 'long' in line_lower:
            order_type = 'LONG'
        elif 'short' in line_lower:
            order_type = 'SHORT'
        elif 'buy' in line_lower:
            order_type = 'LONG'  # Nếu có buy thì mặc định là LONG
        elif 'sell' in line_lower:
            order_type = 'SHORT'  # Nếu có sell thì mặc định là SHORT
    
    prefixs = ["","1000", "10000", "100000", "1000000"]
    symbol = None
    for line in lines:
        parts = line.split()
        for part in parts:
            if 'usd' in part.lower():
                symbol = part.replace('#', '').replace('/', '').replace('@', '').replace('!', '').replace('"','').replace('*','')
                break
            else:
                cleaned_part = part.replace('#', '').replace('/', '').replace('@', '').replace('!', '').replace('"', '').replace('*', '')
                if is_valid_symbol(cleaned_part+"USDT"):
                    symbol = cleaned_part + "USDT"
                    break
                else:
                    for prefix in prefixs:
                        if is_valid_symbol(prefix + cleaned_part):
                            symbol = prefix + cleaned_part
                            break
                
    # Tìm ENTRY: Duyệt các dòng tiếp theo đến khi gặp Target hoặc stoploss
    entries = []
    entries_done = False
    targets = []
    targets_done = False
    stl = []
    stl_done = False
    for line in lines:
        if "leverage" in line.lower():
            continue
        else:
            if "entry" in line.lower():
                entries_done = True;
            if "target" in line.lower() or "targets" in line.lower():
                entries_done = False 
                targets_done = True
            if "stl" in line.lower() or "stop" in line.lower() or "stoploss" in line.lower() or "stop-loss" in line.lower() or "stop_loss" in line.lower():
                targets_done = False
                stl_done = True 
            if entries_done == True:
                for part in line.split():
                    try:
                        part = clean_part(part)
                        number = float(part)  # Chuyển thành số float
                        entries.append(number)  # Nếu có, thêm vào danh sách entry

                    except ValueError:
                        continue 
            if targets_done == True:
                for part in line.split():
                    try:
                        part = clean_part(part)
                        number = float(part)  # Chuyển thành số float
                        targets.append(number)  # Nếu có, thêm vào danh sách entry
                    except ValueError:
                        continue    
            if stl_done == True:
                for part in line.split():
                    try:
                        part = clean_part(part)
                        number = float(part)  # Chuyển thành số float
                        stl.append(number)  # Nếu có, thêm vào danh sách entry
                    except ValueError:
                        continue  
    if not entries:
        entry_price = get_current_market_price(symbol)
    else:
        entry_min = min(entries)
        entry_max = max(entries)
        entry_price = calculate_fibonacci(entry_min,entry_max)        
    leverage = "10X"
    entries_str = ' - '.join(map(str, entries))
    targets_str = ' - '.join(map(str, targets))
    stl_str = ' - '.join(map(str, stl))
    # Log thông tin
    logging.info(f"Symbol: {symbol}")
    if not entries:
        logging.info(f"Entry Price Range: {entry_price}")
    else:
        logging.info(f"Entry Price Range: {entry_min} - {entry_max}")
    logging.info(f"Calculated Entry Price: {entries_str}")
    logging.info(f"Targets: {targets_str}")
    logging.info(f"Stop Loss: {stl_str}")
    logging.info(f"Leverage: {leverage}")
    
    # Trả lại dữ liệu đã phân tích
    return symbol, entry_price, targets, stl, leverage


# Xử lý tin nhắn Telegram
async def main():
    await client.start(phone_number)
    logging.info("Bot is running...")

    @client.on(events.NewMessage)
    async def handler(event):
        if event.is_private:
            message = event.message.text.strip().upper().replace("/", "").replace("#", "")
            await send_analysis(event, message)

    await client.run_until_disconnected()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
