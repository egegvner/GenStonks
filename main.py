import streamlit as st
import sqlite3
import random
import time
import pandas as pd
import datetime
import os
import shutil
from streamlit_autorefresh import st_autorefresh
from streamlit_lightweight_charts import renderLightweightCharts
import argon2
from streamlit_cookies_controller import CookieController

@st.cache_resource
def get_db_connection():
    return sqlite3.connect("stockk.db", check_same_thread=False, uri=True)

ph = argon2.PasswordHasher(
    memory_cost=65536,
    time_cost=5,
    parallelism=4,
)

def hashPass(password):
    return ph.hash(password)

def verifyPass(hashed_password, entered_password):
    try:
        return ph.verify(hashed_password, entered_password)
    except Exception:
        return False

def inject_inter_font():
    # Load Inter from Google Fonts and apply globally
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
        <style>
        :root { --app-font: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', Arial, sans-serif; }
        html, body, [class^="css"], .stApp * { font-family: var(--app-font) !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

def init_db(conn):
    c = conn.cursor()
    # Users table (minimal for stocks app)
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            balance REAL DEFAULT 0,
            last_transaction_time TEXT
        )
        """
    )

    # Stocks master table
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS stocks (
            stock_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            symbol TEXT UNIQUE NOT NULL,
            price REAL NOT NULL,
            stock_amount REAL NOT NULL,
            dividend_rate REAL DEFAULT 0,
            change_rate REAL DEFAULT 0.5,
            open_price REAL,
            close_price REAL,
            last_updated TEXT
        )
        """
    )

    # Historical prices
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS stock_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_id INTEGER NOT NULL,
            price REAL NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY(stock_id) REFERENCES stocks(stock_id)
        )
        """
    )

    # User holdings
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS user_stocks (
            user_id INTEGER NOT NULL,
            stock_id INTEGER NOT NULL,
            quantity REAL NOT NULL,
            avg_buy_price REAL NOT NULL,
            purchase_date TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(user_id, stock_id),
            FOREIGN KEY(user_id) REFERENCES users(user_id),
            FOREIGN KEY(stock_id) REFERENCES stocks(stock_id)
        )
        """
    )

    # Transactions used for buy/sell & volume metrics
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS transactions (
            transaction_id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            type TEXT NOT NULL,
            amount REAL NOT NULL,
            stock_id INTEGER,
            quantity REAL,
            receiver_username TEXT,
            status TEXT DEFAULT 'Completed',
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(user_id),
            FOREIGN KEY(stock_id) REFERENCES stocks(stock_id)
        )
        """
    )

    # Ensure Government account exists (used as counterparty)
    c.execute("SELECT user_id FROM users WHERE username='Government'")
    if not c.fetchone():
        c.execute(
            "INSERT INTO users (username, password, balance) VALUES (?, ?, ?)",
            ("Government", hashPass("government"), 0.0),
        )

    conn.commit()

def format_number_with_dots(number):
    return f"{number:,}"

def format_number(num, decimals=2):
    suffixes = [
        (1e33, 'D'),
        (1e30, 'N'),
        (1e27, 'O'),
        (1e24, 'Sp'),
        (1e21, 'Sx'),
        (1e18, 'Qt'),
        (1e15, 'Qd'),
        (1e12, 'T'),
        (1e9, 'B'),
        (1e6, 'M'),
        (1e3, 'K'),
    ]

    if abs(num) < 1000:
        return f"{num:.{decimals}f}".rstrip('0').rstrip('.')

    for threshold, suffix in suffixes:
        if abs(num) >= threshold:
            formatted_num = num / threshold
            formatted_str = f"{formatted_num:.{decimals}f}".rstrip('0').rstrip('.')
            return f"{formatted_str}{suffix}"
    return str(num)


def update_stock_prices(conn):
    c = conn.cursor()
    now = datetime.datetime.now()

    stocks = c.execute("SELECT stock_id, price, last_updated, change_rate, open_price, close_price FROM stocks").fetchall()

    for stock_id, current_price, last_updated, change_rate, open_price, close_price in stocks:
        try:
            if last_updated:
                last_updated = datetime.datetime.strptime(last_updated, "%Y-%m-%d %H:%M:%S")
            else:
                last_updated = now - datetime.timedelta(seconds=10)

            if open_price is None:
                open_price = current_price

            elapsed_time = (now - last_updated).total_seconds()
            num_updates = int(elapsed_time // 10)   # 🔹 every 10 seconds

            one_month_ago = now - datetime.timedelta(days=30)
            c.execute(
                "DELETE FROM stock_history WHERE stock_id = ? AND timestamp < ?",
                (stock_id, one_month_ago.strftime("%Y-%m-%d %H:%M:%S"))
            )

            if num_updates > 0:
                for i in range(num_updates):
                    change_percent = round(random.uniform(-change_rate, change_rate), 2)
                    new_price = max(1, round(current_price * (1 + change_percent / 100), 2))

                    missed_update_time = last_updated + datetime.timedelta(seconds=(i + 1) * 10)  # 🔹 every 10 seconds
                    if missed_update_time <= now:
                        c.execute(
                            "INSERT INTO stock_history (stock_id, price, timestamp) VALUES (?, ?, ?)",
                            (stock_id, new_price, missed_update_time.strftime("%Y-%m-%d %H:%M:%S"))
                        )
                    current_price = new_price

            close_price = current_price
            c.execute(
                "UPDATE stocks SET price = ?, open_price = ?, close_price = ?, last_updated = ? WHERE stock_id = ?",
                (current_price, open_price, close_price, now.strftime("%Y-%m-%d %H:%M:%S"), stock_id)
            )
        except Exception as e:
            print(f"Error updating stock {stock_id}: {e}")
            continue

    conn.commit()


def get_stock_metrics(conn, stock_id):
    c = conn.cursor()
    last_24_hours = datetime.datetime.now() - datetime.timedelta(days=1)

    result = c.execute(
        """
        SELECT MIN(price), MAX(price), price 
        FROM stock_history 
        WHERE stock_id = ? AND timestamp >= ?
        """,
        (stock_id, last_24_hours.strftime("%Y-%m-%d %H:%M:%S"))
    ).fetchone()

    low_24h, high_24h, last_price = result if result else (None, None, None)

    c.execute(
        """
        SELECT MIN(price), MAX(price) 
        FROM stock_history 
        WHERE stock_id = ?
        """,
        (stock_id,)
    )
    result = c.fetchone()
    all_time_low, all_time_high = result if result else (None, None)

    c.execute(
        """
        SELECT price 
        FROM stock_history 
        WHERE stock_id = ? AND timestamp >= ?
        ORDER BY timestamp ASC
        LIMIT 1
        """,
        (stock_id, last_24_hours.strftime("%Y-%m-%d %H:%M:%S"))
    )
    price_24h_ago = c.fetchone()
    price_24h_ago = price_24h_ago[0] if price_24h_ago else last_price

    delta_24h_high = last_price - high_24h if high_24h else None
    delta_24h_low = last_price - low_24h if low_24h else None
    delta_all_time_high = last_price - all_time_high if all_time_high else None
    delta_all_time_low = last_price - all_time_low if all_time_low else None
    price_change_percent = ((last_price - price_24h_ago) / price_24h_ago * 100) if price_24h_ago else 0

    return {
        "low_24h": low_24h,
        "high_24h": high_24h,
        "all_time_low": all_time_low,
        "all_time_high": all_time_high,
        "price_change": price_change_percent,
        "delta_24h_high": delta_24h_high,
        "delta_24h_low": delta_24h_low,
        "delta_all_time_high": delta_all_time_high,
        "delta_all_time_low": delta_all_time_low,
    }


def adjust_stock_prices(conn, stock_id, quantity, action):
    c = conn.cursor()
    price, stock_amount = c.execute("SELECT price, stock_amount FROM stocks WHERE stock_id = ?", (stock_id,)).fetchone()

    elasticity_factor = 1
    if action == "buy":
        price_change = (quantity / stock_amount) * elasticity_factor * price
    elif action == "sell":
        price_change = -(quantity / stock_amount) * elasticity_factor * price
    else:
        price_change = 0

    new_price = price + price_change
    c.execute("UPDATE stocks SET price = ? WHERE stock_id = ?", (new_price, stock_id))
    conn.commit()
    st.rerun()


def buy_stock(conn, user_id, stock_id, quantity):
    c = conn.cursor()
    price, symbol = c.execute("SELECT price, symbol FROM stocks WHERE stock_id = ?", (stock_id,)).fetchone()
    balance = c.execute("SELECT balance FROM users WHERE user_id = ?", (user_id,)).fetchone()[0]
    cost = price * quantity

    if balance < cost:
        st.toast("Insufficient funds.")
        return

    c.execute("UPDATE users SET balance = balance + ? WHERE username = 'Government'", (cost,))
    c.execute("UPDATE users SET balance = balance - ? WHERE user_id = ?", (cost, user_id))

    existing = c.execute(
        "SELECT quantity, avg_buy_price FROM user_stocks WHERE user_id = ? AND stock_id = ?",
        (user_id, stock_id)
    ).fetchone()

    if existing:
        old_quantity = existing[0]
        old_avg_price = existing[1]
        new_quantity = old_quantity + quantity
        new_avg_price = ((old_quantity * old_avg_price) + (quantity * price)) / new_quantity
        c.execute(
            "UPDATE user_stocks SET quantity = ?, avg_buy_price = ? WHERE user_id = ? AND stock_id = ?",
            (new_quantity, new_avg_price, user_id, stock_id)
        )
    else:
        c.execute(
            "INSERT INTO user_stocks (user_id, stock_id, quantity, avg_buy_price) VALUES (?, ?, ?, ?)",
            (user_id, stock_id, quantity, price)
        )

    c.execute(
        "INSERT INTO transactions (transaction_id, user_id, type, amount, stock_id, quantity, timestamp) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
        (random.randint(100000000000, 999999999999), user_id, f"Buy Stock ({symbol})", cost, stock_id, quantity)
    )
    c.execute("UPDATE stocks SET stock_amount = stock_amount - ? WHERE stock_id = ?", (quantity, stock_id))
    adjust_stock_prices(conn, stock_id, quantity, "buy")

    conn.commit()
    st.toast(f"Purchased :blue[{format_number(quantity)}] shares for :green[${format_number(cost, 2)}]")


def sell_stock(conn, user_id, stock_id, quantity):
    c = conn.cursor()
    price, symbol = c.execute("SELECT price, symbol FROM stocks WHERE stock_id = ?", (stock_id,)).fetchone()
    user_stock = c.execute(
        "SELECT quantity, avg_buy_price FROM user_stocks WHERE user_id = ? AND stock_id = ?",
        (user_id, stock_id)
    ).fetchone()

    new_quantity = user_stock[0] - quantity
    profit = price * quantity
    tax = (profit / 100) * 0.05
    net_profit = profit - tax

    if new_quantity == 0:
        c.execute("DELETE FROM user_stocks WHERE user_id = ? AND stock_id = ?", (user_id, stock_id))
        c.execute("UPDATE stocks SET stock_amount = stock_amount + ? WHERE stock_id = ?", (quantity, stock_id))
    else:
        c.execute("UPDATE user_stocks SET quantity = ? WHERE user_id = ? AND stock_id = ?",
                  (new_quantity, user_id, stock_id))
        c.execute("UPDATE stocks SET stock_amount = stock_amount + ? WHERE stock_id = ?", (quantity, stock_id))

    c.execute(
        "INSERT INTO transactions (transaction_id, user_id, type, amount, stock_id, quantity) VALUES (?, ?, ?, ?, ?, ?)",
        (random.randint(100000000000, 999999999999), user_id, f"Sell Stock ({symbol})", net_profit, stock_id, quantity)
    )
    c.execute("UPDATE users SET balance = balance - ? WHERE username = 'Government'", (profit,))
    c.execute("UPDATE users SET balance = balance + ? WHERE user_id = ?", (net_profit, user_id))
    adjust_stock_prices(conn, stock_id, quantity, "sell")

    conn.commit()
    st.toast(f"Sold :blue[{format_number(quantity)}] shares for :green[${format_number(net_profit, 2)}]")


def login_register_view(conn):
    st.title("Stocks Login")
    c = conn.cursor()
    controller = CookieController()
    tab_login, tab_register = st.tabs(["Login", "Register"])

    with tab_login:
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Sign In", type="primary"):
            row = c.execute("SELECT user_id, password FROM users WHERE username=?", (username,)).fetchone()
            if row and verifyPass(row[1], password):
                st.session_state.user_id = row[0]
                st.session_state.username = username
                controller.set("user_id", str(row[0]))
                controller.set("username", username)
                st.rerun()
            else:
                st.error("Invalid credentials.")

    with tab_register:
        new_username = st.text_input("New Username", key="reg_username")
        new_password = st.text_input("New Password", type="password", key="reg_password")
        if st.button("Create Account"):
            if not new_username or not new_password:
                st.warning("Please provide a username and password.")
            elif len(new_password) < 8:
                st.warning("Password must be at least 8 characters.")
            else:
                try:
                    c.execute(
                        "INSERT INTO users (username, password, balance) VALUES (?, ?, ?)",
                        (new_username, hashPass(new_password), 1000.0),
                    )
                    conn.commit()
                    st.success("Account created. You can sign in now.")
                except sqlite3.IntegrityError:
                    st.error("Username already exists.")

def stocks_view(conn, user_id):
    c = conn.cursor()

    update_stock_prices(conn)
    st_autorefresh(interval=10000, key="stock_autorefresh")

    stocks = c.execute("SELECT stock_id, name, symbol, price, stock_amount, dividend_rate FROM stocks").fetchall()
    balance = c.execute("SELECT balance FROM users WHERE user_id = ?", (user_id,)).fetchone()[0]

    if "selected_game_stock" not in st.session_state and stocks:
        st.session_state.selected_game_stock = stocks[0][0]
    if "ti" not in st.session_state:
        st.session_state.ti = 1
    if "graph_color" not in st.session_state:
        st.session_state.graph_color = (0, 255, 0)
    if "hours" not in st.session_state:
        st.session_state.hours = 168
    if "resample" not in st.session_state:
        st.session_state.resample = 1
    if "selected_real_stock" not in st.session_state:
        st.session_state.selected_real_stock = "AAPL"

    user_balance = c.execute("SELECT balance, username FROM users WHERE user_id=?", (user_id,)).fetchone()
    if user_balance:
        bal, disp_user = user_balance
        c21, c22 = st.columns([4, 1])
        c21.markdown(f"### Wallet &nbsp; :green[$ {format_number(bal, 2)}]", unsafe_allow_html=True)
        if c22.button("Leaderboard", use_container_width=True, type="secondary"):
            st.session_state.page = "leaderboard"
            st.rerun()

    stock_ticker_html = """
    <div style="white-space: nowrap; overflow: hidden; background-color:; color: white; padding: 10px; font-size: 20px;">
        <marquee behavior="scroll" direction="left" scrollamount="5">
    """

    now = datetime.datetime.now()
    start_time = now - datetime.timedelta(hours=24)
    start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")

    for stock_id, name, symbol, current_price, amt, dividend in stocks:
        price_24h_ago = c.execute(
            """
            SELECT price FROM stock_history 
            WHERE stock_id = ? AND timestamp <= ? 
            ORDER BY timestamp DESC LIMIT 1
            """,
            (stock_id, start_time_str)
        ).fetchone()

        if price_24h_ago:
            price_24h_ago = price_24h_ago[0]
            price_color = "lime" if current_price >= price_24h_ago else "red"
        else:
            price_color = "white"

        stock_ticker_html += f" <span style='color: white;'>{symbol}</span> <span style='color: {price_color};'>${format_number(current_price, 2)}</span> <span style='color: darkgray'> | </span>"

    stock_ticker_html += "</marquee></div>"
    st.markdown(stock_ticker_html, unsafe_allow_html=True)

    if not stocks:
        st.info("No stocks available.")
        return

    selected_stock = next(s for s in stocks if s[0] == st.session_state.selected_game_stock)
    stock_id, name, symbol, price, stock_amount, dividend = selected_stock

    now = datetime.datetime.now()
    start_time = now - datetime.timedelta(hours=st.session_state.hours)
    start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")

    history = c.execute(
        """
        SELECT timestamp, price FROM stock_history 
        WHERE stock_id = ? AND timestamp >= ?
        ORDER BY timestamp ASC
        """,
        (stock_id, start_time_str)
    ).fetchall()

    if len(history) > 1:
        last_price = history[-1][1]
        previous_price = history[-2][1]
        percentage_change = ((last_price - previous_price) / previous_price) * 100
        if last_price > previous_price:
            change_color = f":green[+{format_number(percentage_change)}%] :gray[(7d)]"
            st.session_state.graph_color = (0, 255, 0)
        elif last_price < previous_price:
            change_color = f":red[{format_number(percentage_change)}%] :gray[(7d)]"
            st.session_state.graph_color = (255, 0, 0)
        else:
            change_color = f":orange[0.00%] :gray[(7d)]"
            st.session_state.graph_color = (255, 255, 0)
    else:
        percentage_change = 0
        change_color = ":orange[0.00%] :gray[(7d)]"

    if len(history) > 1:
        last_price = history[-1][1]
        previous_price = history[-2][1]
        percentage_change = ((last_price - previous_price) / previous_price) * 100
        if last_price > previous_price:
            change_color = f":green[+{format_number(percentage_change)}%] :gray[(24h)"
            st.session_state.graph_color = (0, 255, 0)
        elif last_price < previous_price:
            change_color = f":red[{format_number(percentage_change)}%] :gray[(24h)"
            st.session_state.graph_color = (255, 0, 0)
        else:
            change_color = f":orange[0.00%] :gray[(24h)"
            st.session_state.graph_color = (255, 255, 0)
    else:
        percentage_change = 0
        change_color = ":orange[0.00%] :gray[(24h)"

    cols = st.columns(len(stocks))
    for i in range(len(stocks)):
        with cols[i]:
            if st.button(label=f"{stocks[i][2]}", key=stocks[i][0], use_container_width=True):
                st.session_state.selected_game_stock = stocks[i][0]
                st.rerun()

    c1, c2 = st.columns([2.1, 1.5])
    with c1:
        if len(history) > 1:
            df = pd.DataFrame(history, columns=["Timestamp", "Price"])
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            df.set_index("Timestamp", inplace=True)
            df_resampled = df.resample(f"{st.session_state.resample}h").ohlc()['Price'].dropna()
            candlestick_data = [
                {
                    "time": int(timestamp.timestamp()),
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                }
                for timestamp, row in df_resampled.iterrows()
            ]
            chartOptions = {
                "layout": {
                    "textColor": 'rgba(180, 180, 180, 1)',
                    "background": {"type": 'solid', "color": 'rgb(15, 17, 22)'},
                },
                "grid": {
                    "vertLines": {"color": "rgba(30, 30, 30, 0.7)"},
                    "horzLines": {"color": "rgba(30, 30, 30, 0.7)"},
                },
                "crosshair": {"mode": 0},
                "watermark": {
                    "visible": True,
                    "fontSize": 70,
                    "horzAlign": 'center',
                    "vertAlign": 'center',
                    "color": 'rgba(50, 50, 50, 0.2)',
                    "text": 'GenStonks',
                },
            }
            seriesCandlestickChart = [{
                "type": 'Candlestick',
                "data": candlestick_data,
                "options": {
                    "upColor": '#26a69a',
                    "downColor": '#ef5350',
                    "borderVisible": False,
                    "wickUpColor": '#26a69a',
                    "wickDownColor": '#ef5350',
                },
            }]
            renderLightweightCharts([{"chart": chartOptions, "series": seriesCandlestickChart}], 'candlestick')
        else:
            st.info("Stock history will be available after 60 seconds of stock creation.")

        q1, q2, q3, q4, q5, q6, q7, q8, q9 = st.columns(9)
        if q1.button("1m", type="tertiary", use_container_width=True):
            st.session_state.resample = "0.01"
            st.session_state.hours = 0.1
            st.rerun()
        if q2.button("30m", type="tertiary", use_container_width=True):
            st.session_state.resample = "0.05"
            st.session_state.hours = 0.5
            st.rerun()
        if q3.button("1h", type="tertiary", use_container_width=True):
            st.session_state.resample = "0.01"
            st.session_state.hours = 1
            st.rerun()
        if q4.button("5h", type="tertiary", use_container_width=True):
            st.session_state.resample = "0.05"
            st.session_state.hours = 1440
            st.rerun()
        if q5.button("1d", type="tertiary", use_container_width=True):
            st.session_state.resample = "0.1"
            st.session_state.hours = 1440
            st.rerun()
        if q6.button("7d", type="tertiary", use_container_width=True):
            st.session_state.resample = "0.5"
            st.session_state.hours = 1440
            st.rerun()
        if q7.button("15d", type="tertiary", use_container_width=True):
            st.session_state.resample = "1"
            st.session_state.hours = 1440
            st.rerun()
        if q8.button("1mo", type="tertiary", use_container_width=True):
            st.session_state.resample = "5"
            st.session_state.hours = 1440
            st.rerun()
        if q9.button("MAX", type="tertiary", use_container_width=True):
            st.session_state.resample = "10"
            st.session_state.hours = 1440
            st.rerun()

        now = datetime.datetime.now()
        days_ahead = (6 - now.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        next_sunday = now + datetime.timedelta(days=days_ahead)
        next_sunday_midnight = next_sunday.replace(hour=0, minute=0, second=0, microsecond=0)
        time_left = next_sunday_midnight - now
        days = time_left.days
        hours, remainder = divmod(time_left.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        with st.container(border=True):
            st.write(f"Next Dividend Payout In :orange[{days}] Day, :orange[{hours}] Hours, :orange[{minutes}] Minutes.")

    with c2:
        st.subheader(f"{name} ({symbol})")
        st.header(f":green[${format_number_with_dots(round(price, 2))}] \n {change_color}]")
        user_stock = c.execute(
            "SELECT quantity, avg_buy_price FROM user_stocks WHERE user_id = ? AND stock_id = ?",
            (user_id, stock_id)
        ).fetchall()

        if user_stock:
            user_quantity = user_stock[0][0] if user_stock[0][0] else 0
            avg_price = user_stock[0][1] if user_stock[0][1] else 0
        else:
            user_quantity = 0
            avg_price = 0

        with st.container(border=True):
            st.markdown(f"**Holding** &nbsp; :blue[{format_number(user_quantity, 2)} {symbol}] ~ :green[${format_number(user_quantity * price, 2)}]", unsafe_allow_html=True)
            st.markdown(f"**AVG. Bought At** &nbsp; :green[${format_number(avg_price, 2)}]", unsafe_allow_html=True)
            ca1, ca2 = st.columns(2)
            ca1.markdown(f"**Available** &nbsp; :orange[{format_number(stock_amount, 2)} {symbol}]", unsafe_allow_html=True)
            ca2.markdown(f"**Dividend Ratio** &nbsp; :orange[%{round(dividend * 100, 2)}]", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            buy_max_quantity = min(balance / price, stock_amount)
            buy_quantity = st.number_input(f"Buy {symbol}", min_value=0.0, step=0.25, key=f"buy_{stock_id}")
            st.write(f"[Cost]  :red[${format_number(buy_quantity * price)}]")
            if st.button(
                f"Buy {symbol}", key=f"buy_btn_{stock_id}", type="primary", use_container_width=True,
                disabled=True if buy_quantity == 0 or stock_amount < buy_quantity else False,
                help="Not enough stock available in the market" if stock_amount < buy_quantity else None,
            ):
                with st.spinner("Purchasing..."):
                    time.sleep(1.5)
                    buy_stock(conn, user_id, stock_id, buy_quantity)
                time.sleep(1)
                st.rerun()
            if st.button(
                f"Buy MAX: :orange[{format_number(buy_max_quantity)}] ~ :green[${format_number(balance)}]",
                key=f"buy_max_btn_{stock_id}", use_container_width=True,
            ):
                with st.spinner("Purchasing..."):
                    time.sleep(1.5)
                    buy_stock(conn, user_id, stock_id, buy_max_quantity)
                time.sleep(1)
                st.rerun()

        with col2:
            sell_quantity = st.number_input(
                f"Sell {symbol}", min_value=0.0, max_value=float(user_quantity), step=0.25, key=f"sell_{stock_id}"
            )
            tax = ((sell_quantity * price) / 100) * 0.05
            net_profit = (sell_quantity * price) - tax
            st.write(f"[Profit] :green[${format_number(net_profit)}] | :red[${format_number(tax)}] [Capital Tax]")
            if st.button(
                f"Sell {symbol}", key=f"sell_btn_{stock_id}", use_container_width=True,
                disabled=True if sell_quantity == 0 else False,
            ):
                with st.spinner("Selling..."):
                    time.sleep(1.5)
                    sell_stock(conn, user_id, stock_id, sell_quantity)
                time.sleep(1)
                st.rerun()
            if st.button(
                f"Sell MAX: :orange[{format_number(user_quantity)}] ~ :green[${format_number(user_quantity * price)}]",
                key=f"sell_max_btn_{stock_id}", use_container_width=True, disabled=True if not user_quantity else False,
            ):
                with st.spinner("Selling..."):
                    time.sleep(1.5)
                    sell_stock(conn, user_id, stock_id, user_quantity)
                time.sleep(1)
                st.rerun()

        stock_metrics = get_stock_metrics(conn, stock_id)
        stock_volume = c.execute(
            "SELECT SUM(quantity) FROM transactions WHERE stock_id = ? AND timestamp >= DATETIME('now', '-24 hours')",
            (stock_id,)
        ).fetchone()[0]
        if not stock_volume:
            stock_volume = 0

    st.text("")
    st.text("")
    st.subheader("Metrics", divider="green")
    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
    col1.write("24h HIGH")
    col1.write(f"#### :green[${format_number(stock_metrics['high_24h'])}]" if stock_metrics['high_24h'] else "N/A")
    col2.write("24h LOW")
    col2.write(f"#### :red[${format_number(stock_metrics['low_24h'])}]" if stock_metrics['low_24h'] else "N/A")
    col3.write("All Time High")
    col3.write(f"#### :green[${format_number(stock_metrics['all_time_high'])}]" if stock_metrics['all_time_high'] else "N/A")
    col4.write("All Time Low")
    col4.write(f"#### :red[${format_number(stock_metrics['all_time_low'])}]" if stock_metrics['all_time_low'] else "N/A")
    col5.write("24h Change")
    col5.write(f"#### :orange[%{format_number(stock_metrics['price_change'], 2)}]")
    col6.write("24h Volume")
    col6.write(f"#### :blue[{format_number(stock_volume)}]")
    col7.write("Volatility Index")
    if stock_metrics['all_time_low']:
        col7.write(f"#### :violet[{format_number(((stock_metrics['all_time_high'] - stock_metrics['all_time_low']) / stock_metrics['all_time_low']) * 100)} σ]")
    else:
        col7.write("N/A")
    col8.write("Market Cap")
    col8.write(f"#### :green[${format_number(stock_amount * price)}]")

    st.text("")
    st.text("")
    st.text("")

    leaderboard_data = []
    selected_stock_id = st.session_state.selected_game_stock
    stockholders = c.execute(
        """
        SELECT u.username, SUM(us.quantity) AS total_quantity
        FROM user_stocks us
        JOIN users u ON us.user_id = u.user_id
        WHERE us.stock_id = ?
        GROUP BY us.user_id
        ORDER BY total_quantity ASC
        """,
        (selected_stock_id,)
    ).fetchall()
    for stockholder in stockholders:
        username = stockholder[0]
        total_quantity = stockholder[1]
        leaderboard_data.append([username, total_quantity])

    st.subheader("🏆 Stockholder Leaderboard", divider="green")
    if leaderboard_data:
        leaderboard_df = pd.DataFrame(leaderboard_data, columns=["Stockholder", "Shares Held"])
        st.dataframe(leaderboard_df, use_container_width=True)
    else:
        st.info("No stockholder data available yet.")

def admin_view(conn):
    st.title("Admin Panel")
    c = conn.cursor()

    st.subheader("Add New Stock")
    col1, col2, col3 = st.columns(3)
    with col1:
        new_name = st.text_input("Name")
        new_symbol = st.text_input("Symbol")
    with col2:
        new_price = st.number_input("Price", min_value=0.0, value=100.0, step=0.01)
        new_amount = st.number_input("Available Amount", min_value=0.0, value=10000.0, step=1.0)
    with col3:
        new_dividend = st.number_input("Dividend Rate", min_value=0.0, value=0.0, step=0.001)
        new_change = st.number_input("Change Rate", min_value=0.01, value=0.5, step=0.01)
    if st.button("Add Stock", type="primary"):
        try:
            # stock_id uses INTEGER PRIMARY KEY; let SQLite assign ROWID or compute next id
            c.execute(
                "INSERT INTO stocks (name, symbol, price, stock_amount, dividend_rate, change_rate, open_price, close_price, last_updated) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (new_name, new_symbol.upper(), float(new_price), float(new_amount), float(new_dividend), float(new_change), new_price, new_price, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            )
            conn.commit()
            st.success("Stock added.")
            st.rerun()
        except sqlite3.IntegrityError:
            st.error("Symbol already exists.")

    st.divider()
    st.subheader("Edit Stocks")
    df = pd.read_sql_query(
        "SELECT stock_id, name, symbol, price, stock_amount, dividend_rate, change_rate, open_price, close_price FROM stocks ORDER BY stock_id",
        conn,
    )
    edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True, key="stocks_editor")
    if st.button("Save Changes", type="primary"):
        # Upsert edited rows back to DB
        for _, row in edited_df.iterrows():
            c.execute(
                """
                UPDATE stocks
                SET name=?, symbol=?, price=?, stock_amount=?, dividend_rate=?, change_rate=?, open_price=?, close_price=?
                WHERE stock_id=?
                """,
                (
                    row["name"], row["symbol"].upper(), float(row["price"]), float(row["stock_amount"]), float(row["dividend_rate"]), float(row["change_rate"]),
                    float(row["open_price"]) if pd.notna(row["open_price"]) else None,
                    float(row["close_price"]) if pd.notna(row["close_price"]) else None,
                    int(row["stock_id"]),
                ),
            )
        conn.commit()
        st.success("Changes saved.")
        st.rerun()

    st.divider()
    st.subheader("Delete Stock")
    stocks_list = c.execute("SELECT stock_id, symbol FROM stocks ORDER BY symbol").fetchall()
    if stocks_list:
        del_map = {f"{sym} (id={sid})": sid for sid, sym in stocks_list}
        to_delete_label = st.selectbox("Select stock to delete", options=list(del_map.keys()))
        if st.button("Delete", type="secondary"):
            sid = del_map[to_delete_label]
            # Cascade delete dependent rows
            c.execute("DELETE FROM stock_history WHERE stock_id=?", (sid,))
            c.execute("DELETE FROM user_stocks WHERE stock_id=?", (sid,))
            c.execute("DELETE FROM transactions WHERE stock_id=?", (sid,))
            c.execute("DELETE FROM stocks WHERE stock_id=?", (sid,))
            conn.commit()
            st.success("Stock deleted.")
            st.rerun()
    else:
        st.info("No stocks to delete.")


def leaderboard_view(conn):
    st.header("Ranking", divider="green")
    c = conn.cursor()

    # Compute total wealth = balance + sum(quantity * price) across holdings
    # Fetch balances
    users = c.execute("SELECT user_id, username, balance FROM users").fetchall()
    user_map = {uid: {"username": uname, "balance": bal, "stocks": 0.0} for uid, uname, bal in users}

    # Aggregate stock holdings value using current prices
    rows = c.execute(
        """
        SELECT us.user_id, us.quantity, s.price
        FROM user_stocks us
        JOIN stocks s ON s.stock_id = us.stock_id
        """
    ).fetchall()
    for uid, qty, price in rows:
        if uid in user_map:
            user_map[uid]["stocks"] += (qty or 0.0) * (price or 0.0)

    totals = []
    for uid, info in user_map.items():
        total = (info["balance"] or 0.0) + (info["stocks"] or 0.0)
        totals.append((info["username"], total))

    if not totals:
        st.info("No users found.")
        return

    totals.sort(key=lambda x: x[1], reverse=True)

    # Styling similar to main.py
    st.markdown('''
        <style>
            .leaderboard-frame { border-radius: 10px; padding: 10px; margin: 5px 0; text-align: center; font-weight: bold; color: white; }
            .leaderboard-row { display: flex; justify-content: space-between; align-items: center; width: 100%; }
            .left { text-align: left; flex-grow: 1; }
            .right { text-align: right; min-width: 120px; color: rgb(50, 218, 125) }
            .first { border: 1px solid gold; font-size: 30px; padding: 20px; background-color: transparent; }
            .second { border: 1px solid silver; font-size: 25px; padding: 15px; background-color: transparent; }
            .third { border: 1px solid #cd7f32; font-size: 22px; padding: 12px; background-color: transparent; }
            .other { border: 1px solid #444; font-size: 15px; padding: 8px; background-color: transparent; }
        </style>
    ''', unsafe_allow_html=True)

    medals = ["🥇", "🥈", "🥉"]
    for idx, (username, total) in enumerate(totals, start=1):
        medal = medals[idx - 1] if idx <= 3 else ""
        class_name = "first" if idx == 1 else "second" if idx == 2 else "third" if idx == 3 else "other"
        st.markdown(
            f'''
            <div class="leaderboard-frame {class_name}">
                <div class="leaderboard-row">
                    <span class="left">{medal} {idx}. {username}</span>
                    <span class="right">${total:,.2f}</span>
                </div>
            </div>
            ''',
            unsafe_allow_html=True,
        )


def main():
    st.set_page_config(page_title="Stocks", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
    inject_inter_font()
    conn = get_db_connection()
    init_db(conn)

    # Restore session from cookies if present
    controller = CookieController()
    try:
        if controller.get("user_id"):
            st.session_state.user_id = int(controller.get("user_id"))
            st.session_state.username = controller.get("username")
    except Exception:
        pass

    # Sidebar: user credentials and logout
    with st.sidebar:
        st.header("Account")
        if st.session_state.get("user_id"):
            c = conn.cursor()
            uid = st.session_state.user_id
            username, balance = c.execute(
                "SELECT username, balance FROM users WHERE user_id=?",
                (uid,)
            ).fetchone()
            st.write(f"User: @{username}")
            # Admin Panel button only for 'admin'
            if username == "admin":
                if st.button("Admin Panel"):
                    st.session_state.page = "admin"
                    st.rerun()
            if st.button("Log out", use_container_width=True):
                controller.remove("user_id")
                controller.remove("username")
                for k in ["user_id", "username"]:
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()
        else:
            st.info("Not signed in")

    # Main routing
    if not st.session_state.get("user_id"):
        login_register_view(conn)
        return

    # Route to admin if selected and user is admin
    if st.session_state.get("page") == "admin" and st.session_state.get("username") == "admin":
        with st.sidebar:
            if st.button("Back to Stocks"):
                st.session_state.page = "stocks"
                st.rerun()
        admin_view(conn)
        return

    # Route to leaderboard if selected
    if st.session_state.get("page") == "leaderboard":
        with st.sidebar:
            if st.button("Back to Stocks"):
                st.session_state.page = "stocks"
                st.rerun()
        leaderboard_view(conn)
        return

    stocks_view(conn, st.session_state.user_id)


if __name__ == "__main__":
    main()
