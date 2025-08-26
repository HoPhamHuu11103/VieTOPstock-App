# VieTOPstock-App

**Overview**

VieTOPstock-App is a personal project developed to analyze and optimize investment portfolios on the Vietnamese stock market. This web application, built with Streamlit, Python, and the vnstock API, integrates stock data analysis, portfolio optimization, and extended features to support investors, such as news tracking, financial report monitoring, and stock price visualization charts.

**Link app:** https://vietopstock-app-110103.streamlit.app/

**Features**
- **Stock Data Fetching:** Retrieve historical stock data from 2020 to 2024 via the vnstock API.
- **Portfolio Optimization:** Implement three optimization algorithms:

SLSQP (Sequential Least Squares Programming)

SGD-Volatility (Stochastic Gradient Descent minimizing volatility)

SGD-Sharpe (Stochastic Gradient Descent maximizing Sharpe Ratio)

Compare portfolio returns against the VN-Index benchmark.

- **Financial Analysis:** Generate detailed reports including balance sheets, income statements, cash flows, and financial ratios (yearly and quarterly).
- **Technical Analysis:** Support multiple indicators (SMA, EMA, RSI, MACD, Bollinger Bands, etc.) with customizable timeframes.
- **Data Visualization:** Interactive charts (line, pie, bar, heatmap) to display stock trends, portfolio performance, and correlations.
- **Company Insights:** Provide company profiles, shareholder details, insider deals, dividends, and news.

**Technologies Used**
- **Python Libraries:** Streamlit, Pandas, NumPy, SciPy, Plotly, vnstock
- **Frontend:** Custom CSS with Google Fonts (Poppins) for an enhanced UI
- **Data Source:** vnstock API (VCI and TCBS sources)
