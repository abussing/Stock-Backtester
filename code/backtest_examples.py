# %%
""" LOAD ALL THE HEADER_CODE """
from backtester import *

""" THIS SCRIPT IS FOR THE STRATEGY OF TRADING INTRADAY USING THE TICKERS WITH THE HIGHEST/LOWEST OVERNIGHT RETURN
 THAT PREVIOUS NIGHT """

Ntickers = 3


ETF_world = [
    "EWY",  # South Korea
    "EWA",  # Australia
    "EWU",  # United Kingdom
    "EWS",  # Singapore
    "EWQ",  # France
    "EWP",  # Spain
    "EWO",  # Austria
    "EWN",  # Netherlands
    "EWM",  # Malaysia
    "EWL",  # Switzerland
    "EWK",  # Belgium
    "EWJ",  # Japan
    "EWI",  # Italy
    "EWH",  # Hong Kong
    "EWD",  # Sweden
    "EWG",  # Germany
    "EWT",  # Taiwan
    "EZA",  # South Africa
    "EWZ",  # Brazil
    "EWW",  # Mexico
    "EWC",  # Canada
]


# %%
""" PULLING THE DATA FROM YFINANCE """

backtest_start = dt.datetime(2015, 2, 4)
backtest_end = dt.datetime(2024, 8, 10)

ETFworld_obj = yfinance_query(
    tick_list=ETF_world, start_date=backtest_start, end_date=backtest_end,
    pickle_path="C:/Users/abussing/Desktop/backtester/pickles/yfinance_file_format"
)


# %%
""" MAKE THE ENTRY AND EXIT FILTER OBJECTS """

simple_entry = EntryFilter(ETFworld_obj)
simple_exit = ExitFilter(ETFworld_obj)

# %%
""" FIRST WE'LL MAKE A PRELIMINARY 'STRATEGY' BUT WE'LL ONLY USE IT TO EXTRACT 'DAYS RETURN' COLUMN SO WE CAN RANK """

""" MAKE A STRATEGY SAYING GO LONG, ENTER AT OPEN, EXIT AT CLOSE. THIS IS ALL JUST TO EXTRACT THE RANKING FOR USE
IN THE ACTUAL STRATEGY STILL TO COME """

""" BACKTESTS GIVE AN ERROR UNLESS THERE IS SOME FILTER INVOLVED, SO WE'LL MAKE A POINTLESS VOLUME FILTER WITH A WEIGHTMETHOD"""
pointless_weight = VolumeFilter(minvolume=0)

overnight_strat = Strategy(
    pointless_weight, shorttrue=False, enteratopen=False, exitatclose=False
)

overnight_back = BackTest(simple_entry, simple_exit, overnight_strat)


""" LET'S EXTRACT THE trade_return COLUMN AND USE IT TO CREATE A FILTER """

chosen_df = overnight_back.apply_filters().copy()

overnight_returns = chosen_df["trade_return"]
overnight_returns.rename("overnight_return")

""" NOTICE THE daysago=1. THIS IS TO ENSURE WE USE THE OVERNIGHT FROM LAST NIGHT """
overnight_topN = ExternalValFilter(
    overnight_returns.copy(), "overnight_return", rank=Ntickers, daysago=1,
)


# %%
""" NOW WE CAN GO SHORT ON THESE TOP N TICKERS, ENTERING AT OPEN, EXITING AT CLOSE """
sharpeweight = SharpeFilter(weightmethod="pos_and_neg")

topNshort_day_strat = Strategy(
    sharpeweight, overnight_topN, shorttrue=True, enteratopen=True, exitatclose=True
)

topNshort_day_back = BackTest(simple_entry, simple_exit, topNshort_day_strat)

# %%
""" PLOT THE RETURNS """

topNshort_day_back.plot_returns("trade_return_weighted_cumulative")


# %%
""" NOW LET'S DO THE SAME THING BUT GOING LONG ON THE LOWEST N RETURNING TICKERS OF THAT INTRADAY SPAN """
overnight_lowN = ExternalValFilter(
    overnight_returns.copy(), "overnight_return", rank=-Ntickers, daysago=1,
)

lowNlong_day_strat = Strategy(
    sharpeweight, overnight_lowN, shorttrue=False, enteratopen=True, exitatclose=True
)

lowNlong_day_back = BackTest(simple_entry, simple_exit, lowNlong_day_strat)


# %%
""" PLOT THE RETURNS """

lowNlong_day_back.plot_returns("trade_return_weighted_cumulative")


# %%
""" MAKE THE COMBO """

lowhigh_nightreturn_intraday_combo = ComboResult(topNshort_day_back, lowNlong_day_back)

lowhigh_nightreturn_intraday_combo.plot_returns("trade_return_weighted_cumulative")

# %%
