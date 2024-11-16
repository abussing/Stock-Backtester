import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import yfinance
import os


orig_dir = os.getcwd()

# =============================================================================
#  THIS RawData CLASS IS USED TO PUT THE VARIETY OF DIFFERENT LOOKING DATA SOURCES
#  INTO A COMMON FORM THAT WILL BE USED BY THIS SCRIPT. IT'S NECESSARY TO CONVERT
#  ALL DATA SOURCES TO THIS RawData SOURCE BEFORE STARTING.
# =============================================================================
class RawData:
    converted_cols = [
        "date",
        "ticker",
        "open",
        "close",
        "low",
        "high",
        "volume",
        "divydend",
        "splyt",
    ]

    def __init__(
        self,
        df_toconvert,
        datecol_name="date",
        tickercol_name="ticker",
        opencol_name="open",
        closecol_name="close",
        lowcol_name="low",
        highcol_name="high",
        volumecol_name="volume",
        dividendcol_name="dividend",
        splitcol_name="split",
    ):

        self.original_cols = [
            datecol_name,
            tickercol_name,
            opencol_name,
            closecol_name,
            lowcol_name,
            highcol_name,
            volumecol_name,
            dividendcol_name,
            splitcol_name,
        ]

        self.original_df = df_toconvert
        self.converted_df = self.convert()

    def convert(self):
        # FIRST CUT OUT THE COLUMNS OF THE ENTERED DATAFRAME THAT WE DON'T CARE ABOUT
        df_converted = self.original_df.loc[
            :, self.original_df.columns.intersection(self.original_cols)
        ].copy()
        # NOW RENAME THE COLUMNS OF THE ENTERED DATAFRAME WITH STANDARD NAMES THAT WE'LL USE IN ALL PARTS OF THIS SCRIPT
        df_converted.rename(
            columns=dict(zip(self.original_cols, self.converted_cols)), inplace=True
        )
        # NOW CONVERT THE DATE COLUMN INTO THE DATETIME FORMAT
        df_converted["date"] = pd.to_datetime(df_converted["date"]).dt.tz_localize(None)
        # NOW ORDER BY DATE (EARLIEST TO LATEST)
        df_converted.sort_values(by="date", inplace=True)
        # CREATE MULTI-INDEX WITH DATE AND TICKER
        df_converted.set_index(["ticker", "date"], inplace=True)
        return df_converted



## IF YOU'D LIKE TO USE YFINANCE QUERY
def yfinance_query(tick_list, start_date, end_date, pickle_path):

    os.chdir(orig_dir)
    
    os.chdir(pickle_path)

    def onetick_fun(name, start_date, end_date):
        tempdf = yfinance.Ticker(name).history(
            start=start_date,
            end=end_date + dt.timedelta(hours=12),
            interval="1d",
            auto_adjust=False,
        )

        tempdf.index = pd.to_datetime(tempdf.index).tz_localize(None)
        
        # adding ticker column
        tempdf["ticker"] = name
        tempdf.set_index("ticker", append=True, inplace=True)

        return tempdf

    results = []
    for name in tick_list:
        change_count = 0
        if os.path.isfile(os.path.join(name)):
            tempdf = pd.read_pickle(os.path.join(name))
            # check to make sure pickled file contains all the dates in our range
            # fill in the dates it doesn't have
            have_min = tempdf.index.get_level_values(0).min()
            have_max = tempdf.index.get_level_values(0).max()



            # checking lower end
            # if start_date <= have_min - dt.timedelta(days=1):
            if have_min - start_date > dt.timedelta(days=2):
                toadd_df = onetick_fun(
                    name, min(start_date, have_min), have_min - dt.timedelta(days=1)
                )

            elif (
                (have_min - start_date <= dt.timedelta(days=2))
                & (have_min - start_date > dt.timedelta(days=0))
                & (have_min.weekday() != 0)
            ):
                toadd_df = onetick_fun(
                    name, min(start_date, have_min), have_min - dt.timedelta(days=1)
                )
            else:
                toadd_df = []

            if len(toadd_df) > 0:
                change_count += 1
                tempdf = pd.concat([tempdf, toadd_df])

            # checking upper end
            # if end_date >= have_max + dt.timedelta(days=1):
            if end_date - have_max > dt.timedelta(days=2):
                toadd_df = onetick_fun(
                    name, have_max + dt.timedelta(days=1), max(end_date, have_max)
                )
            elif (
                (end_date - have_max <= dt.timedelta(days=2))
                & (end_date - have_max > dt.timedelta(days=0))
                & (have_max.weekday() != 4)
            ):
                toadd_df = onetick_fun(
                    name, have_max + dt.timedelta(days=1), max(end_date, have_max)
                )
            else:
                toadd_df = []

            if len(toadd_df) > 0:
                change_count += 1
                tempdf = pd.concat([tempdf, toadd_df])

        else:
            change_count += 1
            tempdf = onetick_fun(name, start_date, end_date)

        if len(tempdf) > 0:
            if change_count > 0:
                # pickle and save it
                tempdf.to_pickle(os.path.join(name))

            # cut it down to size and return it
            tempdf = tempdf.loc[
                (tempdf.index.get_level_values(0).date >= start_date.date())
                & (tempdf.index.get_level_values(0).date <= end_date.date()),
                :,
            ]

            results.append(tempdf)

        else:
            print(str(name) + " returned 0 length dataframe")

    if len(results) > 0:
        # combining the different ticker dataframes into one dataframe
        concat_df = pd.concat(results)
        concat_df.reset_index(inplace=True)

        # putting the output dataframe into an object of the RawData class
        fromyfinance_df = RawData(
            concat_df,
            datecol_name="Date",
            tickercol_name="ticker",
            opencol_name="Open",
            closecol_name="Close",
            highcol_name="High",
            lowcol_name="Low",
            volumecol_name="Volume",
            dividendcol_name="Dividends",
            splitcol_name="Stock Splits",
        )

        return fromyfinance_df

    else:
        print("none of those tickers returned data")



# =============================================================================
# THESE ARE JUST HELPER FUNCTIONS USED THROUGHOUT THE SCRIPT
# =============================================================================

def drawdown_stats(date_price_series):
    # THIS FUNCTION FINDS MAX PERCENT PEAK TO VALLEY DRAWDOWN AND ALSO THE RECOVERY TIME
    price_col = date_price_series.name
    testdf = date_price_series.to_frame().copy()

    # FIRST WE'LL JUST CHECK IF THE INDEX HAS DATES IN IT
    if type(testdf.index.values[0]) == tuple:
        try:
            pd.to_datetime(testdf.index.levels[0])
        except:
            testdf.reset_index(level=0, drop=True, inplace=True)

    if type(testdf.index.values[0]) == tuple:
        try:
            pd.to_datetime(testdf.index.levels[1])
        except:
            testdf.reset_index(level=1, drop=True, inplace=True)
        else:
            raise TypeError("need for there to be dates in the index")

    else:
        try:
            pd.to_datetime(testdf.index)
        except:
            raise TypeError("need for there to be dates in the index")

    # NOW WE'LL FIND THE STATS WE'RE LOOKING FOR
    testdf["max_sofar"] = (
        testdf[price_col].rolling(window=len(testdf), min_periods=1).max()
    )
    testdf["price_reversed"] = list(testdf[price_col])[::-1]
    testdf["min_tocome"] = (
        testdf["price_reversed"].rolling(window=len(testdf), min_periods=1).min()
    )
    testdf["min_tocome"] = list(testdf["min_tocome"])[::-1]
    testdf = testdf.drop("price_reversed", axis=1)
    testdf["drawdown"] = testdf["max_sofar"] - testdf["min_tocome"]

    max_drawdown = testdf["drawdown"].max()
    max_sofar_atmaxdraw = testdf.loc[
        testdf["drawdown"] == max_drawdown, "max_sofar"
    ].values[0]
    drawdown_start = testdf[testdf["max_sofar"] == max_sofar_atmaxdraw].index[0]
    drawdown_end = testdf[testdf["max_sofar"] == max_sofar_atmaxdraw].index[-1]
    if drawdown_end == testdf.index[-1]:
        recovery_time = "hasn't fully recovered"
    else:
        recovery_time = (drawdown_end - drawdown_start).days

    # plt.figure(figsize=(10, 7))
    # plt.tight_layout()
    # plt.plot(testdf.index.values, testdf[price_col])
    # plt.plot(testdf.index.values, testdf["max_sofar"])
    # plt.plot(testdf.index.values, testdf["min_tocome"])

    # return testdf
    return {"max drawdown": max_drawdown, "recovery time": recovery_time}


def checkcols(df_attempted):
    if set(["open", "close", "low", "high", "volume", "divydend", "splyt"]).issubset(
        set(df_attempted.columns)
    ) and set(df_attempted.index.names).issubset(
        set(["date", "ticker", "Ticker", "Date"])
    ):
        pass
    else:
        raise ValueError(
            "one of (open,close,low,high,volume,divydend,splyt) is not present in df columns \n"
            + "or index is not (ticker, date)"
        )


def checkcols_secondary(df_attempted):
    if set(
        [
            "trade_return",
            "open",
            "close",
            "low",
            "high",
            "volume",
            "divydend",
            "splyt",
            "date_exit",
        ]
    ).issubset(set(df_attempted.columns)) and set(df_attempted.index.names).issubset(
        set(["date", "ticker", "Ticker", "Date"])
    ):
        pass
    else:
        raise ValueError(
            "one of (trade_return,open,close,low,high,volume,divydend,splyt,date_exit) is not present in df columns \n"
            + "or index is not (ticker, date)"
        )


def toflatlist(list_of_lists):
    return_list = list_of_lists.copy()
    while any([isinstance(x, list) for x in return_list]):
        for subitem in return_list:
            if isinstance(subitem, list):
                for item in subitem:
                    return_list.append(item)
                return_list.remove(subitem)
            else:
                pass
    return return_list


def subclass_find(*obj_list):
    return_list = []
    for combine_class in obj_list:
        if isinstance(combine_class, (AndFilter, OrFilter)):
            filter_use = list(combine_class.filterz)
            while any([isinstance(x, (AndFilter, OrFilter)) for x in filter_use]):
                for subfilt in filter_use:
                    if isinstance(subfilt, (AndFilter, OrFilter)):
                        filter_use = filter_use + list(subfilt.filterz)
                        filter_use.remove(subfilt)
        else:
            filter_use = combine_class

        return_list.append(filter_use)
    return toflatlist(return_list)


def rolling_calc(
    multiind_df,
    calc_column,
    groupby_column,
    rollon_column,
    rolling_days,
    mindays,
    calc_type,
):
    todelete = []
    if isinstance(rollon_column, int):
        multiind_df[
            str(multiind_df.index.names[rollon_column]) + "_extra"
        ] = multiind_df.index.get_level_values(rollon_column)
        rollon_column = str(multiind_df.index.names[rollon_column])
        rollon_column = rollon_column + "_extra"
        todelete.append(rollon_column)

    if isinstance(groupby_column, int):
        multiind_df[
            str(multiind_df.index.names[groupby_column]) + "_extra"
        ] = multiind_df.index.get_level_values(groupby_column)
        groupby_column = str(multiind_df.index.names[groupby_column])
        groupby_column = groupby_column + "_extra"
        todelete.append(groupby_column)

    multiind_df.sort_values(
        [rollon_column, "date"], ascending=[True, True], inplace=True
    )

    if calc_type == "mean":
        multiind_df["rolling_" + calc_column + "_mean"] = (
            multiind_df.sort_values([rollon_column, "date"])
            .groupby(groupby_column, group_keys=False)[[rollon_column, calc_column]]
            .rolling(rolling_days, mindays, on=rollon_column)
            .mean()
            .reset_index(level=0, drop=True)[calc_column]
        )
        multiind_df[
            "rolling_" + calc_column + "_mean" + "_daycombine"
        ] = multiind_df.groupby([groupby_column, rollon_column])[
            "rolling_" + calc_column + "_mean"
        ].transform(
            "last"
        )
        todelete.append("rolling_" + calc_column + "_mean")
    if calc_type == "std":
        multiind_df["rolling_" + calc_column + "_std"] = (
            multiind_df.sort_values([rollon_column, "date"])
            .groupby(groupby_column, group_keys=False)[[rollon_column, calc_column]]
            .rolling(rolling_days, mindays, on=rollon_column)
            .std()
            .reset_index(level=0, drop=True)[calc_column]
        )
        multiind_df[
            "rolling_" + calc_column + "_std" + "_daycombine"
        ] = multiind_df.groupby([groupby_column, rollon_column])[
            "rolling_" + calc_column + "_std"
        ].transform(
            "last"
        )
        todelete.append("rolling_" + calc_column + "_std")
    if calc_type == "negstd":
        multiind_df.loc[
            multiind_df[calc_column] < 0, "rolling_" + calc_column + "_negstd"
        ] = (
            multiind_df.loc[multiind_df[calc_column] < 0, :]
            .sort_values([rollon_column, "date"])
            .groupby(groupby_column, group_keys=False)[[rollon_column, calc_column]]
            .rolling(rolling_days, mindays, on=rollon_column)
            .std()
            .reset_index(level=0, drop=True)[calc_column]
        )
        multiind_df[
            "rolling_" + calc_column + "_negstd" + "_daycombine"
        ] = multiind_df.groupby([groupby_column, rollon_column])[
            "rolling_" + calc_column + "_negstd"
        ].transform(
            "last"
        )
        todelete.append("rolling_" + calc_column + "_negstd")
    if calc_type == "count":
        multiind_df["rolling_" + calc_column + "_count"] = (
            multiind_df.sort_values([rollon_column, "date"])
            .groupby(groupby_column, group_keys=False)[[rollon_column, calc_column]]
            .rolling(rolling_days, mindays, on=rollon_column)
            .count()
            .reset_index(level=0, drop=True)[calc_column]
        )
        multiind_df[
            "rolling_" + calc_column + "_count" + "_daycombine"
        ] = multiind_df.groupby([groupby_column, rollon_column])[
            "rolling_" + calc_column + "_count"
        ].transform(
            "last"
        )
        todelete.append("rolling_" + calc_column + "_count")
    if calc_type == "sum":
        multiind_df["rolling_" + calc_column + "_sum"] = (
            multiind_df.sort_values([rollon_column, "date"])
            .groupby(groupby_column, group_keys=False)[[rollon_column, calc_column]]
            .rolling(rolling_days, mindays, on=rollon_column)
            .sum()
            .reset_index(level=0, drop=True)[calc_column]
        )
        multiind_df[
            "rolling_" + calc_column + "_sum" + "_daycombine"
        ] = multiind_df.groupby([groupby_column, rollon_column])[
            "rolling_" + calc_column + "_sum"
        ].transform(
            "last"
        )
        todelete.append("rolling_" + calc_column + "_sum")

    multiind_df.drop(todelete, axis=1, inplace=True)
    return multiind_df


def lagged_column(
    multiind_df, calc_column, groupby_column, lagon_column, lag_days, within_days
):
    todelete = []
    if isinstance(lagon_column, int):
        multiind_df[
            str(multiind_df.index.names[lagon_column]) + "_extra"
        ] = multiind_df.index.get_level_values(lagon_column)
        lagon_column = str(multiind_df.index.names[lagon_column])
        lagon_column = lagon_column + "_extra"
        todelete.append(lagon_column)

    if isinstance(groupby_column, int):
        multiind_df[
            str(multiind_df.index.names[groupby_column]) + "_extra"
        ] = multiind_df.index.get_level_values(groupby_column)
        groupby_column = str(multiind_df.index.names[groupby_column])
        groupby_column = groupby_column + "_extra"
        todelete.append(groupby_column)

    todelete = todelete + ["lastvalid_date"] + [lagon_column + "_lag"] + [calc_column]
    multiind_df.sort_values(
        [lagon_column, "date"], ascending=[True, True], inplace=True
    )

    multiind_df["lastvalid_date"] = multiind_df[lagon_column] - dt.timedelta(
        days=lag_days + within_days
    )
    multiind_df[[lagon_column + "_lag", calc_column + "_lag"]] = multiind_df.groupby(
        groupby_column
    )[[lagon_column, calc_column]].transform("shift")
    multiind_df[[lagon_column + "_lag", calc_column + "_lag"]] = multiind_df.groupby(
        [groupby_column, lagon_column]
    )[[lagon_column + "_lag", calc_column + "_lag"]].transform(lambda x: x.iloc[0])
    multiind_df.loc[
        multiind_df[lagon_column + "_lag"] <= multiind_df["lastvalid_date"],
        calc_column + "_lag",
    ] = np.nan
    multiind_df.drop(todelete, axis=1, inplace=True)
    return multiind_df


def property_rank(
    multiind_df, calc_column, groupby_column, method="dense", ascending=False
):
    todelete = []
    if isinstance(groupby_column, int):
        multiind_df[
            str(multiind_df.index.names[groupby_column]) + "_extra"
        ] = multiind_df.index.get_level_values(groupby_column)
        groupby_column = str(multiind_df.index.names[groupby_column])
        groupby_column = groupby_column + "_extra"
        todelete.append(groupby_column)

    multiind_df.sort_index(level=[1, 0], inplace=True)
    multiind_df[calc_column + "_rank"] = multiind_df.groupby(groupby_column)[
        calc_column
    ].rank(method=method, ascending=ascending)
    multiind_df.drop(todelete, axis=1, inplace=True)
    return multiind_df


def weightingfun(dataframe, calc_column, groupby_column, weight_method=None):
    todelete = []
    if isinstance(calc_column, int):
        dataframe[
            str(dataframe.index.names[calc_column]) + "_extra"
        ] = dataframe.index.get_level_values(calc_column)
        calc_column = str(dataframe.index.names[calc_column])
        calc_column = calc_column + "_extra"
        todelete.append(calc_column)

    if isinstance(groupby_column, int):
        dataframe[
            str(dataframe.index.names[groupby_column]) + "_extra"
        ] = dataframe.index.get_level_values(groupby_column)
        groupby_column = str(dataframe.index.names[groupby_column])
        groupby_column = groupby_column + "_extra"
        todelete.append(groupby_column)

    # IF weight_method IS NONE, WE JUST GIVE ALL TRADES OF THE DAY EQUAL WEIGHT
    if weight_method is None:
        dataframe["count_col"] = 1
        dataframe["weighting"] = 1 / dataframe.groupby(groupby_column)[
            "count_col"
        ].transform("sum")
        todelete.append("count_col")

    # IF weight_method IS pos_and_neg WE NEED TO FIND A WAY TO WEIGHT WHEN THERE ARE BOTH POSITIVE AND NEGATIVE VALUES
    # OBVIOUSLY THE USUAL WEIGHTED AVERAGE DOESN'T WORK BECAUSE THE DENOMINATOR COULD BE 0 BECAUSE OF THE NEGATIVE VALS
    if weight_method == "pos_and_neg":
        dataframe["minval"] = dataframe.groupby(groupby_column)[calc_column].transform(
            "min"
        )
        dataframe["maxval"] = dataframe.groupby(groupby_column)[calc_column].transform(
            "max"
        )
        dataframe.loc[dataframe["minval"] > 0, "minval"] = 0
        dataframe["tempval"] = (
            dataframe[calc_column]
            - dataframe["minval"]
            + 0.25 * (dataframe["maxval"] - dataframe["minval"])
        )
        dataframe["count_col"] = 1
        dataframe["denom"] = dataframe.groupby(groupby_column)["tempval"].transform(
            "sum"
        )
        dataframe.loc[dataframe["denom"] != 0, "weighting"] = (
            dataframe.loc[dataframe["denom"] != 0, "tempval"]
            / dataframe.loc[dataframe["denom"] != 0, "denom"]
        )
        dataframe.loc[dataframe["denom"] == 0, "weighting"] = 1 / dataframe.loc[
            dataframe["denom"] == 0, :
        ].groupby(groupby_column)["count_col"].transform("sum")
        todelete = (
            todelete + ["minval"] + ["maxval"] + ["tempval"] + ["count_col"] + ["denom"]
        )

    # IF weight_method IS bigger_better WE ASSUME IT MEANS ALL VALUES ARE POSITIVE AND BIGGER MEANS BETTER
    if weight_method == "bigger_better":
        dataframe["count_col"] = 1
        dataframe["denom"] = dataframe.groupby(groupby_column)[calc_column].transform(
            "sum"
        )
        dataframe.loc[dataframe["denom"] != 0, "weighting"] = (
            dataframe.loc[dataframe["denom"] != 0, calc_column]
            / dataframe.loc[dataframe["denom"] != 0, "denom"]
        )
        dataframe.loc[dataframe["denom"] == 0, "weighting"] = 1 / dataframe.loc[
            dataframe["denom"] == 0, :
        ].groupby(groupby_column)["count_col"].transform("sum")
        todelete.append("denom")
        todelete.append("count_col")

    # IF weight_method IS smaller_better IT MEANS IT'S SOMETHING LIKE STD WHERE WE WANT TO REWARD SMALL POSITIVE VALUES
    if weight_method == "smaller_better":
        dataframe["count_col"] = 1
        dataframe["denom"] = dataframe.groupby(groupby_column)[calc_column].transform(
            "sum"
        )
        dataframe.loc[dataframe["denom"] != 0, "weighting"] = (
            dataframe.loc[dataframe["denom"] != 0, calc_column]
            / dataframe.loc[dataframe["denom"] != 0, "denom"]
        )
        dataframe.loc[dataframe["denom"] == 0, "weighting"] = 1 / dataframe.loc[
            dataframe["denom"] == 0, :
        ].groupby(groupby_column)["count_col"].transform("sum")
        dataframe["weighting"] = 1 - dataframe["weighting"]
        dataframe["denom"] = dataframe.groupby(groupby_column)["weighting"].transform(
            "sum"
        )
        dataframe.loc[dataframe["denom"] == 0, "weighting"] = 1
        dataframe.loc[dataframe["denom"] != 0, "weighting"] = (
            dataframe.loc[dataframe["denom"] != 0, "weighting"]
            / dataframe.loc[dataframe["denom"] != 0, "denom"]
        )
        todelete.append("denom")
        todelete.append("count_col")

    dataframe.drop(todelete, axis=1, inplace=True)
    return dataframe


# =============================================================================
#  THESE PRIMARY FILTER CLASSES (DateFilter, IBSFilter, UpdayFilter, VolumeFilter etc...)
#  ARE APPLIED TO THE TIME SERIES TO NARROW DOWN THE POTENTIAL TRADE DAYS DOWN
#  TO ONLY THOSE DAYS WHEN OUR PRIMARY FILTER CONDITIONS ARE MET.
# =============================================================================
class TickerFilter:
    def __init__(self, tickerlist):
        self.ticker_list = tickerlist

    @classmethod
    def column_add(cls, dataframe):
        checkcols(dataframe)
        return dataframe

    def filtered_df(self, dataframe):
        checkcols(dataframe)

        dataframe = dataframe.loc[pd.IndexSlice[self.ticker_list, :], :]
        return dataframe


class DateFilter:
    def __init__(self, datelist, start_date=None, end_date=None):
        if (start_date is None) or (end_date is None):
            self.date_list = datelist
        else:
            self.date_list = pd.date_range(
                pd.Timestamp(start_date), pd.Timestamp(end_date)
            )

    @classmethod
    def column_add(cls, dataframe):
        checkcols(dataframe)
        return dataframe

    def filtered_df(self, dataframe):
        checkcols(dataframe)
        dataframe = dataframe.loc[
            dataframe.index.get_level_values("date").isin(self.date_list), :
        ]
        return dataframe


class VolumeFilter:
    lasttrade = 7  # THIS CLASS VARIABLE lasttrade IS SAYING LAST TRADE NEEDS TO HAVE OCCURRED WITHIN PAST 7 DAYS IN ORDER FOR US TO CONSIDER THE "YESTERDAY'S VOLUME" VALID
    colname = "volume_usd"

    def __init__(
        self, minvolume=1000000, daysago=0, weightmethod=False
    ):  # THIS minvolume IS IN USD
        self.min_volume = minvolume
        self.days_ago = daysago
        self.weight_method = weightmethod

    @classmethod
    def column_add(cls, dataframe):
        checkcols(dataframe)
        dataframe[cls.colname] = dataframe["volume"] * dataframe["close"]
        return dataframe

    def weights_col(self, dataframe):
        if self.weight_method == False:
            raise ValueError(
                "Cant call weights_col() while weightmethod argument is False"
            )
        else:
            checkcols_secondary(dataframe)
            if set([self.colname]).issubset(set(dataframe.columns)) == False:
                dataframe = self.column_add(dataframe)

            if set([self.colname + "_lag"]).issubset(set(dataframe.columns)):
                dataframe = weightingfun(
                    dataframe, self.colname + "_lag", 1, self.weight_method
                )
            else:
                dataframe = weightingfun(dataframe, self.colname, 1, self.weight_method)
            return dataframe

    def filtered_df(self, dataframe):
        if set([self.colname]).issubset(set(dataframe.columns)):
            checkcols(dataframe)
        else:
            dataframe = self.column_add(dataframe)

        if self.days_ago > 0:
            dataframe = lagged_column(
                dataframe, self.colname, 0, 1, self.days_ago, self.lasttrade
            )
            dataframe = dataframe.loc[
                (dataframe[self.colname + "_lag"] >= self.min_volume), :
            ]
        else:
            dataframe = dataframe.loc[(dataframe[self.colname] >= self.min_volume), :]
        return dataframe


class RollingVolumeFilter:
    rolling_days = "90D"  # THIS CLASS VARIABLE rolling_days IS HOW MANY DAYS BACK WE'RE LOOKING IN THE CALCULATION OF THE MOVING AVERAGE
    mintrades = 30  # THIS CLASS VARIABLE mintrades IS SAYING HOW MANY TRADES HAVE TO OCCURED IN THE ROLLING_DAYS IN ORDER FOR THE ROLLING VOLUME TO BE VALID
    lasttrade = 7  # THIS CLASS VARIABLE lasttrade IS SAYING LAST TRADE NEEDS TO HAVE OCCURRED WITHIN PAST 7 DAYS IN ORDER FOR US TO CONSIDER THE ROLLING AVG VOLUME TO BE VALID
    colname = "rolling_volume_usd_mean_daycombine"

    def __init__(self, min_avgvol=None, rank=None, daysago=0, weightmethod=False):
        self.days_ago = daysago

        if (
            min_avgvol is None
        ):  # min_volume IS THE MINIMUM ACCEPTABLE ROLLING AVERAGE VOLUME FOR A (TICKER,DATE TO BE KEPT IN CONSIDERATION
            self.min_volume = 0
        else:
            self.min_volume = min_avgvol

        if (
            rank is None
        ):  # rank IS SAYING ONLY TO ACCEPT THE TOP (OR BOTTOM) N TICKERS IN TERMS OF ROLLING VOLUME FOR THAT DAY
            self.rank = None
            self.rankascend = None
        else:
            self.rank = abs(rank)
            self.rankascend = rank < 0

        self.weight_method = weightmethod

    @classmethod
    def column_add(cls, dataframe):
        checkcols(dataframe)
        dataframe["volume_usd"] = dataframe["volume"] * dataframe["close"]
        dataframe = rolling_calc(
            dataframe, "volume_usd", 0, 1, cls.rolling_days, cls.mintrades, "mean"
        )
        return dataframe

    def weights_col(self, dataframe):
        if self.weight_method == False:
            raise ValueError(
                "Cant call weights_col() while weightmethod argument is False"
            )
        else:
            checkcols_secondary(dataframe)
            if set([self.colname]).issubset(set(dataframe.columns)) == False:
                dataframe = self.column_add(dataframe)

            if set([self.colname + "_lag"]).issubset(set(dataframe.columns)):
                dataframe = weightingfun(
                    dataframe, self.colname + "_lag", 1, self.weight_method
                )
            else:
                dataframe = weightingfun(dataframe, self.colname, 1, self.weight_method)
            return dataframe

    def filtered_df(self, dataframe):
        if set([self.colname]).issubset(set(dataframe.columns)):
            checkcols(dataframe)
        else:
            dataframe = self.column_add(dataframe)

        if self.days_ago > 0:
            dataframe = lagged_column(
                dataframe, self.colname, 0, 1, self.days_ago, self.lasttrade
            )
            usename = self.colname + "_lag"
        else:
            usename = self.colname

        if self.rankascend is not None:
            dataframe = property_rank(dataframe, usename, 1, ascending=self.rankascend)
            dataframe = dataframe.loc[
                (dataframe[usename + "_rank"] <= self.rank)
                & (dataframe[usename] >= self.min_volume),
                :,
            ]
        else:
            dataframe = dataframe.loc[(dataframe[usename] >= self.min_volume), :]

        return dataframe


class WeekdayFilter:
    colname = "weekday"

    def __init__(self, wanted_days=[0, 1, 2, 3, 4, 5, 6]):
        self.day_list = wanted_days  # a list of numbers, 0 is monday

    @classmethod
    def column_add(cls, dataframe):
        checkcols(dataframe)
        dataframe[cls.colname] = dataframe.index.get_level_values("date").dayofweek
        return dataframe

    def filtered_df(self, dataframe):
        if set([self.colname]).issubset(set(dataframe.columns)):
            checkcols(dataframe)
        else:
            dataframe = self.column_add(dataframe)

        dataframe = dataframe.loc[dataframe[self.colname].isin(self.day_list), :]
        return dataframe


# TO GET IBSMIN IBSMAX FILTER ALONG WITH IBSRANK FILTER, JUST CREATE TWO SEPARATE OBJECTS
class IBSFilter:
    lasttrade = 7  # THIS CLASS VARIABLE lasttrade IS SAYING LAST TRADE NEEDS TO HAVE OCCURRED WITHIN PAST 7 DAYS IN ORDER FOR US TO CONSIDER THE "YESTERDAY'S IBS" VALID
    colname = "ibs"

    def __init__(self, lowbound=0, upbound=1, rank=None, daysago=0, weightmethod=False):
        self.IBS_low = lowbound
        self.IBS_high = upbound
        self.days_ago = daysago
        self.weight_method = weightmethod
        if (
            rank is None
        ):  # rank IS SAYING ONLY TO ACCEPT THE TOP (OR BOTTOM) N TICKERS IN TERMS OF IBS FOR THAT DAY
            self.rank = None
            self.rankascend = None
        else:
            self.rank = abs(rank)
            self.rankascend = rank < 0

    @classmethod
    def column_add(cls, dataframe):
        checkcols(dataframe)
        if set([cls.colname]).issubset(set(dataframe.columns)) == False:
            dataframe[cls.colname] = (dataframe["close"] - dataframe["low"]) / (
                dataframe["high"] - dataframe["low"]
            )
            dataframe[cls.colname].replace(
                to_replace=-np.Inf, value=np.nan, inplace=True
            )
            dataframe[cls.colname].replace(
                to_replace=np.Inf, value=np.nan, inplace=True
            )

        return dataframe

    def weights_col(self, dataframe):
        if self.weight_method == False:
            raise ValueError(
                "Cant call weights_col() while weightmethod argument is False"
            )
        else:
            checkcols_secondary(dataframe)
            if set([self.colname]).issubset(set(dataframe.columns)) == False:
                dataframe = self.column_add(dataframe)

            if set([self.colname + "_lag"]).issubset(set(dataframe.columns)):
                dataframe = weightingfun(
                    dataframe, self.colname + "_lag", 1, self.weight_method
                )
            else:
                dataframe = weightingfun(dataframe, self.colname, 1, self.weight_method)
            return dataframe

    def filtered_df(self, dataframe):
        if set([self.colname]).issubset(set(dataframe.columns)):
            checkcols(dataframe)
        else:
            dataframe = self.column_add(dataframe)

        if self.days_ago > 0:
            dataframe = lagged_column(
                dataframe, self.colname, 0, 1, self.days_ago, self.lasttrade
            )
            usecol = self.colname + "_lag"
        else:
            usecol = self.colname

        if self.rankascend is not None:
            dataframe = property_rank(dataframe, usecol, 1, ascending=self.rankascend)
            dataframe = dataframe.loc[
                (dataframe[usecol + "_rank"] <= self.rank)
                & (dataframe[usecol] >= self.IBS_low)
                & (dataframe[usecol] <= self.IBS_high),
                :,
            ]
        else:
            dataframe = dataframe.loc[
                (dataframe[usecol] >= self.IBS_low)
                & (dataframe[usecol] <= self.IBS_high),
                :,
            ]

        return dataframe


class UpdayFilter:
    lasttrade = 7  # THIS CLASS VARIABLE lasttrade IS SAYING LAST TRADE NEEDS TO HAVE OCCURRED WITHIN PAST 7 DAYS IN ORDER FOR US TO CONSIDER THE "YESTERDAY'S IBS" VALID
    colname = "upday"

    def __init__(self, updaytrue=True, daysago=0):
        self.upday = updaytrue
        self.days_ago = daysago

    @classmethod
    def column_add(cls, dataframe):
        checkcols(dataframe)
        dataframe[cls.colname] = (dataframe["close"] - dataframe["open"]) >= 0
        return dataframe

    def filtered_df(self, dataframe):
        if set([self.colname]).issubset(set(dataframe.columns)):
            checkcols(dataframe)
        else:
            dataframe = self.column_add(dataframe)
        if self.days_ago > 0:
            dataframe = lagged_column(
                dataframe, self.colname, 0, 1, self.days_ago, self.lasttrade
            )
            dataframe = dataframe.loc[dataframe[self.colname + "_lag"] == self.upday, :]
        else:
            dataframe = dataframe.loc[dataframe[self.colname] == self.upday, :]
        return dataframe


# IF TICKLIST/DATELIST PAIRS ARE NOT PROVIDED THROUGH multiind THEN ONE MUST MAKE SURE THAT ticklist
# and datelist HAVE THE CORRECT TICKERS MATCHED WITH THE CORRECT DATES, BECAUSE WE'RE GOING TO COMBINE THEM
class TickerDateFilter:
    def __init__(self, multiind, ticklist=None, datelist=None):
        if (ticklist is None) or (datelist is None):
            multiind = multiind.set_levels(
                pd.to_datetime(multiind.levels[1]), level=1
            )
            self.multiind_use = multiind
        else:
            setmulti = pd.MultiIndex.from_arrays(
                [ticklist, datelist], names=["ticker", "date"]
            )
            setmulti = setmulti.set_levels(
                pd.to_datetime(setmulti.levels[1]), level=1
            )
            self.multiind_use = setmulti

    @classmethod
    def column_add(cls, dataframe):
        checkcols(dataframe)
        return dataframe

    def filtered_df(self, dataframe):
        checkcols(dataframe)

        dataframe = dataframe.loc[dataframe.index.intersection(self.multiind_use), :]
        return dataframe


# THIS FILTER BRINGS IN AN OUTSIDE COLUMN FOR USE IN A BACKTEST. FOR EXAMPLE, IF YOU WANT TO TRADE USING STRATEGY 'A'
# USING TOP N TICKERS ON THAT DAY UNDER STRATEGY 'B', YOU COULD RUN STRATEGY B SEPARATELY FIRST, AND THEN BRING ITS
# RANK COLUMN INTO STRATEGY A'S BACKTEST USING ExternalValFilter
class ExternalValFilter:
    lasttrade = 7  # THIS CLASS VARIABLE lasttrade IS SAYING LAST TRADE NEEDS TO HAVE OCCURRED WITHIN PAST 7 DAYS IN ORDER FOR US TO CONSIDER THE "YESTERDAY'S VAL" VALID

    def __init__(
        self,
        series_w_multiind,
        col_name,
        lowbound=-10000000,
        upbound=10000000,
        rank=None,
        daysago=0,
        weightmethod=False,
    ):
        self.val_low = lowbound
        self.val_high = upbound
        self.days_ago = daysago
        self.weight_method = weightmethod
        self.colname = col_name

        series_w_multiind.index = series_w_multiind.index.set_levels(
            pd.to_datetime(series_w_multiind.index.levels[1]), level=1
        )
        self.external_series = series_w_multiind

        if (
            rank is None
        ):  # rank IS SAYING ONLY TO ACCEPT THE TOP (OR BOTTOM) N TICKERS IN TERMS OF IBS FOR THAT DAY
            self.rank = None
            self.rankascend = None
        else:
            self.rank = abs(rank)
            self.rankascend = rank < 0

    def column_add(self, dataframe):
        checkcols(dataframe)
        if set([self.colname]).issubset(set(dataframe.columns)) == False:
            dataframe = pd.merge(
                dataframe,
                self.external_series.rename(self.colname),
                how="left",
                left_index=True,
                right_index=True,
            )

        return dataframe

    def weights_col(self, dataframe):
        if self.weight_method == False:
            raise ValueError(
                "Cant call weights_col() while weightmethod argument is False"
            )
        else:
            checkcols_secondary(dataframe)
            if set([self.colname]).issubset(set(dataframe.columns)) == False:
                dataframe = self.column_add(dataframe)

            if set([self.colname + "_lag"]).issubset(set(dataframe.columns)):
                dataframe = weightingfun(
                    dataframe, self.colname + "_lag", 1, self.weight_method
                )
            else:
                dataframe = weightingfun(dataframe, self.colname, 1, self.weight_method)
            return dataframe

    def filtered_df(self, dataframe):
        if set([self.colname]).issubset(set(dataframe.columns)):
            checkcols(dataframe)
        else:
            dataframe = self.column_add(dataframe)

        if self.days_ago > 0:
            dataframe = lagged_column(
                dataframe, self.colname, 0, 1, self.days_ago, self.lasttrade
            )
            usecol = self.colname + "_lag"
        else:
            usecol = self.colname

        if self.rankascend is not None:
            dataframe = property_rank(dataframe, usecol, 1, ascending=self.rankascend)
            dataframe = dataframe.loc[
                (dataframe[usecol + "_rank"] <= self.rank)
                & (dataframe[usecol] >= self.val_low)
                & (dataframe[usecol] <= self.val_high),
                :,
            ]
        else:
            dataframe = dataframe.loc[
                (dataframe[usecol] >= self.val_low)
                & (dataframe[usecol] <= self.val_high),
                :,
            ]

        return dataframe


# =============================================================================
#  THE CLASSES AndFilter AND OrFilter LET YOU CREATE A FILTER LIKE (UpDayFilter AND TickerFilter),
#  OR YOU COULD CREATE A FILTER LIKE (IBSFilter OR WeekdayFilter) ...
#  YOU CAN EVEN COMBINE AndFilters WITH OrFilters.
# =============================================================================
class AndFilter:
    def __init__(self, *filterz):
        self.filterz = list(filterz)

    def column_add(self, dataframe):
        for clses in list(set([x.__class__ for x in subclass_find(*self.filterz)])):
            dataframe = clses.column_add(dataframe).copy()

        return dataframe

    def filtered_df(self, dataframe):
        for fltrs in self.filterz:
            dataframe = fltrs.filtered_df(dataframe).copy()

        return dataframe


class OrFilter:
    def __init__(self, *filterz):
        self.filterz = list(filterz)

    def column_add(self, dataframe):
        for clses in list(set([x.__class__ for x in subclass_find(*self.filterz)])):
            dataframe = clses.column_add(dataframe).copy()

        return dataframe

    def filtered_df(self, dataframe):
        df_list = []
        for fltrs in self.filterz:
            df = fltrs.filtered_df(dataframe)
            df_list.append(df)

        dataframe = pd.concat(df_list, axis=0, join="outer", sort=False, copy=False)
        dataframe = dataframe.loc[~dataframe.index.duplicated(), :]

        return dataframe


# =============================================================================
#  THIS EntryFilter CLASS TAKES *args OF PRIMARY FILTERS YOU WANT TO APPLY,
#  AND IT APPLIES THEM TO YOUR RawData OBJECT THAT YOU DEFINED AT THE BEGINNING.
#  YOU USE THIS TO NARROW DOWN THE DATA TO A SET OF (TICKER, DATE) PAIRS ON
#  WHICH WE'RE ALLOWED TO ENTER A TRADE. FOR EXAMPLE, ONLY ENTER TRADES ON
#  (MON, WED, FRI) WHEN YESTERDAY WAS A DOWNDAY WITH IBS < 0.2
# =============================================================================
class EntryFilter:
    def __init__(self, data_object, *filterz):
        if type(data_object) is RawData:
            self.dataframe_obj = data_object.converted_df.copy()
        else:
            raise TypeError("need for data_object to be of class RawData")

        self.filterz = list(filterz)
        self.filtered_df = self.apply_filters()

    def add_columns(self):
        toadd_df = self.dataframe_obj.copy()
        for clses in list(set([x.__class__ for x in subclass_find(*self.filterz)])):
            toadd_df = clses.column_add(toadd_df)

        return toadd_df

    def apply_filters(self):
        if len(self.filterz) == 0:
            return self.dataframe_obj
        else:
            tofilter_df = self.add_columns()
            for fltrs in self.filterz:
                tofilter_df = fltrs.filtered_df(tofilter_df)

            return tofilter_df


# =============================================================================
#  THIS ExitFilter CLASS TAKES *args OF PRIMARY FILTERS YOU WANT TO APPLY,
#  AND IT APPLIES THEM TO YOUR RawData OBJECT THAT YOU DEFINED AT THE BEGINNING.
#  YOU USE THIS TO NARROW DOWN THE DATA TO A SET OF (TICKER, DATE) PAIRS ON
#  WHICH WE'D BE ALLOWED TO EXIT A TRADE. FOR EXAMPLE, ONLY EXIT TRADES ON
#  DAYS WHEN SOME OTHER TICKERS IN A TickerFIlter HAD AN UPDAY YESTERDAY
# =============================================================================
class ExitFilter:
    def __init__(self, data_object, *filterz):
        if type(data_object) is RawData:
            self.dataframe_obj = data_object.converted_df.copy()
        else:
            raise TypeError("need for data_object to be of class RawData")

        self.filterz = list(filterz)
        self.filtered_df = self.apply_filters()

    def add_columns(self):
        toadd_df = self.dataframe_obj.copy()
        for clses in list(set([x.__class__ for x in subclass_find(*self.filterz)])):
            toadd_df = clses.column_add(toadd_df)

        return toadd_df

    def apply_filters(self):
        if len(self.filterz) == 0:
            return self.dataframe_obj
        elif len(subclass_find(*self.filterz)) == 1:
            tofilter_df = self.add_columns()
            tofilter_df = self.filterz[0].filtered_df(tofilter_df)
            return tofilter_df
        else:
            tofilter_df = self.add_columns()
            lastchance_df = tofilter_df.copy()
            fltr_list = self.filterz.copy()
            try:
                first_date = next(
                    x
                    for x in fltr_list
                    if any(
                        [
                            isinstance(y, (DateFilter, TickerDateFilter))
                            for y in subclass_find(x)
                        ]
                    )
                )
            except StopIteration:
                first_date = 12345
                fltr_list += [12345]

            if first_date != 12345:
                lastchance_df = first_date.filtered_df(tofilter_df)

            lastdate = (
                lastchance_df.reset_index().groupby("ticker")["date"].max().to_frame()
            )
            lastdate.set_index("date", append=True, inplace=True)
            lastchance_df = lastchance_df.loc[
                lastchance_df.index.intersection(lastdate.index), :
            ]
            fltr_list.remove(first_date)

            for fltrs in fltr_list:
                tofilter_df = fltrs.filtered_df(tofilter_df)

            tofilter_df = pd.concat(
                [tofilter_df, lastchance_df],
                axis=0,
                join="outer",
                sort=False,
                copy=False,
            )
            tofilter_df = tofilter_df.loc[~tofilter_df.index.duplicated(), :]
            return tofilter_df


# ALWAYS PUT THE FILTER WITH THE LATEST DATE FIRST. THIS IS BECAUSE WE'LL USE THAT
# LATEST DATE TO CLOSE OUT THE TRADES WHOSE EXIT CONDITIONS HAVEN'T BEEN MET BY THE
# END OF THE BACKTEST... WE DON'T WANT THOSE TRADES JUST LEFT OPEN SO WE'LL DO AN
# "EMERGENCY EXIT" USING THE LAST DATE IN THE DATASET


# =============================================================================
#  THESE SECONDARY FILTERS (SharpeFilter, StdFilter, CumReturnFilter, etc...) WILL BE
#  USED AFTER ALREADY HAVING RUN A PRELIMINARY BACKTEST USING THE PRIMARY FILTERS. THE
#  PRELIMINARY BACKTEST WILL TELL US HOW EACH TICKER IS PERFORMING AND THEN WE CAN USE
#  THESE SECONDARY FILTERS TO DO THINGS LIKE ONLY TRADE THE TICKERS WHICH, AS OF YESTERDAY,
#  HAD A ROLLING SHARPE (PAST 364 DAYS) OF ABOVE 3... THINGS LIKE THAT
# =============================================================================
class SharpeFilter:
    def __init__(
        self,
        min_sharpe=None,  # min_sharpe IS THE MINIMUM ACCEPTABLE ROLLING SHARPE FOR A (TICKER,DATE TO BE KEPT IN CONSIDERATION
        rank=None,  # rank IS SAYING ONLY TO ACCEPT THE TOP (OR BOTTOM) N TICKERS IN TERMS OF ROLLING SHARPE FOR THAT DAY
        weightmethod=False,  # CHOOSE WHETHER BIGGER IS BETTER, SMALLER IS BETTER, WEIGHTED EVENLY ETC.
        rolling_days=90,  # THIS INSTANCE VARIABLE rolling_days IS HOW MANY DAYS BACK WE'RE LOOKING IN THE CALCULATION OF THE ROLLING SHARPE
        mintrades=30,  # THIS INSTANCE VARIABLE mintrades IS SAYING HOW MANY TRADES HAVE TO OCCURED IN THE ROLLING_DAYS IN ORDER FOR THE ROLLING SHARPE TO BE VALID
    ):
        self.rolling_days = str(rolling_days) + "D"

        self.mintrades = mintrades

        self.colname = (
            "rolling_sharpe_" + self.rolling_days + "_atleast" + str(self.mintrades)
        )

        if min_sharpe is None:
            self.min_sharpe = -1000
        else:
            self.min_sharpe = min_sharpe

        if rank is None:
            self.rank = None
            self.rankascend = None
        else:
            self.rank = abs(rank)
            self.rankascend = rank < 0

        self.weight_method = weightmethod

    def column_add(self, dataframe):
        checkcols_secondary(dataframe)
        if set([self.colname]).issubset(set(dataframe.columns)) == False:
            dataframe = rolling_calc(
                dataframe,
                "trade_return",
                0,
                "date_exit",
                self.rolling_days,
                self.mintrades,
                "mean",
            )
            dataframe = rolling_calc(
                dataframe,
                "trade_return",
                0,
                "date_exit",
                self.rolling_days,
                self.mintrades,
                "std",
            )
            dataframe[self.colname] = (
                dataframe["rolling_trade_return_mean_daycombine"]
                / dataframe["rolling_trade_return_std_daycombine"]
            )
        return dataframe

    def weights_col(self, dataframe):
        if self.weight_method == False:
            raise ValueError(
                "Cant call weights_col() while weightmethod argument is False"
            )
        else:
            checkcols_secondary(dataframe)
            if set([self.colname]).issubset(set(dataframe.columns)) == False:
                dataframe = self.column_add(dataframe)

            dataframe = weightingfun(dataframe, self.colname, 1, self.weight_method)
            return dataframe

    def filtered_df(self, dataframe):
        if set([self.colname]).issubset(set(dataframe.columns)):
            checkcols_secondary(dataframe)
        else:
            dataframe = self.column_add(dataframe)

        if self.rankascend is not None:
            dataframe = property_rank(
                dataframe, self.colname, 1, ascending=self.rankascend
            )
            dataframe = dataframe.loc[
                (dataframe[self.colname + "_rank"] <= self.rank)
                & (dataframe[self.colname] >= self.min_sharpe),
                :,
            ]
        else:
            dataframe = dataframe.loc[(dataframe[self.colname] >= self.min_sharpe), :]

        return dataframe


class StdevFilter:
    def __init__(
        self,
        max_std=None,  # max_std IS THE MAXIMUM ACCEPTABLE ROLLING STD FOR A (TICKER,DATE TO BE KEPT IN CONSIDERATION
        rank=None,  # rank IS SAYING ONLY TO ACCEPT THE TOP (OR BOTTOM) N TICKERS IN TERMS OF ROLLING STD FOR THAT DAY
        weightmethod=False,  # CHOOSE WHETHER BIGGER IS BETTER, SMALLER IS BETTER, WEIGHTED EVENLY ETC.
        rolling_days=90,  # THIS INSTANCE VARIABLE rolling_days IS HOW MANY DAYS BACK WE'RE LOOKING IN THE CALCULATION OF THE ROLLING STD
        mintrades=30,  # THIS INSTANCE VARIABLE mintrades IS SAYING HOW MANY TRADES HAVE TO OCCURED IN THE ROLLING_DAYS IN ORDER FOR THE ROLLING STD TO BE VALID
    ):
        self.rolling_days = str(rolling_days) + "D"

        self.mintrades = mintrades

        self.colname = (
            "rolling_trade_return_std_daycombine_"
            + self.rolling_days
            + "_atleast"
            + str(self.mintrades)
        )
        if max_std is None:
            self.max_std = 1000
        else:
            self.max_std = max_std

        if rank is None:
            self.rank = None
            self.rankascend = None
        else:
            self.rank = abs(rank)
            self.rankascend = rank > 0

        self.weight_method = weightmethod

    def column_add(self, dataframe):
        checkcols_secondary(dataframe)
        if set([self.colname]).issubset(set(dataframe.columns)) == False:
            dataframe = rolling_calc(
                dataframe,
                "trade_return",
                0,
                "date_exit",
                self.rolling_days,
                self.mintrades,
                "std",
            )
            dataframe.rename(
                columns={"rolling_trade_return_std_daycombine": self.colname},
                inplace=True,
            )
        return dataframe

    def weights_col(self, dataframe):
        if self.weight_method == False:
            raise ValueError(
                "Cant call weights_col() while weightmethod argument is False"
            )
        else:
            checkcols_secondary(dataframe)
            if set([self.colname]).issubset(set(dataframe.columns)) == False:
                dataframe = self.column_add(dataframe)

            dataframe = weightingfun(dataframe, self.colname, 1, self.weight_method)
            return dataframe

    def filtered_df(self, dataframe):
        if set([self.colname]).issubset(set(dataframe.columns)):
            checkcols_secondary(dataframe)
        else:
            dataframe = self.column_add(dataframe)

        if self.rankascend is not None:
            dataframe = property_rank(
                dataframe, self.colname, 1, ascending=self.rankascend
            )
            dataframe = dataframe.loc[
                (dataframe[self.colname + "_rank"] <= self.rank)
                & (dataframe[self.colname] <= self.max_std),
                :,
            ]
        else:
            dataframe = dataframe.loc[(dataframe[self.colname] <= self.max_std), :]

        return dataframe


class CumReturnFilter:
    def __init__(
        self,
        min_cumreturn=None,  # min_cumreturn IS THE MINIMUM ACCEPTABLE ROLLING CUMRETURN FOR A (TICKER,DATE TO BE KEPT IN CONSIDERATION
        rank=None,  # rank IS SAYING ONLY TO ACCEPT THE TOP (OR BOTTOM) N TICKERS IN TERMS OF ROLLING CUMRETURN FOR THAT DAY
        weightmethod=False,  # CHOOSE WHETHER BIGGER IS BETTER, SMALLER IS BETTER, WEIGHTED EVENLY ETC.
        rolling_days=90,  # THIS INSTANCE VARIABLE rolling_days IS HOW MANY DAYS BACK WE'RE LOOKING IN THE CALCULATION OF THE ROLLING CUM RETURN
        mintrades=30,  # THIS INSTANCE VARIABLE mintrades IS SAYING HOW MANY TRADES HAVE TO OCCURED IN THE ROLLING_DAYS IN ORDER FOR THE ROLLING CUM RETURN TO BE VALID
    ):
        self.rolling_days = str(rolling_days) + "D"

        self.mintrades = mintrades

        self.colname = (
            "rolling_trade_return_sum_daycombine_"
            + self.rolling_days
            + "_atleast"
            + str(self.mintrades)
        )
        if min_cumreturn is None:
            self.min_cumreturn = -(10 ** 5)
        else:
            self.min_cumreturn = min_cumreturn

        if rank is None:
            self.rank = None
            self.rankascend = None
        else:
            self.rank = abs(rank)
            self.rankascend = rank < 0

        self.weight_method = weightmethod

    def column_add(self, dataframe):
        checkcols_secondary(dataframe)
        if set([self.colname]).issubset(set(dataframe.columns)) == False:
            dataframe = rolling_calc(
                dataframe,
                "trade_return",
                0,
                "date_exit",
                self.rolling_days,
                self.mintrades,
                "sum",
            )
            dataframe.rename(
                columns={"rolling_trade_return_sum_daycombine": self.colname},
                inplace=True,
            )
        return dataframe

    def weights_col(self, dataframe):
        if self.weight_method == False:
            raise ValueError(
                "Cant call weights_col() while weightmethod argument is False"
            )
        else:
            checkcols_secondary(dataframe)
            if set([self.colname]).issubset(set(dataframe.columns)) == False:
                dataframe = self.column_add(dataframe)

            dataframe = weightingfun(dataframe, self.colname, 1, self.weight_method)
            return dataframe

    def filtered_df(self, dataframe):
        if set([self.colname]).issubset(set(dataframe.columns)):
            checkcols_secondary(dataframe)
        else:
            dataframe = self.column_add(dataframe)

        if self.rankascend is not None:
            dataframe = property_rank(
                dataframe, self.colname, 1, ascending=self.rankascend
            )
            dataframe = dataframe.loc[
                (dataframe[self.colname + "_rank"] <= self.rank)
                & (dataframe[self.colname] >= self.min_cumreturn),
                :,
            ]
        else:
            dataframe = dataframe.loc[
                (dataframe[self.colname] >= self.min_cumreturn), :
            ]

        return dataframe


class NegStdFilter:
    def __init__(
        self,
        max_negstd=None,  # max_negstd IS THE MAXIMUM ACCEPTABLE ROLLING NEGSTD FOR A (TICKER,DATE TO BE KEPT IN CONSIDERATION
        rank=None,  # rank IS SAYING ONLY TO ACCEPT THE TOP (OR BOTTOM) N TICKERS IN TERMS OF ROLLING NEGSTD FOR THAT DAY
        weightmethod=False,  # CHOOSE WHETHER BIGGER IS BETTER, SMALLER IS BETTER, WEIGHTED EVENLY ETC.
        rolling_days=90,  # THIS INSTANCE VARIABLE rolling_days IS HOW MANY DAYS BACK WE'RE LOOKING IN THE CALCULATION OF THE ROLLING NEGSTDEV
        mintrades=15,  # THIS INSTANCE VARIABLE mintrades IS SAYING HOW MANY TRADES HAVE TO OCCURED IN THE ROLLING_DAYS IN ORDER FOR THE ROLLING NEGSTD TO BE VALID
        carryforward_days=7,  # THIS INSTANCE VARIABLE IS SAYING HOW MANY DAYS FORWARD ARE WE WILLING TO CARRY FORWARD A NEGSTD VALUE WITHOUT GETTING A NEW NEGATIVE RETURN TO UPDATE IT WITH
    ):
        self.rolling_days = str(rolling_days) + "D"

        self.mintrades = mintrades

        self.carryforward_days = carryforward_days

        self.colname = (
            "rolling_negstd_"
            + self.rolling_days
            + "_atleast"
            + str(self.mintrades)
            + "_carryforward"
            + str(self.carryforward_days)
        )
        if max_negstd is None:
            self.max_negstd = 1000
        else:
            self.max_negstd = max_negstd

        if rank is None:
            self.rank = None
            self.rankascend = None
        else:
            self.rank = abs(rank)
            self.rankascend = rank > 0

        self.weight_method = weightmethod

    def column_add(self, dataframe):
        checkcols_secondary(dataframe)
        if set([self.colname]).issubset(set(dataframe.columns)) == False:
            dataframe = rolling_calc(
                dataframe,
                "trade_return",
                0,
                "date_exit",
                self.rolling_days,
                self.mintrades,
                "negstd",
            )

            # NEGSTD CALC CAN ONLY BE UPDATED ON DAYS WHEN WE HAD A NEGATIVE RETURN, SO WE NEED TO CARRY THAT UPDATED VALUE
            # FORWARD THROUGH POSITIVE DAYS (POSITIVE DAYS PROVIDE NO NEW INFORMATION FOR NEGSTD)
            dataframe.loc[
                dataframe["trade_return"] < 0, "neg_date_exit"
            ] = dataframe.loc[dataframe["trade_return"] < 0, "date_exit"]
            dataframe["last_negative_date"] = (
                dataframe.sort_values("date_exit", ascending=True)
                .groupby(level=0)["neg_date_exit"]
                .fillna(method="ffill")
            )
            # MAKING THIS tempnan DESIGNATION BECAUSE WE WANT TO PROPOGATE negstd VALUES FORWARD TO THE POSITIVE RETURN DAYS (WHICH WILL HAVE nan in the NEGSTD COL)
            # BUT WE DONT WANT TO PROPOGATE VALUES TO NEGATIVE DAYS THAT HAVE nan IN THE NEGSTD COLUMN DUE TO INSUFFICIENT NEG TRADES IN THE PAST 90 DAYS
            dataframe["temp_daysstd"] = dataframe[
                "rolling_trade_return_negstd_daycombine"
            ]
            dataframe.loc[
                (dataframe["trade_return"] < 0) & np.isnan(dataframe["temp_daysstd"]),
                "temp_daysstd",
            ] = -999
            dataframe["last_negstd"] = (
                dataframe.sort_values("date_exit", ascending=True)
                .groupby(level=0)["temp_daysstd"]
                .fillna(method="ffill")
            )
            dataframe.loc[dataframe["last_negstd"] == -999, "last_negstd"] = np.nan

            # NOW EVERYTHING IS PROPOGATED FORWARD, BUT WE NEED TO USE carryforward_days TO PREVENT A NEGSTD VALUE BEING CARRIED FORWARD TOO FAR WITHOUT AN UPDATE
            dataframe["first_acceptable"] = dataframe["date_exit"] - dt.timedelta(
                days=self.carryforward_days
            )
            dataframe["isacceptable"] = (
                dataframe["last_negative_date"] >= dataframe["first_acceptable"]
            )
            dataframe.loc[dataframe["isacceptable"] == False, "last_negstd"] = np.nan

            # NOW JUST RENAME THE COLUMN last_negstd AS JUST negstd
            dataframe.rename(columns={"last_negstd": self.colname}, inplace=True)

            # AND DROP ALL THE EXTRANEOUS COLUMNS
            dataframe.drop(
                [
                    "temp_daysstd",
                    "neg_date_exit",
                    "last_negative_date",
                    "first_acceptable",
                    "isacceptable",
                ],
                axis=1,
                inplace=True,
            )
        return dataframe

    def weights_col(self, dataframe):
        if self.weight_method == False:
            raise ValueError(
                "Cant call weights_col() while weightmethod argument is False"
            )
        else:
            checkcols_secondary(dataframe)
            if set([self.colname]).issubset(set(dataframe.columns)) == False:
                dataframe = self.column_add(dataframe)

            dataframe = weightingfun(dataframe, self.colname, 1, self.weight_method)
            return dataframe

    def filtered_df(self, dataframe):
        if set([self.colname]).issubset(set(dataframe.columns)):
            checkcols_secondary(dataframe)
        else:
            dataframe = self.column_add(dataframe)

        dataframe = dataframe.loc[(dataframe[self.colname] <= self.max_negstd), :]
        if self.rankascend is not None:
            dataframe = property_rank(
                dataframe, self.colname, 1, ascending=self.rankascend
            )
            dataframe = dataframe.loc[
                (dataframe[self.colname] <= self.rank)
                & (dataframe[self.colname] <= self.max_negstd),
                :,
            ]
        else:
            dataframe = dataframe.loc[(dataframe[self.colname] <= self.max_negstd), :]

        return dataframe


class SortinoFilter(NegStdFilter):
    def __init__(
        self,
        min_sortino=None,  # min_sortino IS THE MINIMUM ACCEPTABLE ROLLING SORTINO FOR A (TICKER,DATE TO BE KEPT IN CONSIDERATION
        rank=None,  # rank IS SAYING ONLY TO ACCEPT THE TOP (OR BOTTOM) N TICKERS IN TERMS OF ROLLING SORTINO FOR THAT DAY
        weightmethod=False,  # CHOOSE WHETHER BIGGER IS BETTER, SMALLER IS BETTER, WEIGHTED EVENLY ETC.
        rolling_days="90D",  # THIS INSTANCE VARIABLE rolling_days IS HOW MANY DAYS BACK WE'RE LOOKING IN THE CALCULATION OF THE ROLLING SORTINO
        mintrades=15,  # THIS INSTANCE VARIABLE mintrades IS SAYING HOW MANY TRADES HAVE TO OCCURED IN THE ROLLING_DAYS IN ORDER FOR THE ROLLING SORTINO TO BE VALID
        carryforward_days=7,  # THIS INSTANCE VARIABLE IS SAYING HOW MANY DAYS FORWARD ARE WE WILLING TO CARRY FORWARD A NEGSTD VALUE WITHOUT GETTING A NEW NEGATIVE RETURN TO UPDATE IT WITH
    ):
        self.rolling_days = str(rolling_days) + "D"

        self.mintrades = mintrades

        self.carryforward_days = carryforward_days

        self.colname = (
            "rolling_sortino_"
            + self.rolling_days
            + "_atleast"
            + str(self.mintrades)
            + "_carryforward"
            + str(self.carryforward_days)
        )

        if min_sortino is None:
            self.min_sortino = -1000
        else:
            self.min_sortino = min_sortino

        if rank is None:
            self.rank = None
            self.rankascend = None
        else:
            self.rank = abs(rank)
            self.rankascend = rank < 0

        self.weight_method = weightmethod

    def column_add(self, dataframe):
        checkcols_secondary(dataframe)
        if set([self.colname]).issubset(set(dataframe.columns)) == False:
            # TEMPORARILY CHANGE COLNAME SO THAT WE CAN USE COLUMN_ADD FROM NegStdFilter
            self.colname = (
                "rolling_negstd_"
                + self.rolling_days
                + "_atleast"
                + str(self.mintrades)
                + "_carryforward"
                + str(self.carryforward_days)
            )
            dataframe = super().column_add(dataframe)
            self.colname = (
                "rolling_sortino_"
                + self.rolling_days
                + "_atleast"
                + str(self.mintrades)
                + "_carryforward"
                + str(self.carryforward_days)
            )

            # CALCULATING ROLLING AVG RETURN AND THEN SORTINO
            dataframe = rolling_calc(
                dataframe,
                "trade_return",
                0,
                "date_exit",
                self.rolling_days,
                self.mintrades,
                "mean",
            )
            dataframe[self.colname] = (
                dataframe["rolling_trade_return_mean_daycombine"]
                / dataframe["rolling_negstd"]
            )
        return dataframe

    def weights_col(self, dataframe):
        if self.weight_method == False:
            raise ValueError(
                "Cant call weights_col() while weightmethod argument is False"
            )
        else:
            checkcols_secondary(dataframe)
            if set([self.colname]).issubset(set(dataframe.columns)) == False:
                dataframe = self.column_add(dataframe)

            dataframe = weightingfun(dataframe, self.colname, 1, self.weight_method)
            return dataframe

    def filtered_df(self, dataframe):
        if set([self.colname]).issubset(set(dataframe.columns)):
            checkcols_secondary(dataframe)
        else:
            dataframe = self.column_add(dataframe)

        if self.rankascend is not None:
            dataframe = property_rank(
                dataframe, self.colname, 1, ascending=self.rankascend
            )
            dataframe = dataframe.loc[
                (dataframe[self.colname + "_rank"] <= self.rank)
                & (dataframe[self.colname] >= self.min_sortino),
                :,
            ]
        else:
            dataframe = dataframe.loc[(dataframe[self.colname] >= self.min_sortino), :]

        return dataframe


class AvgReturnFilter:
    def __init__(
        self,
        min_avg=None,  # min_avg IS THE MINIMUM ACCEPTABLE ROLLING AVG FOR A (TICKER,DATE) TO BE KEPT IN CONSIDERATION
        rank=None,  # rank IS SAYING ONLY TO ACCEPT THE TOP (OR BOTTOM) N TICKERS IN TERMS OF ROLLING AVG FOR THAT DAY
        weightmethod=False,  # CHOOSE WHETHER BIGGER IS BETTER, SMALLER IS BETTER, WEIGHTED EVENLY ETC.
        rolling_days=90,  # THIS INSTANCE VARIABLE rolling_days IS HOW MANY DAYS BACK WE'RE LOOKING IN THE CALCULATION OF THE ROLLING AVG
        mintrades=30,  # THIS INSTANCE VARIABLE mintrades IS SAYING HOW MANY TRADES HAVE TO OCCURED IN THE ROLLING_DAYS IN ORDER FOR THE ROLLING AVG TO BE VALID
    ):
        self.rolling_days = str(rolling_days) + "D"

        self.mintrades = mintrades

        self.colname = (
            "rolling_trade_return_mean_daycombine_"
            + self.rolling_days
            + "_atleast"
            + str(self.mintrades)
        )
        if (
            min_avg is None
        ):  # min_avg IS THE MINIMUM ACCEPTABLE ROLLING AVG RETURN FOR A (TICKER,DATE TO BE KEPT IN CONSIDERATION
            self.min_avg = -1000
        else:
            self.min_avg = min_avg

        if (
            rank is None
        ):  # rank IS SAYING ONLY TO ACCEPT THE TOP (OR BOTTOM) N TICKERS IN TERMS OF ROLLING AVG FOR THAT DAY
            self.rank = None
            self.rankascend = None
        else:
            self.rank = abs(rank)
            self.rankascend = rank < 0

        self.weight_method = weightmethod

    def column_add(self, dataframe):
        checkcols_secondary(dataframe)
        if set([self.colname]).issubset(set(dataframe.columns)) == False:
            dataframe = rolling_calc(
                dataframe,
                "trade_return",
                0,
                "date_exit",
                self.rolling_days,
                self.mintrades,
                "mean",
            )
            dataframe.rename(
                columns={"rolling_trade_return_mean_daycombine": self.colname},
                inplace=True,
            )
        return dataframe

    def weights_col(self, dataframe):
        if self.weight_method == False:
            raise ValueError(
                "Cant call weights_col() while weightmethod argument is False"
            )
        else:
            checkcols_secondary(dataframe)
            if set([self.colname]).issubset(set(dataframe.columns)) == False:
                dataframe = self.column_add(dataframe)

            dataframe = weightingfun(dataframe, self.colname, 1, self.weight_method)
            return dataframe

    def filtered_df(self, dataframe):
        if set([self.colname]).issubset(set(dataframe.columns)):
            checkcols_secondary(dataframe)
        else:
            dataframe = self.column_add(dataframe)

        if self.rankascend is not None:
            dataframe = property_rank(
                dataframe, self.colname, 1, ascending=self.rankascend
            )
            dataframe = dataframe.loc[
                (dataframe[self.colname + "_rank"] <= self.rank)
                & (dataframe[self.colname] >= self.min_avg),
                :,
            ]
        else:
            dataframe = dataframe.loc[(dataframe[self.colname] >= self.min_avg), :]

        return dataframe


# =============================================================================
#  THIS Strategy CLASS IS WHERE YOU COLLECT THE SECONDARY FILTERS YOU WANT TO APPLY,
#  AS WELL AS DECIDE THINGS LIKE enteratopen, shorttrue, AND minhold. THIS STRATEGY OBJECT
#  WILL BE PROVIDED ALONG WITH AN EntryFilter AND ExitFilter OBJECT TO THE BACKTEST
#  CLASS IN ORDER TO PROCESS AND GIVE RESULTS WITH ALL THOSE FACTORS BEING INCLUDED
# =============================================================================
class Strategy:
    def __init__(
        self,
        *secondary_filters,
        shorttrue=False,
        enteratopen=False,
        exitatclose=False,
        hold_min=0
    ):  # THIS IS IN CASE YOU WANT TO HOLD EACH TRADE AT LEAST N DAYS. IF YOU DON'T CARE HOW MANY DAYS, JUST KEEP IT AT 0

        self.secondary_filterz = list(secondary_filters)

        if shorttrue == True:
            self.short_true = -1
        else:
            self.short_true = 1

        if enteratopen == True:
            self.enter_when = "open"
        else:
            self.enter_when = "close"

        if exitatclose == True:
            self.exit_when = "close"
        else:
            self.exit_when = "open"

        # TAKE INTO ACCOUNT HOW MANY DAYS STRATEGY SAYS TO HOLD BEFORE SELLING
        if self.enter_when == "close" and hold_min == 0:
            self.min_hold = 1
        else:
            self.min_hold = hold_min


# =============================================================================
#  THIS BackTest CLASS IS WHERE YOU COMBINE THE ENTRY AND EXIT FILTER OBJECTS YOU MADE,
#  ALONG WITH THE STRATEGY OBJECT YOU JUST DEFINED. THE BACKTEST CLASS HAS METHODS TO FIND OUT
#  HOW THE STRATEGY WOULD HAVE PERFORMED WHEN APPLIED WITH THE ENTRY AND EXIT CONDITIONS PROVIDED
# =============================================================================
class BackTest:
    # WE'RE CALCULATING SHARPE AND SORTINO USING DAILY RETURNS. min_trades IS THE MINIMUM NUMBER OF TRADES
    # WE'RE REQUIRING IN ORDER TO SAY IT'S VALID TO EXTEND THE DAILY RETURNS TO YEARLY. USING TOO SMALL A VALUE
    # FOR min_days WILL ALLOW SOME UNREALISTICALLY SMALL SD AND NEGSD VALUES TO INFLATE SHARPE AND SORTINO
    min_trades = 15

    # sharpe_lookback IS HOW MANY DAYS BACK WE WANT TO USE IN CALCULATING THE EXPECTED VALUE
    # AND THE STANDARD DEVIATION OF DAILY RETURNS
    sharpe_lookback = "364D"

    # lasttrade_ago IS HOW RECENTLY WE'LL REQUIRE THE LAST TRADE OF THIS TICKER USING OUR STRATEGY IN ORDER TO CONSIDER THE
    # SHARPE/ STD ETC. AS ACCEPTABLE. MORE THAN 30 DAYS OR SO AND MAYBE IT WOULD NO LONGER REFLECT THE CURRENT CONDITIONS
    lasttrade_ago = 30

    def __init__(self, entryfilter_obj, exitfilter_obj, strategy_obj):
        if isinstance(entryfilter_obj, EntryFilter):
            self.entry_df = entryfilter_obj.filtered_df.copy()
        else:
            raise TypeError("need for entrydates_obj to be of class EntryFilter")

        if isinstance(exitfilter_obj, ExitFilter):
            self.exit_df = exitfilter_obj.filtered_df.copy()
        else:
            raise TypeError("need for exitdates_obj to be of class ExitFilter")

        if isinstance(strategy_obj, Strategy):
            self.strategy_obj = strategy_obj
        else:
            raise TypeError("need for strategy_obj to be of class Strategy")

        self.all_filters = subclass_find(
            [
                strategy_obj.secondary_filterz.copy(),
                entryfilter_obj.filterz.copy(),
                exitfilter_obj.filterz.copy(),
            ]
        )

        self.allmatches_df = self.entryexit()
        self.performance_df = self.backtest_summary()

    # NOW TO CALCULATE ALL THE MATCHES
    def entryexit(self):
        entry_df = self.entry_df
        entry_df.reset_index(inplace=True)
        entry_df["earliest_date"] = entry_df["date"] + dt.timedelta(
            days=self.strategy_obj.min_hold
        )

        exit_df = self.exit_df

        exit_df.index.names = ["ticker_exit", "date_exit"]
        exit_df.reset_index(inplace=True)

        entryexit_df = pd.merge_asof(
            entry_df.sort_values("earliest_date"),
            exit_df.sort_values("date_exit"),
            left_on="earliest_date",
            right_on="date_exit",
            left_by="ticker",
            right_by="ticker_exit",
            suffixes=["", "_exit"],
            allow_exact_matches=True,
            direction="forward",
        )

        entryexit_df.drop(columns="earliest_date", inplace=True)
        entryexit_df.set_index(["ticker", "date"], inplace=True)
        entryexit_df = entryexit_df.loc[~np.isnat(entryexit_df["date_exit"]), :]
        enterwhen = self.strategy_obj.enter_when
        exitwhen = self.strategy_obj.exit_when + "_exit"
        entryexit_df["trade_return"] = self.strategy_obj.short_true * (
            (entryexit_df[exitwhen] - entryexit_df[enterwhen]) / entryexit_df[enterwhen]
        )

        # DROPPING UNREALISTICALLY HUGE OVERNIGHT GAINS/LOSSES
        entryexit_df = entryexit_df.loc[
            (entryexit_df["trade_return"] <= 0.96)
            & (entryexit_df["trade_return"] >= -0.96),
            :,
        ]

        return entryexit_df

    def add_columns(self):
        toadd_df_roll = self.allmatches_df.copy()
        toadd_df_nonroll = self.allmatches_df.copy()
        #        open_indices = toadd_df.index.copy().to_frame().reset_index(drop=True)
        newcols_roll = []
        newcols_nonroll = []
        oldcols_roll = toadd_df_roll.columns.copy()
        oldcols_nonroll = toadd_df_nonroll.columns.copy()
        fltr_list = self.strategy_obj.secondary_filterz.copy()
        for filtobjs in [x for x in subclass_find(*fltr_list)]:
            if hasattr(filtobjs, "rolling_days"):
                toadd_df_roll = filtobjs.column_add(toadd_df_roll)
                newcols_roll += [
                    item for item in toadd_df_roll.columns if item not in oldcols_roll
                ]
                oldcols_roll = toadd_df_roll.columns.copy()
            else:
                toadd_df_nonroll = filtobjs.column_add(toadd_df_nonroll)
                newcols_nonroll += [
                    item
                    for item in toadd_df_nonroll.columns
                    if item not in oldcols_nonroll
                ]
                oldcols_nonroll = toadd_df_nonroll.columns.copy()

        toadd_df_roll.set_index(["ticker_exit", "date_exit"], inplace=True)
        toadd_df_roll.index.names = ["ticker_exit2", "date_exit2"]
        toadd_df_roll = toadd_df_roll.loc[:, toadd_df_roll.columns.isin(newcols_roll)]
        toadd_df_roll.reset_index(inplace=True)

        toadd_df_nonroll = toadd_df_nonroll.loc[
            :, toadd_df_nonroll.columns.isin(newcols_nonroll)
        ]
        toadd_df_nonroll.reset_index(inplace=True)

        closest_prevclose = pd.merge_asof(
            toadd_df_nonroll.sort_values("date"),
            toadd_df_roll.sort_values("date_exit2"),
            left_on="date",
            right_on="date_exit2",
            left_by="ticker",
            right_by="ticker_exit2",
            suffixes=["", "_exit"],
            allow_exact_matches=(self.strategy_obj.enter_when == "close"),
            direction="backward",
        )

        closest_prevclose["first_acceptable"] = closest_prevclose[
            "date"
        ] - dt.timedelta(days=self.lasttrade_ago)
        closest_prevclose["isacceptable"] = (
            closest_prevclose["date_exit2"] >= closest_prevclose["first_acceptable"]
        )

        return closest_prevclose

    def apply_filters(self):
        tofilter_df = self.allmatches_df.copy()
        fltr_list = self.strategy_obj.secondary_filterz.copy()
        # APPLYING THE SECONDARY FILTERS (SHARPE GREATER THAN X, TOP 3 AVG RETURN THAT TYPE THING)
        if len(self.strategy_obj.secondary_filterz) > 0:
            w_newcolumns = self.add_columns().set_index(["ticker", "date"])
            tofilter_df = pd.concat(
                [tofilter_df, w_newcolumns], axis=1, ignore_index=False
            )
            tofilter_df = tofilter_df.loc[tofilter_df["isacceptable"] == True, :]
            for fltrs in fltr_list:
                tofilter_df = fltrs.filtered_df(tofilter_df)

        # ADDING THE WEIGHTS COLUMN ACCORDING TO THE FILTER THAT WAS GIVEN A weight_method OTHER THAN FALSE
        try:
            weight_filt = next(
                x
                for x in self.all_filters
                if (hasattr(x, "weight_method") and x.weight_method != False)
            )
            tofilter_df = weight_filt.weights_col(tofilter_df)
        except StopIteration:
            weight_filt = next(
                x for x in self.all_filters if (hasattr(x, "weight_method"))
            )
            weight_filt.weight_method = None
            tofilter_df = weight_filt.weights_col(tofilter_df)

        # BREAKING TIES USING THE SAME WEIGHT CRITERIA
        try:
            rankfilt = next(
                x
                for x in self.all_filters
                if ((hasattr(x, "rank")) and (x.rank is not None))
            )
            rank_use = rankfilt.rank
            rank_colname = rankfilt.colname
            try:
                tofilter_df = property_rank(
                    tofilter_df,
                    "weighting",
                    ["date", rank_colname + "_lag_rank"],
                    method="first",
                    ascending=False,
                )
                tofilter_df = tofilter_df.loc[
                    (tofilter_df["weighting_rank"] <= rank_use), :
                ]
                tofilter_df = (
                    tofilter_df.sort_values(
                        [rank_colname + "_lag_rank", "weighting_rank"],
                        ascending=[True, True],
                    )
                    .groupby(level=1)
                    .head(rank_use)
                    .copy()
                )
            except:
                tofilter_df = property_rank(
                    tofilter_df,
                    "weighting",
                    ["date", rank_colname + "_rank"],
                    method="first",
                    ascending=False,
                )
                tofilter_df = tofilter_df.loc[
                    (tofilter_df["weighting_rank"] <= rank_use), :
                ]
                tofilter_df = (
                    tofilter_df.sort_values(
                        [rank_colname + "_rank", "weighting_rank"],
                        ascending=[True, True],
                    )
                    .groupby(level=1)
                    .head(rank_use)
                    .copy()
                )

            tofilter_df = weight_filt.weights_col(tofilter_df)

            if tofilter_df.index.names[1] != "date":
                raise ValueError("index got messed up")

        except:
            print("no ranking involved")

        # CALCULATING TRADES CARRIED OVER FROM PREVIOUS DAYS AND HOW MUCH OF THE 1 DOLLAR THAT LEAVES US TO WORK WITH
        tofilter_df["index_col"] = tofilter_df.index.get_level_values(1)
        tofilter_df["allocation_for_day"] = 1
        tofilter_df["allocation_for_trade"] = (
            tofilter_df["allocation_for_day"] * tofilter_df["weighting"]
        )
        active_df = tofilter_df.groupby(level=1)["allocation_for_day"].last().to_frame()
        for entrdts in active_df.index:
            active_df.loc[entrdts, "allocation_for_day"] = (
                active_df.loc[entrdts, "allocation_for_day"]
                - tofilter_df.loc[
                    (tofilter_df["date_exit"] > entrdts)
                    & (tofilter_df["index_col"] < entrdts),
                    "allocation_for_trade",
                ].sum()
            )
            tofilter_df.loc[
                pd.IndexSlice[:, entrdts], "allocation_for_day"
            ] = active_df.loc[entrdts, "allocation_for_day"]
            tofilter_df["allocation_for_trade"] = (
                tofilter_df["allocation_for_day"] * tofilter_df["weighting"]
            )

        tofilter_df["trade_return_weighted"] = (
            tofilter_df["trade_return"] * tofilter_df["allocation_for_trade"]
        )
        tofilter_df.drop("index_col", axis=1, inplace=True)
        return tofilter_df

    def backtest_summary(self):
        summary_df = self.apply_filters()

        summary_df["money_freed_up_dollars"] = summary_df.groupby("date_exit")[
            "allocation_for_trade"
        ].transform("sum")
        summary_df["days_return_dollars"] = summary_df.groupby("date_exit")[
            "trade_return_weighted"
        ].transform("sum")

        summary_df["days_return_percent"] = (
            summary_df["days_return_dollars"] / summary_df["money_freed_up_dollars"]
        )

        summary_df["trade_held"] = (
            summary_df["date_exit"] - summary_df.index.get_level_values(1)
        ) + dt.timedelta(days=0 + (self.strategy_obj.enter_when == "open"))

        summary_df.sort_values("date_exit", inplace=True)
        summary_df["trade_return_weighted_cumulative"] = summary_df[
            "trade_return_weighted"
        ].cumsum()

        return summary_df

    def plot_returns(self, *toplot):
        toplot_df = self.performance_df.copy()

        drawdown_dict = drawdown_stats(
            toplot_df.set_index("date_exit")["trade_return_weighted_cumulative"].copy()
        )

        tradedays_year = (
            toplot_df.index.get_level_values(1)
            .to_series()
            .resample("364D")
            .nunique()[0:-1]
            .mean()
        )

        print(tradedays_year)

        days_return_dollars_1st = toplot_df.groupby("date_exit")[
            "days_return_dollars"
        ].transform(lambda x: x.iloc[0])

        unscale_sharpe = days_return_dollars_1st.mean() / days_return_dollars_1st.std()

        unscale_sortino = (
            days_return_dollars_1st.mean()
            / days_return_dollars_1st[days_return_dollars_1st < 0].std()
        )

        plotdaterange = pd.date_range(
            start=toplot_df.index[0][1] - dt.timedelta(days=365),
            end=toplot_df.index[-1][1],
            freq="YS",
        )
        plotreturnrange = [
            x
            for x in range(
                int(toplot_df["trade_return_weighted_cumulative"].min()),
                int(toplot_df["trade_return_weighted_cumulative"].max()) + 2,
            )
        ]

        for dataseries in toplot:
            plt.figure(figsize=(10, 7))
            plt.tight_layout()
            plt.plot(toplot_df["date_exit"], toplot_df[dataseries])
            plt.xticks(ticks=plotdaterange, labels=plotdaterange.year, rotation=30)
            plt.yticks(plotreturnrange)
            plt.grid()
            plt.xlabel("Date")
            plt.ylabel(dataseries)
            plt.title(
                "trades executed: "
                + str(len(toplot_df.index))
                #'Avg positions held per day: ' + str(np.round(toplot_df.groupby('date_exit',group_keys=False)['active_trades'].nth(0)[0:-1].mean(),4)) +
                # "\nAvg Days Trade Held: "
                # + str(toplot_df["trade_held"].mean())[0:12]
                # + " hours"
                + "\nPercent Days in Any Trade: "
                + str(
                    np.round(
                        (
                            len(toplot_df.index.get_level_values(1).unique())
                            / len(
                                pd.bdate_range(
                                    toplot_df.index.get_level_values(1).min(),
                                    toplot_df.index.get_level_values(1).max(),
                                )
                            )
                        )
                        * 100,
                        3,
                    )
                )
                + "%"
                + "\nAvg Return per Day: "
                + str(np.round(days_return_dollars_1st.mean() * 100, 3,))
                + "%"
                + "  Std of Daily Returns: "
                + str(np.round(days_return_dollars_1st.std(), 5,))
                +
                #'\nStd of All Trades: '+str(np.round(toplot_df['trade_return'].std(),4)) +
                "\nSharpe: "
                + str(np.round(unscale_sharpe * np.sqrt(tradedays_year), 4))
                + "  Sortino: "
                + str(np.round(unscale_sortino * np.sqrt(tradedays_year), 4))
                + "\nMax Drawdown: "
                + str(np.round(drawdown_dict["max drawdown"], 4))
                + "  Recovery Time: "
                + str(drawdown_dict["recovery time"])
            )

        # print(
        #     toplot_df.set_index("date_exit")["trade_return_weighted_cumulative"]
        #     .resample("YS")
        #     .last()
        #     - toplot_df.set_index("date_exit")["trade_return_weighted_cumulative"]
        #     .resample("YS")
        #     .first()
        # )


class ComboResult:
    def __init__(self, *backtests):
        if all([isinstance(x, BackTest) for x in backtests]):
            self.backtest_list = list(backtests)
        else:
            raise TypeError("need for backtests to be of class BackTest")

        self.performance_df = self.combo_summary()

    def combo_summary(self):
        enterwhen = self.backtest_list[0].strategy_obj.enter_when == "open"
        df_list = []
        countr = 0
        for bcktst in self.backtest_list:
            df = bcktst.apply_filters()
            df["strategy_x"] = countr
            df_list.append(df)
            countr += 1

        summary_df = pd.concat(df_list, axis=0, join="outer", sort=False, copy=False)
        multiplr = 1 / len(self.backtest_list)

        summary_df["allocation_for_trade"] = (
            summary_df["allocation_for_trade"] * multiplr
        )
        summary_df["trade_return_weighted"] = (
            summary_df["trade_return_weighted"] * multiplr
        )
        summary_df["weighting"] = summary_df["weighting"] * multiplr
        summary_df["trade_return"] = summary_df["trade_return"] * multiplr

        summary_df["money_freed_up_dollars"] = summary_df.groupby("date_exit")[
            "allocation_for_trade"
        ].transform("sum")
        summary_df["days_return_dollars"] = summary_df.groupby("date_exit")[
            "trade_return_weighted"
        ].transform("sum")

        summary_df["days_return_percent"] = (
            summary_df["days_return_dollars"] / summary_df["money_freed_up_dollars"]
        )

        summary_df["trade_held"] = (
            summary_df["date_exit"]
            - summary_df.index.get_level_values(1)
            + dt.timedelta(days=0 + enterwhen)
        )
        summary_df.sort_values("date_exit", inplace=True)
        summary_df["trade_return_weighted_cumulative"] = summary_df[
            "trade_return_weighted"
        ].cumsum()

        return summary_df

    def plot_returns(self, *toplot):
        toplot_df = self.performance_df.copy()

        drawdown_dict = drawdown_stats(
            toplot_df.set_index("date_exit")["trade_return_weighted_cumulative"].copy()
        )

        tradedays_year = (
            toplot_df.index.get_level_values(1)
            .to_series()
            .resample("364D")
            .nunique()[0:-1]
            .mean()
        )

        days_return_dollars_1st = toplot_df.groupby("date_exit")[
            "days_return_dollars"
        ].transform(lambda x: x.iloc[0])

        unscale_sharpe = days_return_dollars_1st.mean() / days_return_dollars_1st.std()

        unscale_sortino = (
            days_return_dollars_1st.mean()
            / days_return_dollars_1st[days_return_dollars_1st < 0].std()
        )

        plotdaterange = pd.date_range(
            start=toplot_df.index[0][1] - dt.timedelta(days=365),
            end=toplot_df.index[-1][1],
            freq="1Y",
        )
        plotreturnrange = [
            x
            for x in range(
                int(toplot_df["trade_return_weighted_cumulative"].min()),
                int(toplot_df["trade_return_weighted_cumulative"].max()) + 2,
            )
        ]

        for dataseries in toplot:
            plt.figure(figsize=(10, 7))
            plt.tight_layout()
            plt.plot(toplot_df["date_exit"], toplot_df[dataseries])
            plt.xticks(ticks=plotdaterange, labels=plotdaterange.year + 1, rotation=30)
            plt.yticks(plotreturnrange)
            plt.grid()
            plt.xlabel("Date")
            plt.ylabel(dataseries)
            plt.title(
                "trades executed: "
                + str(len(toplot_df.index))
                #'Avg positions held per day: ' + str(np.round(toplot_df.groupby('date_exit',group_keys=False)['active_trades'].nth(0)[0:-1].mean(),4)) +
                # "\nAvg Days Trade Held: "
                # + str(toplot_df["trade_held"].mean())[0:12]
                # + " hours"
                + "\nPercent Days in Any Trade: "
                + str(
                    np.round(
                        (
                            len(toplot_df.index.get_level_values(1).unique())
                            / len(
                                pd.bdate_range(
                                    toplot_df.index.get_level_values(1).min(),
                                    toplot_df.index.get_level_values(1).max(),
                                )
                            )
                        )
                        * 100,
                        3,
                    )
                )
                + "%"
                + "\nAvg Return per Day: "
                + str(np.round(days_return_dollars_1st.mean() * 100, 3,))
                + "%"
                + "  Std of Daily Returns: "
                + str(np.round(days_return_dollars_1st.std(), 5,))
                +
                #'\nStd of All Trades: '+str(np.round(toplot_df['trade_return'].std(),4)) +
                "\nSharpe: "
                + str(np.round(unscale_sharpe * np.sqrt(tradedays_year), 4))
                + "  Sortino: "
                + str(np.round(unscale_sortino * np.sqrt(tradedays_year), 4))
                + "\nMax Drawdown: "
                + str(np.round(drawdown_dict["max drawdown"], 4))
                + "  Recovery Time: "
                + str(drawdown_dict["recovery time"])
            )

        # print(
        #     toplot_df.set_index("date_exit")["trade_return_weighted_cumulative"]
        #     .resample("YS")
        #     .last()
        #     - toplot_df.set_index("date_exit")["trade_return_weighted_cumulative"]
        #     .resample("YS")
        #     .first()
        # )

