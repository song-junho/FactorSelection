import pandas as pd
import pickle
from tqdm import tqdm
import numpy as np
import asset
from scipy import stats

from abc import *
import threading
from multiprocessing import Queue
from sklearn.preprocessing import RobustScaler
from lib import get_list_mkt_date, get_list_eom_date
from concurrent.futures import ThreadPoolExecutor, wait
from collections import deque


class Stock(metaclass=ABCMeta):

    def __init__(self, start_date, end_date=asset.date_today):
        self.start_date = start_date
        self.end_date = end_date
        self.list_date_mkt = get_list_mkt_date(start_date, end_date)
        self.list_date_eom = get_list_eom_date(self.list_date_mkt)


    @staticmethod
    def get_list_quantile(data, q_count=5):
        interval = 1 / q_count
        quantile_list = np.arange(0, 1, interval)

        list_res = []
        for q in quantile_list:
            list_res.append(data.quantile(q=q, interpolation='nearest'))

        return list_res

    @staticmethod
    def apply_quantile(x, list_quantile):

        max_index = len(list_quantile) - 1
        for i, q in enumerate(list_quantile):

            if i < max_index:
                if (list_quantile[i] <= x) and (x < list_quantile[i + 1]):
                    break
            elif i == max_index:
                break
        return i

    @staticmethod
    def get_z_score(data):

        z_score = stats.zscore(data)

        return z_score

    def scoring(self, df_factor):

        df_factor["quantile"] = 0
        df_factor["z_score"] = 0
        df_factor["val"] = df_factor["val"].astype("float")
        list_date = sorted(df_factor["date"].unique())
        for v_date in tqdm(list_date):
            z_score = self.get_z_score(df_factor.loc[df_factor["date"] == v_date, "val"])
            df_factor.loc[df_factor["date"] == v_date, "z_score"] = z_score
            list_quantile = self.get_list_quantile(df_factor.loc[df_factor["date"] == v_date, "z_score"], 5)
            df_factor.loc[df_factor["date"] == v_date, "quantile"] = df_factor.loc[
                df_factor["date"] == v_date, "z_score"].apply(lambda x: self.apply_quantile(x, list_quantile))

        return df_factor

    @abstractmethod
    def create_factor_data(self):
        pass


class Value(Stock):

    # save data
    with open(r'D:\MyProject\QuantModeling.pickle', 'rb') as fr:
        df_ec_phase = pickle.load(fr)

    # save data
    with open(r'D:\MyProject\종목분석_환경\multiple_DB\dict_multiple_his_cmp_cd.pickle', 'rb') as fr:
        dict_multiple_his_cmp_cd = pickle.load(fr)

    # save data
    with open(r'D:\MyProject\종목분석_환경\multiple_DB\dict_multiple_cmp_cd.pickle', 'rb') as fr:
        dict_multiple_cmp_cd = pickle.load(fr)

    # factor_list
    factor_list = {
        "raw":[
            ("por", 900003)
        ],
        "cagr":[
            ("por_cagr", 900003)
        ],
        "spread":[
            ("por_spr", 900003)
        ]
    }

    def factor_raw(self, factor_cd, factor_nm):
        '''
        실적 Raw 데이터 기준 Value 팩터
        :return:
        '''

        df_factor = pd.DataFrame(columns = ["date", "cmp_cd", "val"])

        list_cmp_cd = sorted(self.dict_multiple_cmp_cd.keys())
        for cmp_cd in tqdm(list_cmp_cd):
            df = self.dict_multiple_cmp_cd[cmp_cd]
            df = df[df["item_cd"] == factor_cd]
            df = df[df["date"].isin(self.list_date_eom)].sort_values(["date", "cmp_cd"])[["date", "cmp_cd", "multiple"]]
            df = df.rename(columns={"multiple": "val"})[["date", "cmp_cd", "val"]]

            df_factor = pd.concat([df_factor, df])

        df_factor = df_factor[df_factor["val"] > 0]
        df_factor = df_factor.reset_index(drop=True)
        df_factor = self.scoring(df_factor)

        df_factor["item_nm"] = factor_nm
        df_factor = df_factor[["date", "cmp_cd", "item_nm", "val", "z_score", "quantile"]]

        return df_factor

    def factor_cagr(self, factor_cd, factor_nm):
        '''
        실적 Raw 데이터 기준 Value 팩터
        :return:
        '''

        df_factor = pd.DataFrame(columns = ["date", "cmp_cd", "val"])

        list_cmp_cd = sorted(self.dict_multiple_his_cmp_cd.keys())
        for cmp_cd in tqdm(list_cmp_cd):
            df = self.dict_multiple_his_cmp_cd[cmp_cd]
            df = df[df["item_cd"] == factor_cd]
            df = df[df["value_q"] == 0.5]
            df = df[df["date"].isin(self.list_date_eom)].sort_values(["date", "cmp_cd"])[["date", "cmp_cd", "history_multiple"]]
            df = df.rename(columns={"history_multiple": "val"})[["date", "cmp_cd", "val"]]

            df_factor = pd.concat([df_factor, df])

        df_factor = df_factor[df_factor["val"] > 0]
        df_factor = df_factor.reset_index(drop=True)
        df_factor = self.scoring(df_factor)

        df_factor["item_nm"] = factor_nm
        df_factor = df_factor[["date", "cmp_cd", "item_nm", "val", "z_score", "quantile"]]

        return df_factor

    def factor_spread(self, factor_cd, factor_nm):
        '''
        실적 Raw 데이터 기준 Value 팩터
        :return:
        '''

        df_factor = pd.DataFrame(columns = ["date", "cmp_cd", "val"])

        list_cmp_cd = sorted(self.dict_multiple_his_cmp_cd.keys())
        for cmp_cd in tqdm(list_cmp_cd):
            df = self.dict_multiple_his_cmp_cd[cmp_cd]
            df = df[df["item_cd"] == factor_cd]
            df = df[df["value_q"] == 0.5]
            df = df[df["date"].isin(self.list_date_eom)].sort_values(["date","cmp_cd"])[["date", "cmp_cd", "upside"]]
            df = df.rename(columns={"upside": "val"})[["date", "cmp_cd", "val"]]

            df_factor = pd.concat([df_factor, df])

        df_factor = df_factor[df_factor["val"] > 0]
        df_factor = df_factor.reset_index(drop=True)
        df_factor = self.scoring(df_factor)

        df_factor["item_nm"] = factor_nm
        df_factor = df_factor[["date", "cmp_cd", "item_nm", "val", "z_score", "quantile"]]

        return df_factor

    def create_factor_data(self):

        df_factor_data = pd.DataFrame(columns = ["date", "cmp_cd", "item_nm", "val", "z_score", "quantile"])

        factor_types = self.factor_list.keys()
        for f_type in factor_types:

            for factor in self.factor_list[f_type]:

                factor_nm = factor[0]
                factor_cd = factor[1]

                df = pd.DataFrame()
                if f_type == "raw":
                    df = self.factor_raw(factor_cd, factor_nm)
                elif f_type == "cagr":
                    df = self.factor_cagr(factor_cd, factor_nm)
                elif f_type == "spread":
                    df = self.factor_spread(factor_cd, factor_nm)
                df_factor_data = pd.concat([df_factor_data, df])

        # save data
        with open(r'D:\MyProject\FactorSelection\stock_factor_value_quantiling.pickle', 'wb') as fw:
            pickle.dump(df_factor_data, fw)


class Growth(Stock):
    # 재무데이터
    list_item_cd = (121000, 121500)
    q = 'SELECT * FROM financial_data.financial_statement_q' + ' where item_cd in {}'.format(list_item_cd)
    df_fin_q = pd.read_sql_query(q, asset.conn).sort_values(['cmp_cd', 'yymm', 'fin_typ', 'freq']).drop_duplicates(
        ["term_typ", "cmp_cd", "item_cd", "yymm", "freq"], keep="last").reset_index(drop=True)

    # factor_list
    factor_list = {
        "op_qoq": 121500,
        "op_yoy": 121500
    }

    list_q = deque([])
    _lock = threading.Lock()

    def preprocessing(self, item_nm, item_cd, list_cmp_cd):

        target_col = "change_pct"

        df_fin = self.df_fin_q[self.df_fin_q["item_cd"] == item_cd]
        df_fin = df_fin[df_fin["freq"] == item_nm.split("_")[1]]
        df_fin = df_fin[["cmp_cd", "yymm", target_col]]

        # 변화율 스케일링
        scaler = RobustScaler()
        data = np.array(df_fin[target_col]).reshape(-1, 1)
        df_fin[target_col] = scaler.fit_transform(data)

        for cmp_cd in tqdm(list_cmp_cd):

            df_cmp = df_fin[df_fin["cmp_cd"] == cmp_cd].copy()
            df_cmp.loc[:, "cagr_pct"] = 0
            df_cmp.loc[:, "spread"] = 0

            df_cmp["cagr_pct"] = df_cmp[target_col].rolling(window=12).mean()
            df_cmp.loc[df_cmp["cagr_pct"] == 0, "cagr_pct"] = 0.1

            df_cmp["spread"] = (df_cmp[target_col] - df_cmp["cagr_pct"]) / abs(df_cmp["cagr_pct"]) + 1
            df_cmp = df_cmp[~df_cmp["cagr_pct"].isna()]
            df_cmp = df_cmp[df_cmp["yymm"] >= 200509]

            df_factor = pd.DataFrame()
            for v_date in self.list_date_eom:

                year = pd.to_datetime(v_date).year
                month = pd.to_datetime(v_date).month

                yymm = 0
                if month in (3, 4):
                    yymm = ((year - 1) * 100) + 12
                elif month in (5, 6, 7):
                    yymm = (year * 100) + 3
                elif month in (8, 9, 10):
                    yymm = (year * 100) + 6
                elif month in (11, 12):
                    yymm = (year * 100) + 9
                elif month in (1, 2):
                    yymm = ((year - 1) * 100) + 9

                if len(df_cmp.loc[df_cmp["yymm"] == yymm]) == 0:
                    continue

                df = df_cmp.loc[df_cmp["yymm"] == yymm]

                target_val = df[target_col].values[0]
                cagr_pct = df["cagr_pct"].values[0]
                spread = df["spread"].values[0]

                # 스프레드값이 Nan 값인 경우
                if pd.isna(spread):
                    continue

                df_factor = pd.concat([df_factor, pd.DataFrame([[v_date, cmp_cd, 0, target_val],
                                                                [v_date, cmp_cd, 1, cagr_pct],
                                                                [v_date, cmp_cd, 2, spread]])])

            with self._lock:
                self.list_q.append(df_factor)

    def get_raw_data(self,item_nm, item_cd):

        n = 500
        list_cmp_cd_t = sorted(self.df_fin_q["cmp_cd"].unique())
        list_cmp_cd_t = [list_cmp_cd_t[i * n:(i + 1) * n] for i in range((len(list_cmp_cd_t) + n - 1) // n)]

        threads = []
        with ThreadPoolExecutor(max_workers=5) as executor:

            for list_cmp_cd in list_cmp_cd_t:
                threads.append(executor.submit(self.preprocessing, item_nm, item_cd, list_cmp_cd))
            wait(threads)

        df_raw_data = pd.concat(self.list_q)
        self.list_q = deque([])

        df_raw_data.columns = ["date", "cmp_cd", "item_nm", "val"]

        return df_raw_data

    def get_factor_data(self, df_raw_data, factor_nm):

        df_factor = df_raw_data[df_raw_data["item_nm"] == factor_nm][["date", "cmp_cd", "val"]].reset_index(
            drop=True)

        df_factor = self.scoring(df_factor)
        df_factor["item_nm"] = factor_nm
        df_factor = df_factor[["date", "cmp_cd", "item_nm", "val", "z_score", "quantile"]]

        return df_factor

    def create_factor_data(self):

        df_factor_data = pd.DataFrame(columns = ["date", "cmp_cd", "item_nm", "val", "z_score", "quantile"])

        for factor in self.factor_list.keys():

            factor_nm = factor
            factor_cd = self.factor_list[factor_nm]

            # Growth 데이터 생성 및 Factor 명 변경(원본, cagr, spread)
            df_raw_data = self.get_raw_data(factor_nm, factor_cd)
            df_raw_data["item_nm"] = df_raw_data["item_nm"].replace([0, 1, 2], [factor_nm, factor_nm + "_cagr",
                                                                                      factor_nm + "_spread"])
            detail_factor_list = list(df_raw_data["item_nm"].unique())

            for detail_factor in detail_factor_list:
                df = self.get_factor_data(df_raw_data, detail_factor)
                df_factor_data = pd.concat([df_factor_data, df])

        # save data
        with open(r'D:\MyProject\FactorSelection\stock_factor_growth_quantiling.pickle', 'wb') as fw:
            pickle.dump(df_factor_data, fw)


class Size(Stock):

    # 가격 데이터
    with open(r"D:\MyProject\StockPrice\DictDfStockDaily.pickle", 'rb') as fr:
        dict_df_stock_daily = pickle.load(fr)

    # factor_list
    factor_list = {
        "market_cap": '',
    }

    def get_raw_data(self, factor_nm):

        df_size_data = pd.DataFrame()

        for p_date in tqdm(self.list_date_eom):
            df_stock_daily = self.dict_df_stock_daily[p_date].reset_index()
            df_stock_daily = df_stock_daily.rename(
                columns={df_stock_daily.columns[0]: "date", "StockCode": "cmp_cd", "MarketCap": "val"})
            df_stock_daily = df_stock_daily[["date", "cmp_cd", "val"]]

            df_size_data = pd.concat([df_size_data, df_stock_daily])

        df_size_data["item_nm"] = factor_nm

        return df_size_data

    def get_factor_data(self, df_factor_data, factor_nm):

        df_factor = df_factor_data[df_factor_data["item_nm"] == factor_nm][["date", "cmp_cd", "val"]].reset_index(
            drop=True)

        df_factor = self.scoring(df_factor)
        df_factor["item_nm"] = factor_nm
        df_factor = df_factor[["date", "cmp_cd", "item_nm", "val", "z_score", "quantile"]]

        return df_factor

    def create_factor_data(self):

        df_factor_data = pd.DataFrame(columns = ["date", "cmp_cd", "item_nm", "val", "z_score", "quantile"])

        factor_nm = list(self.factor_list.keys())[0]

        # size 데이터
        df_size_data = self.get_raw_data(factor_nm)

        df_factor_data = self.get_factor_data(df_size_data, factor_nm)

        # save data
        with open(r'D:\MyProject\FactorSelection\stock_factor_size_quantiling.pickle', 'wb') as fw:
            pickle.dump(df_factor_data, fw)


class Quality(Stock):

    # 재무데이터
    list_item_cd = (211300, 211000, 211500)
    q = 'SELECT * FROM financial_data.financial_statement_q' + ' where item_cd in {}'.format(list_item_cd) + 'and freq = "yoy"'
    df_fin_q = pd.read_sql_query(q, asset.conn).sort_values(['cmp_cd', 'yymm', 'fin_typ', 'freq']).drop_duplicates(
        ["term_typ", "cmp_cd", "item_cd", "yymm", "freq"], keep="last").reset_index(drop=True)

    # factor_list
    factor_list = {
        "gpm": 211300,
        "opm": 211000,
        "roe": 211500
    }

    list_q = deque([])
    _lock = threading.Lock()

    def preprocessing(self, item_nm, item_cd, list_cmp_cd):

        target_col = "val"

        df_fin = self.df_fin_q[self.df_fin_q["item_cd"] == item_cd]
        df_fin = df_fin[["cmp_cd", "yymm", target_col]]

        for cmp_cd in tqdm(list_cmp_cd):

            df_cmp = df_fin[df_fin["cmp_cd"] == cmp_cd].copy()
            df_cmp.loc[:, "cagr_pct"] = 0
            df_cmp.loc[:, "spread"] = 0

            df_cmp["cagr_pct"] = df_cmp[target_col].rolling(window=12).mean()
            df_cmp.loc[df_cmp["cagr_pct"] == 0, "cagr_pct"] = 0.1

            df_cmp["spread"] = (df_cmp[target_col] - df_cmp["cagr_pct"]) / abs(df_cmp["cagr_pct"]) + 1
            df_cmp = df_cmp[~df_cmp["cagr_pct"].isna()]
            df_cmp = df_cmp[df_cmp["yymm"] >= 200509]

            df_factor = pd.DataFrame()
            for v_date in self.list_date_eom:

                year = pd.to_datetime(v_date).year
                month = pd.to_datetime(v_date).month

                yymm = 0
                if month in (3, 4):
                    yymm = ((year - 1) * 100) + 12
                elif month in (5, 6, 7):
                    yymm = (year * 100) + 3
                elif month in (8, 9, 10):
                    yymm = (year * 100) + 6
                elif month in (11, 12):
                    yymm = (year * 100) + 9
                elif month in (1, 2):
                    yymm = ((year - 1) * 100) + 9

                if len(df_cmp.loc[df_cmp["yymm"] == yymm]) == 0:
                    continue

                df = df_cmp.loc[df_cmp["yymm"] == yymm]

                target_val = df[target_col].values[0]
                cagr_pct = df["cagr_pct"].values[0]
                spread = df["spread"].values[0]

                # 스프레드값이 Nan 값인 경우
                if pd.isna(spread):
                    continue

                df_factor = pd.concat([df_factor, pd.DataFrame([[v_date, cmp_cd, 0, target_val],
                                                                [v_date, cmp_cd, 1, cagr_pct],
                                                                [v_date, cmp_cd, 2, spread]])])

            with self._lock:
                self.list_q.append(df_factor)

    def get_raw_data(self,item_nm, item_cd):

        n = 50
        list_cmp_cd_t = sorted(self.df_fin_q["cmp_cd"].unique())[:100]
        list_cmp_cd_t = [list_cmp_cd_t[i * n:(i + 1) * n] for i in range((len(list_cmp_cd_t) + n - 1) // n)]

        threads = []
        with ThreadPoolExecutor(max_workers=5) as executor:

            for list_cmp_cd in list_cmp_cd_t:
                threads.append(executor.submit(self.preprocessing, item_nm, item_cd, list_cmp_cd))
            wait(threads)

        df_raw_data = pd.concat(self.list_q)
        self.list_q = deque([])

        df_raw_data.columns = ["date", "cmp_cd", "item_nm", "val"]

        return df_raw_data

    def get_factor_data(self, df_raw_data, factor_nm):

        df_factor = df_raw_data[df_raw_data["item_nm"] == factor_nm][["date", "cmp_cd", "val"]].reset_index(
            drop=True)

        df_factor = self.scoring(df_factor)
        df_factor["item_nm"] = factor_nm
        df_factor = df_factor[["date", "cmp_cd", "item_nm", "val", "z_score", "quantile"]]

        return df_factor

    def create_factor_data(self):

        df_factor_data = pd.DataFrame(columns = ["date", "cmp_cd", "item_nm", "val", "z_score", "quantile"])

        for factor_nm, factor_cd in self.factor_list.items():

            # RAW 데이터 생성 및 Factor 명 변경(원본, cagr, spread)
            df_raw_data = self.get_raw_data(factor_nm, factor_cd)
            df_raw_data["item_nm"] = df_raw_data["item_nm"].replace([0, 1, 2], [factor_nm, factor_nm + "_cagr",
                                                                                      factor_nm + "_spread"])
            detail_factor_list = list(df_raw_data["item_nm"].unique())

            for detail_factor in detail_factor_list:
                df = self.get_factor_data(df_raw_data, detail_factor)
                df_factor_data = pd.concat([df_factor_data, df])

        # save data
        with open(r'D:\MyProject\FactorSelection\stock_factor_quality_quantiling.pickle', 'wb') as fw:
            pickle.dump(df_factor_data, fw)
