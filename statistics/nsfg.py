import sys
import numpy as np
from collections import defaultdict

import stats

def ReadFemResp(dct_file='dataset/2002FemResp.dct', dat_file='dataset/2002FemResp.dat.gz', nrows=None):
    """
    NSFG respondent データを読み込む

    dct_file: ファイル名
    dat_file: ファイル名

    returns: DataFrame
    """
    dct = stats.ReadStataDct(dct_file)
    df = dct.ReadFixedWidth(dat_file, compression='gzip', nrows=nrows)
    CleanFemResp(df)
    return df

def CleanFemResp(df):
    """
    respondentデータ内の変数を再コードする

    df: DataFrame
    """
    pass

def ReadFemPreg(dct_file='dataset/2002FemPreg.dct', dat_file='dataset/2002FemPreg.dat.gz'):
    """
    NSFG pregnancy データを読み込む

    dct_file: ファイル名
    dat_file: ファイル名

    returns: DataFrame
    """
    dct = stats.ReadStataDct(dct_file)
    df = dct.ReadFixedWidth(dat_file, compression='gzip')
    CleanFemPreg(df)
    return df

def CleanFemPreg(df):
    """
    データクリーニングを行う関数
    pregnancyデータ内の変数を再コードする

    df: DataFrame
    """
    df.agepreg /= 100.0

    df.loc[df.birthwgt_lb > 20, 'birthwgt'] = np.nan

    na_vals = [97, 98, 99]

    df.replace(
        to_replace = {
            "birthwgt_lb": na_vals,
            "birthwgt_oz": na_vals,
            "hpagelb": na_vals,
            "babysex": [7, 9],
            "nbrnaliv": [9],
        },
        value = np.nan,
        inplace = True
    )

    df['totalwgt_lb'] = df.birthwgt_lb + df.birthwgt_oz / 16.0

    df.cmintvw = np.nan

def ValidatePregnum(resp, preg):
    preg_map = MakePregMap(preg)

    for index, pregnum in resp.pregnum.items():
        caseid = resp.caseid[index]
        indices = preg_map[caseid]

        if len(indices) != pregnum:
            print(caseid, len(indices), pregnum)
            return False
        
    return True

def MakePregMap(df):
    d = defaultdict(list)
    for index, caseid in df.caseid.items():
        d[caseid].append(index)
    return d

def main():

    resp = ReadFemResp()

    assert(len(resp) == 7643)
    assert(resp.pregnum.value_counts()[1] == 1267)

    preg = ReadFemPreg()
    print(preg.shape)

    assert len(preg) == 13593
    assert preg.caseid[13592] == 12571
    assert preg.pregordr.value_counts()[1] == 5033
    assert preg.nbrnaliv.value_counts()[1] == 8981
    assert preg.babysex.value_counts()[1] == 4641
    assert preg.birthwgt_lb.value_counts()[7] == 3049
    assert preg.birthwgt_oz.value_counts()[0] == 1037
    assert preg.prglngth.value_counts()[39] == 4744
    assert preg.outcome.value_counts()[1] == 9148
    assert preg.birthord.value_counts()[1] == 4413
    assert preg.agepreg.value_counts()[22.75] == 100
    assert preg.totalwgt_lb.value_counts()[7.5] == 302

    weights = preg.finalwgt.value_counts()
    key = max(weights.keys())
    assert preg.finalwgt.value_counts()[key] == 6

    assert(ValidatePregnum(resp, preg))

    print('All tests passed')

if __name__ == '__main__':
    main()