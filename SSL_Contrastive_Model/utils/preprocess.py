from bs4 import BeautifulSoup
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import sparse
from scipy.sparse.linalg import spsolve
import numpy as np
import SimpleITK as sitk   
import torch
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

def process_snp(tensor):
    field_list = ['24120','24118','24119','24123','24121','24122','24141','24150','24153','24155','24142','24145',
                  '24148','24157','24177','24179','24180','24158','24167','24168','24162','24163','24174','24113','24100','24103','24101','24105',
                '24102','24106','24109','24107','24108','24124','24133','24134','24135','24136','24137','24138','24139','24125',
                '24126','24128','24129','24130','24131','24132','24140']



    snp_dict = {'AAo_aortic_distensibility': [36], 
                'AAo_max_area': [84, 89, 101, 102, 103, 105, 112, 116, 120, 121, 136, 137, 139, 141, 140, 156, 8, 23, 22, 28, 31, 33, 35, 37, 39, 40, 45, 52, 53, 56, 57, 58, 61, 63, 64, 69, 71, 74, 76, 77], 
                'AAo_min_area': [83, 88, 101, 102, 103, 104, 106, 112, 116, 119, 121, 136, 138, 141, 140, 156, 8, 23, 22, 21, 28, 31, 33, 34, 35, 37, 38, 41, 45, 46, 52, 53, 56, 60, 62, 63, 64, 70, 73, 77],
                'DAo_aortic_distensibility': [87, 78], 
                'DAo_max_area': [82, 86, 113, 117, 118, 122, 151, 5, 9, 30, 49, 55],
                'DAo_min_area': [81, 85, 86, 117, 118, 122, 135, 151, 5, 9, 30, 49, 54, 55],
                'Ecc_AHA_1': [11], 
                'Ecc_AHA_10': [14], 
                'Ecc_AHA_13': [75], 
                'Ecc_AHA_15': [65], 
                'Ecc_AHA_2': [68, 72], 
                'Ecc_AHA_5': [95, 12], 
                'Ecc_AHA_8': [32], 
                'Ecc_global': [94, 1, 11], 
                'Ell_3': [149], 
                'Ell_5': [7], 
                'Ell_6': [0], 
                'Err_AHA_1': [111, 44], 
                'Err_AHA_10': [15], 
                'Err_AHA_11': [17], 
                'Err_AHA_5': [107], 
                'Err_AHA_6': [59], 
                'Err_global': [90], 
                'LAEF': [154], 
                'LVEDV': [92, 93, 110, 11, 16, 18, 50], 
                'LVEF': [94, 2, 12, 79], 
                'LVESV': [91, 94, 3, 153, 11, 16, 18, 66, 79], 
                'LVM': [96, 124, 11, 18], 
                'LVSV': [50], 
                'RVEDV': [109, 149, 13], 
                'RVEF': [94, 6, 20, 67, 80], 
                'RVESV': [91, 94, 110, 149, 6, 10, 13, 19], 
                'RVSV': [109, 114], 
                'WT_AHA_1': [123, 48], 
                'WT_AHA_10': [133, 51], 
                'WT_AHA_11': [133, 142, 144, 145, 152, 24, 47], 
                'WT_AHA_12': [129, 133, 134, 144, 147, 26, 47], 
                'WT_AHA_13': [115, 125, 146], 
                'WT_AHA_14': [128, 147], 
                'WT_AHA_15': [133, 4], 
                'WT_AHA_16': [132, 144, 147, 4], 
                'WT_AHA_2': [48], 
                'WT_AHA_3': [127], 
                'WT_AHA_5': [97, 126, 155, 25], 
                'WT_AHA_6': [131, 25], 
                'WT_AHA_7': [99, 129, 130], 
                'WT_AHA_8': [100, 131, 150, 29, 42], 
                'WT_AHA_9': [143, 43], 
                'WT_global': [98, 99, 108, 128, 143, 144, 148, 27, 48]}
    selected_data = []
    for key in snp_dict.keys():
        while len(snp_dict[key]) < 40:
            snp_dict[key] = snp_dict[key] + snp_dict[key]
        snp_dict[key] = snp_dict[key][:40]
        snp_dict[key] = [i*3+j for i in snp_dict[key] for j in range(3)]
        index_tensor = torch.tensor(snp_dict[key])
        selected_data.append(tensor[:, index_tensor]) 
    selected_data = torch.stack(selected_data, dim=1)
    return selected_data

def get_snp(train_csv,val_csv,test_csv):
    col = [
    "rs79534072_A", "rs643420_T", "rs12404144_C", "rs2009594_A", "rs7354918_G", "rs650720_T", "rs3738685_T", "rs12724121_A", "rs934012_A", "rs7255_T", "rs3813243_T", "rs12988307_C", "rs2562845_C", "rs2042995_C", "rs17076_G", "rs142556838_T", "rs1873164_G", "rs55844607_G", "rs10497529_A", "rs55834511_C", "rs9856926_A", "rs744892_A", "rs6809328_C", "rs2686630_C", "rs57078287_G", "rs6549251_T", "rs62253179_A", "rs62253185_A", "rs55914222_C", "rs9850919_C", "rs698099_G", "rs67846163_G", "rs2968210_C", "rs154455_T", "rs10043782_T", "rs55745974_T", "rs10065122_C", "rs2438150_C", "rs72787559_T", "rs335196_A", "rs7702622_T", "rs434775_T", "rs72801474_A", "rs11745702_C", "rs13165478_A", "rs1630736_T", "rs7744333_C", "rs730506_C", "rs4151702_C", "rs4707174_C", "rs7752142_A", "rs9401921_G", "rs2328474_T", "rs13203975_A", "rs58127685_T", "rs2107595_A", "rs336284_A", "rs741408_T", "rs150260620_A", "rs13234515_T", "rs4078435_C", "rs6974735_G", "rs11768878_G", "rs11761337_A", "rs1583081_T", "rs7786419_A", "rs2307036_A", "rs1915986_A", "rs3789849_C", "rs907183_C", "rs4840467_A", "rs7832708_T", "rs6601450_T", "rs12541800_G", "rs11250162_T", "rs7823808_C", "rs7009229_C", "rs34557926_T", "rs13252512_G", "rs34866937_A", "rs11786896_T", "rs10740811_G", "rs10763764_A", "rs2893923_T", "rs1896995_T", "rs11593126_G", "rs2797983_G", "rs1343094_T", "rs12217597_C", "rs7904979_G", "rs10885378_C", "rs12241957_C", "rs7921223_C", "rs117550412_T", "rs17617337_T", "rs72842211_T", "rs621679_A", "rs78777726_C", "rs12285933_T", "rs11604825_T", "rs11039348_A", "rs72931764_A", "rs875107_C", "rs747249_A", "rs861202_G", "rs4148674_C", "rs73139037_T", "rs73145172_T", "rs7299436_G", "rs597808_A", "rs653178_C", "rs3914956_T", "rs7994761_G", "rs376439_G", "rs2284651_C", "rs61991200_G", "rs4905134_A", "rs11844114_T", "rs17352842_T", "rs1561207_T", "rs627634_T", "rs1441358_G", "rs1048661_T", "rs12905223_C", "rs11638445_A", "rs11633377_G", "rs11073716_T", "rs12595786_C", "rs35630683_C", "rs72630465_T", "rs56864281_A", "rs8039472_A", "rs35808647_A", "rs3803405_A", "rs7166287_C", "rs77870048_T", "rs62053262_G", "rs7500448_G", "rs488327_C", "rs511893_G", "rs2126202_C", "rs4791494_G", "rs12453217_T", "rs61572747_G", "rs55938136_G", "rs242562_A", "rs2696532_G", "rs199470_C", "rs1563304_T", "rs17608766_C", "rs617759_T", "rs59945167_T", "rs2070458_A", "rs2267038_G", "rs133885_A", "rs4820654_G", "rs57774511_C"
    ]
    train_tar = train_csv[col]
    val_tar = val_csv[col]
    test_tar = test_csv[col]


    train_tar = train_tar.fillna(train_tar.mode().iloc[0])
    val_tar = val_tar.fillna(val_tar.mode().iloc[0])
    test_tar = test_tar.fillna(test_tar.mode().iloc[0])

    train_tar = train_tar.astype('category')
    val_tar = val_tar.astype('category')
    test_tar = test_tar.astype('category')

    train_tar_ = pd.get_dummies(train_tar, prefix=col)
    val_tar_ = pd.get_dummies(val_tar, prefix=col)
    test_tar_ = pd.get_dummies(test_tar, prefix=col)
    print(test_tar_.info())
    print(val_tar_.info())
    print(test_tar_.info())

    train_tar_ = train_tar_.astype({col: int for col in train_tar_.select_dtypes(include=[bool]).columns})
    val_tar_ = val_tar_.astype({col: int for col in val_tar_.select_dtypes(include=[bool]).columns})
    test_tar_ = test_tar_.astype({col: int for col in test_tar_.select_dtypes(include=[bool]).columns})
    
    return train_tar_,val_tar_,test_tar_


def get_cha(train_csv,val_csv,test_csv):
    col = ["24100-2.0", "24101-2.0", "24102-2.0", "24103-2.0", "24104-2.0", "24105-2.0", "24106-2.0", "24107-2.0", "24108-2.0", "24109-2.0", "24110-2.0", "24111-2.0", "24112-2.0", "24113-2.0", "24114-2.0", "24115-2.0", "24116-2.0", "24117-2.0", "24118-2.0", "24119-2.0", "24120-2.0", "24121-2.0", "24122-2.0", "24123-2.0", "24124-2.0", "24125-2.0", "24126-2.0", "24127-2.0", "24128-2.0", "24129-2.0", "24130-2.0", "24131-2.0", "24132-2.0", "24133-2.0", "24134-2.0", "24135-2.0", "24136-2.0", "24137-2.0", "24138-2.0", "24139-2.0", "24140-2.0", "24141-2.0", "24142-2.0", "24143-2.0", "24144-2.0", "24145-2.0", "24146-2.0", "24147-2.0", "24148-2.0", "24149-2.0", "24150-2.0", "24151-2.0", "24152-2.0", "24153-2.0", "24154-2.0", "24155-2.0", "24156-2.0", "24157-2.0", "24158-2.0", "24159-2.0", "24160-2.0", "24161-2.0", "24162-2.0", "24163-2.0", "24164-2.0", "24165-2.0", "24166-2.0", "24167-2.0", "24168-2.0", "24169-2.0", "24170-2.0", "24171-2.0", "24172-2.0", "24173-2.0", "24174-2.0", "24175-2.0", "24176-2.0", "24177-2.0", "24178-2.0", "24179-2.0", "24180-2.0", "24181-2.0"]
    train_data = train_csv[col].astype('float64')
    val_data = val_csv[col].astype('float64')
    test_data = test_csv[col].astype('float64')
    if train_data.isnull().values.any() or val_data.isnull().values.any() or test_data.isnull().values.any():
        print('cha data has nan')
        exit(0)


    return train_data, val_data, test_data

def get_I(type_of_dis,train_csv,val_csv,test_csv):
    col = [type_of_dis]
    train_tar = train_csv[col]
    train_tar = train_tar.fillna(0)
    train_tar = train_tar.astype(str)
    train_tar = train_tar.where(train_tar == '0', 1)
    train_tar = train_tar.astype(int)

    val_tar = val_csv[col]
    val_tar = val_tar.fillna(0)
    val_tar = val_tar.astype(str)
    val_tar = val_tar.where(val_tar == '0', 1)
    val_tar = val_tar.astype(int)

    test_tar = test_csv[col]
    test_tar = test_tar.fillna(0)
    test_tar = test_tar.astype(str)
    test_tar = test_tar.where(test_tar == '0', 1)
    test_tar = test_tar.astype(int)

    return train_tar,val_tar,test_tar


def get_select_tar(train_csv,val_csv,test_csv):
    col = ['21003-2.0','1558-2.0','21001-2.0','884-2.0','SEX','1160-2.0',
           '20116-2.0','50-2.0','1990-2.0','21002-2.0','1289-2.0',
           '864-2.0','1389-2.0','1200-2.0','22426-2.0']
    train_tar = train_csv[col].astype('float64')
    val_tar = val_csv[col].astype('float64')
    test_tar = test_csv[col].astype('float64')
    category_feats = [
        'SEX',
        '20116-2.0',
        '1990-2.0',
        '1200-2.0']
    col_set = set(col)
    category_feats_set = set(category_feats)
    numeric_feats = col_set - category_feats_set
    numeric_feats = list(numeric_feats)

    for feat in category_feats:
        if train_tar[feat].isnull().any() or val_tar[feat].isnull().any() or test_tar[feat].isnull().any():
            train_tar[feat] = train_tar[feat].astype('category').cat.add_categories('Unknown').fillna('Unknown')
            val_tar[feat] = val_tar[feat].astype('category').cat.add_categories('Unknown').fillna('Unknown')
            test_tar[feat] = test_tar[feat].astype('category').cat.add_categories('Unknown').fillna('Unknown')
        else:
            train_tar[feat] = train_tar[feat].astype("category")
            val_tar[feat] = val_tar[feat].astype("category")
            test_tar[feat] = test_tar[feat].astype("category")
    
    one_hot_data_train = pd.get_dummies(train_tar[category_feats], prefix=category_feats)
    one_hot_data_val = pd.get_dummies(val_tar[category_feats], prefix=category_feats)
    one_hot_data_test = pd.get_dummies(test_tar[category_feats], prefix=category_feats)

    train_tar[numeric_feats] = train_tar[numeric_feats].fillna(train_tar[numeric_feats].mode().iloc[0])
    val_tar[numeric_feats] = val_tar[numeric_feats].fillna(val_tar[numeric_feats].mode().iloc[0])
    test_tar[numeric_feats] = test_tar[numeric_feats].fillna(test_tar[numeric_feats].mode().iloc[0])

    scaler = StandardScaler()
    scaler.fit(train_tar[numeric_feats])
    train_tar[numeric_feats] = scaler.transform(train_tar[numeric_feats])
    val_tar[numeric_feats] = scaler.transform(val_tar[numeric_feats])
    test_tar[numeric_feats] = scaler.transform(test_tar[numeric_feats])

    train_tar_ = pd.concat([one_hot_data_train, train_tar[numeric_feats]], axis=1)
    val_tar_ = pd.concat([one_hot_data_val, val_tar[numeric_feats]], axis=1)
    test_tar_ = pd.concat([one_hot_data_test, test_tar[numeric_feats]], axis=1)
    
    train_tar_ = train_tar_.astype({col: int for col in train_tar_.select_dtypes(include=[bool]).columns})
    val_tar_ = val_tar_.astype({col: int for col in val_tar_.select_dtypes(include=[bool]).columns})
    test_tar_ = test_tar_.astype({col: int for col in test_tar_.select_dtypes(include=[bool]).columns})
    print(f'train_tar_ shape: {train_tar_.shape}, val_tar_ shape: {val_tar_.shape}, test_tar_ shape: {test_tar_.shape}')
    print(train_tar_.head())
    print(val_tar_.head())
    print(test_tar_.head())
    return train_tar_,val_tar_,test_tar_


def get_tar(train_csv,val_csv,test_csv):
    col = ['22420-2.0', '22421-2.0', '22422-2.0', '22423-2.0', '12697-2.0', '21000-2.0', '21003-2.0', 'SEX', '20117-2.0', '1558-2.0', 
           '1618-2.0', '102-2.0', '12681-2.0', '22426-2.0', '1369-2.0', '23099-2.0', '23104-2.0', '21001-2.0', '22427-2.0', '22425-2.0', 
           '12702-2.0', '22424-2.0', '12682-2.0', '12680-2.0', '12678-2.0', '12677-2.0', '1289-2.0', '1239-2.0', '2443-2.0', '4079-2.0',
           '12675-2.0', '1021-2.0', '894-2.0', '874-2.0', '981-2.0', '12683-2.0', '12684-2.0', '20160-2.0', '1269-2.0', '1279-2.0', '2296-2.0', 
           '943-2.0', '971-2.0', '12673-2.0', '12144-2.0', '49-2.0', '23106-2.0', '1379-2.0', '12687-2.0', '12679-2.0', '884-2.0', '904-2.0', 
           '864-2.0', '2178-2.0', '12338-2.0', '1249-2.0', '12676-2.0', '1389-2.0', '22334-2.0', '22330-2.0', '1349-2.0', '21021-2.0', '12340-2.0', 
           '22333-2.0', '1299-2.0', '4717-2.0', '20015-2.0', '1160-2.0', '1200-2.0', '20116-2.0', '1259-2.0', '50-2.0', '12686-2.0', '4080-2.0', 
           '12674-2.0', '1990-2.0', '1090-2.0', '1080-2.0', '1070-2.0', '23283-2.0', '924-2.0', '12336-2.0', '48-2.0', '23098-2.0', '21002-2.0', 
           '2306-2.0', '23101-2.0', '23100-2.0', '23102-2.0', '1980-2.0']
    train_tar = train_csv[col].astype('float64')
    val_tar = val_csv[col].astype('float64')
    test_tar = test_csv[col].astype('float64')


    category_feats = [
        "21000-2.0",
        "SEX",
        "20117-2.0",
        "1558-2.0",
        "1618-2.0",
        "1369-2.0",
        "1239-2.0",
        "2443-2.0",
        "1021-2.0",
        "981-2.0",
        "20160-2.0",
        "2296-2.0",
        "943-2.0",
        "971-2.0",
        "1379-2.0",
        "884-2.0",
        "904-2.0",
        "864-2.0",
        "2178-2.0",
        "1249-2.0",
        "1389-2.0",
        "1349-2.0",
        "4717-2.0",
        "1200-2.0",
        "20116-2.0",
        "1259-2.0",
        "1990-2.0",
        "924-2.0",
        "2306-2.0",
        "1980-2.0"
    ]
    col_set = set(col)
    category_feats_set = set(category_feats)
    numeric_feats = col_set - category_feats_set
    numeric_feats = list(numeric_feats)

    for feat in category_feats:
        train_tar[feat] = train_tar[feat].astype("category")
        val_tar[feat] = val_tar[feat].astype("category")
        test_tar[feat] = test_tar[feat].astype("category")
    
    one_hot_data_train = pd.get_dummies(train_tar[category_feats], prefix=category_feats)
    one_hot_data_val = pd.get_dummies(val_tar[category_feats], prefix=category_feats)
    one_hot_data_test = pd.get_dummies(test_tar[category_feats], prefix=category_feats)

    train_columns = set(one_hot_data_train.columns)
    val_columns = set(one_hot_data_val.columns)
    test_columns = set(one_hot_data_test.columns)

    # 找出只在训练集中出现的列
    train_val_columns = train_columns - val_columns
    train_test_columns = train_columns - test_columns
    
    for col in train_val_columns:
        one_hot_data_val[col] = False
    for col in train_test_columns:
        one_hot_data_test[col] = False

    train_tar[numeric_feats] = train_tar[numeric_feats].fillna(train_tar[numeric_feats].mode().iloc[0])
    val_tar[numeric_feats] = val_tar[numeric_feats].fillna(val_tar[numeric_feats].mode().iloc[0])
    test_tar[numeric_feats] = test_tar[numeric_feats].fillna(test_tar[numeric_feats].mode().iloc[0])

    scaler = StandardScaler()
    scaler.fit(train_tar[numeric_feats])
    train_tar[numeric_feats] = scaler.transform(train_tar[numeric_feats])
    val_tar[numeric_feats] = scaler.transform(val_tar[numeric_feats])
    test_tar[numeric_feats] = scaler.transform(test_tar[numeric_feats])

    train_tar_ = pd.concat([one_hot_data_train, train_tar[numeric_feats]], axis=1)
    val_tar_ = pd.concat([one_hot_data_val, val_tar[numeric_feats]], axis=1)
    test_tar_ = pd.concat([one_hot_data_test, test_tar[numeric_feats]], axis=1)
    
    train_tar_ = train_tar_.astype({col: int for col in train_tar_.select_dtypes(include=[bool]).columns})
    val_tar_ = val_tar_.astype({col: int for col in val_tar_.select_dtypes(include=[bool]).columns})
    test_tar_ = test_tar_.astype({col: int for col in test_tar_.select_dtypes(include=[bool]).columns})

    return train_tar_,val_tar_,test_tar_


def get_ecg(ecg_path):
    ecg_file = open(ecg_path).read()
    bs = None
    try:
        bs = BeautifulSoup(ecg_file, features="lxml")
    except Exception as e:
        print(f"Error when parsing {ecg_path}: {e}")
        return None
    ecg_waveform_length = 5000
    if ecg_waveform_length == 600:
        waveform = bs.body.cardiologyxml.mediansamples
    else:
        waveform = bs.body.cardiologyxml.stripdata
    # print(waveform)
    # print(type(waveform))
    data_numpy = None
    bs_measurement = bs.body.cardiologyxml.restingecgmeasurements
    heartbeat = int(bs_measurement.find_all("VentricularRate".lower())[0].string)

    for each_wave in waveform.find_all("waveformdata"):
        each_data = each_wave.string.strip().split(",")
        each_data = [s.replace('\n\t\t', '') for s in each_data]
        each_data = np.array(each_data, dtype=np.float32)
        # plt.plot(each_data)
        seasonal_decompose_result = seasonal_decompose(each_data, model="additive",
                                                        period=int(ecg_waveform_length*6/heartbeat))
        trend = seasonal_decompose_result.trend
        start, end = 0, ecg_waveform_length - 1
        sflag, eflag = False, False
        for i in range(ecg_waveform_length):
            if np.isnan(trend[i]):
                start += 1
            else:
                sflag = True
            if np.isnan(trend[ecg_waveform_length-1-i]):
                end -= 1
            else:
                eflag = True
            if sflag and eflag:
                break
        trend[:start] = trend[start]
        trend[end:] = trend[end]
        # trend[np.isnan(trend)] = 0.0
        result = np.array(seasonal_decompose_result.observed - trend)
        # plt.plot(result)
        # plt.show()
        # exit()
        if data_numpy is None:
            data_numpy = result
        else:
            data_numpy = np.vstack((data_numpy, result))

    return data_numpy
def norm(array):
    min_val = np.min(array)
    max_val = np.max(array)
    # 进行最小-最大归一化
    normalized_array = (array - min_val) / (max_val - min_val)
    return normalized_array

def get_img(cmr_path,is_continuous=True):
    i = cmr_path
    cmr_es_path = os.path.join('/mnt/data/ukb_heartmri/ukb_20209', i.split('/')[0],'seg_sa_ES.nii.gz')
    cmr_ed_path = os.path.join('/mnt/data/ukb_heartmri/ukb_20209', i.split('/')[0],'seg_sa_ED.nii.gz')
    img_es_path = os.path.join('/mnt/data/ukb_heartmri/ukb_20209', i.split('/')[0],'sa_ES.nii.gz')
    img_ed_path = os.path.join('/mnt/data/ukb_heartmri/ukb_20209', i.split('/')[0],'sa_ED.nii.gz')
    sa_path = os.path.join('/mnt/data/ukb_heartmri/ukb_20209', i.split('/')[0],'sa.nii.gz')
    sa_mask_path = os.path.join('/mnt/data/ukb_heartmri/ukb_20209', i.split('/')[0],'seg_sa.nii.gz')

    img_ed = nib.load(cmr_ed_path).get_fdata()
    img_es = nib.load(cmr_es_path).get_fdata()
    sa = nib.load(sa_path).get_fdata()
    slice_index = find_slice_with_most_ones(img_es)
    # print(f'i: {i}, slice index: {slice_index}')

    cmr_ed = nib.load(img_ed_path).get_fdata()
    cmr_es = nib.load(img_es_path).get_fdata()
    
    if slice_index <=2 and img_ed.shape[2]>=5:
        slice_index += 1
    
    if is_continuous:
        sa_mask = nib.load(sa_mask_path).get_fdata()
        data_sa,mask_sa = crop_image_and_mask_3dim(image=sa[:,:,slice_index,:],mask=sa_mask[:,:,slice_index,:],structure_size=3,square_size=80)
        data_sa = np.transpose(norm(data_sa),(2,0,1))
        # print(data_sa.shape)
        # nib.save(nib.Nifti1Image(data_sa, np.eye(4)), os.path.join('/home/dingzhengyao/Work/ECG_CMR/ECG_CMR_TAR/Project_version1/test_code/test_generate', i.split('/')[0]+'sa.nii.gz'))
        return data_sa

    else:
        data_ed,mask_ed = crop_image_and_mask(image=cmr_ed[:,:,slice_index],mask=img_ed[:,:,slice_index],structure_size=2,square_size=80)
        data_es,mask_es = crop_image_and_mask(image=cmr_es[:,:,slice_index],mask=img_es[:,:,slice_index],structure_size=2,square_size=80)
        data_ed = np.expand_dims(norm(data_ed),axis=-1)
        data_es = np.expand_dims(norm(data_es),axis=-1)
        mask_ed = np.eye(4)[mask_ed.astype(int)]
        mask_es = np.eye(4)[mask_es.astype(int)]
        integrated_ed = np.transpose(np.concatenate((data_ed, mask_ed), axis=-1),(2,0,1))
        integrated_es = np.transpose(np.concatenate((data_es, mask_es), axis=-1),(2,0,1))
        preprocess_cmr = np.concatenate((integrated_ed, integrated_es), axis=0)

    return preprocess_cmr

def get_img2(eid):
    i = eid
    cmr_es_path = os.path.join('/mnt/data/ukb_heartmri/ukb_20209', str(i)+'_20209_2_0','seg_sa_ES.nii.gz')
    cmr_ed_path = os.path.join('/mnt/data/ukb_heartmri/ukb_20209', str(i)+'_20209_2_0','seg_sa_ED.nii.gz')
    img_es_path = os.path.join('/mnt/data/ukb_heartmri/ukb_20209', str(i)+'_20209_2_0','sa_ES.nii.gz')
    img_ed_path = os.path.join('/mnt/data/ukb_heartmri/ukb_20209', str(i)+'_20209_2_0','sa_ED.nii.gz')
    sa_path = os.path.join('/mnt/data/ukb_heartmri/ukb_20209', str(i)+'_20209_2_0','sa.nii.gz')
    sa_mask_path = os.path.join('/mnt/data/ukb_heartmri/ukb_20209', str(i)+'_20209_2_0','seg_sa.nii.gz')

    img_ed = nib.load(cmr_ed_path).get_fdata()
    img_es = nib.load(cmr_es_path).get_fdata()
    sa = nib.load(sa_path).get_fdata()
    slice_index = find_slice_with_most_ones(img_es)
    # print(f'i: {i}, slice index: {slice_index}')

    cmr_ed = nib.load(img_ed_path).get_fdata()
    cmr_es = nib.load(img_es_path).get_fdata()
    
    if slice_index <=2 and img_ed.shape[2]>=5:
        slice_index += 1
    
    
    sa_mask = nib.load(sa_mask_path).get_fdata()
    data_sa,mask_sa = crop_image_and_mask_3dim(image=sa[:,:,slice_index,:],mask=sa_mask[:,:,slice_index,:],structure_size=3,square_size=80)
    data_sa = np.transpose(norm(data_sa),(2,0,1))
    mask_sa = np.transpose(mask_sa,(2,0,1))
    print(data_sa.shape)
    print(mask_sa.shape)
    # nib.save(nib.Nifti1Image(data_sa, np.eye(4)), os.path.join('/home/dingzhengyao/Work/ECG_CMR/ECG_CMR_TAR/Project_version1/test_code/test_generate', i.split('/')[0]+'sa.nii.gz'))
    return data_sa,mask_sa



def crop_center(image_array,square_size=80,dim=2):
    # 确定图像尺寸
    height, width = image_array.shape[:2]

    # 计算中心点
    center_x, center_y = width // 2, height // 2

    # 计算裁剪区域
    start_x = int(center_x - square_size/2)
    end_x = int(center_x + square_size/2)
    start_y = int(center_y - square_size/2)
    end_y = int(center_y + square_size/2)

    # 裁剪图像
    if dim == 2:
        cropped_image = image_array[start_y:end_y, start_x:end_x]
    if dim == 3:
        cropped_image = image_array[start_y:end_y, start_x:end_x,:]
    else:
        raise ValueError("input dim is not 2 or 3")
    return cropped_image


from scipy.ndimage import binary_opening
from scipy.ndimage.morphology import generate_binary_structure
def find_min_enclosing_box_and_size(mask):

    structure = generate_binary_structure(2, 2)

    # 应用形态学开运算
    cleaned_mask = binary_opening(mask, structure=structure)
    if np.all(cleaned_mask == 0):
        
        return None, None,None,80,80
    # 确定非零元素的行和列
    rows = np.any(cleaned_mask != 0, axis=1)
    cols = np.any(cleaned_mask != 0, axis=0)

    # 找到非零元素的最小和最大行索引
    rows_where = np.where(rows)
    row_min = np.min(rows_where)
    row_max = np.max(rows_where)

    # 找到非零元素的最小和最大列索引
    cols_where = np.where(cols)
    col_min = np.min(cols_where)
    col_max = np.max(cols_where)

    # 计算最小外接矩形的长和宽
    width = col_max - col_min + 1
    height = row_max - row_min + 1

    # 返回最小外接矩形的坐标及其尺寸
    return cleaned_mask,(row_min, col_min), (row_max, col_max), width, height

import numpy as np
from scipy.ndimage import binary_opening
from scipy.ndimage.morphology import generate_binary_structure

def crop_to_square_and_process_mask(image, mask, structure_size=2, square_size=80):
    # 创建一个二维的结构元素
    structure = generate_binary_structure(2, structure_size)

    # 应用形态学开运算
    cleaned_mask = binary_opening(mask, structure=structure)

    # 确定非零元素的行和列
    rows = np.any(cleaned_mask != 0, axis=1)
    cols = np.any(cleaned_mask != 0, axis=0)

    # 找到非零元素的最小和最大行索引
    rows_where = np.where(rows)
    row_min = np.min(rows_where)
    row_max = np.max(rows_where)

    # 找到非零元素的最小和最大列索引
    cols_where = np.where(cols)
    col_min = np.min(cols_where)
    col_max = np.max(cols_where)

    # 计算最小外接矩形的中心点
    center_row = (row_min + row_max) // 2
    center_col = (col_min + col_max) // 2

    # 计算 80x80 正方形的边界
    half_size = square_size // 2
    square_top = center_row - half_size
    square_left = center_col - half_size
    square_bottom = center_row + half_size
    square_right = center_col + half_size

    # 创建一个空白的80x80图像
    cropped_image = np.zeros((square_size, square_size, image.shape[2]), dtype=image.dtype)

    # 计算裁剪的源坐标和目标坐标
    src_top = max(square_top, 0)
    src_left = max(square_left, 0)
    src_bottom = min(square_bottom, image.shape[0])
    src_right = min(square_right, image.shape[1])

    dst_top = src_top - square_top
    dst_left = src_left - square_left
    dst_bottom = dst_top + (src_bottom - src_top)
    dst_right = dst_left + (src_right - src_left)

    # 将图像的相关部分复制到新的80x80图像中
    cropped_image[dst_top:dst_bottom, dst_left:dst_right] = image[src_top:src_bottom, src_left:src_right]

    return cropped_image

# 示例用法
# image = np.array([...])  #

def find_slice_with_most_ones(array):
    # 检查输入数组的维度
    if array.ndim != 3:
        raise ValueError("输入数组不是三维的。")

    # 在 z 轴方向计算每个切片中 1 的数量
    count_ones_per_slice = np.sum(array == 3, axis=(0, 1))

    # 找到含 1 最多的切片索引
    slice_index = np.argmax(count_ones_per_slice)

    # 返回含 1 最多的切片
    return slice_index

def crop_image_and_mask_3dim(image, mask, structure_size=3, square_size=80):
    structure = generate_binary_structure(3, structure_size)
    # 应用形态学开运算
    cleaned_mask = binary_opening(mask, structure=structure)
    if np.all(cleaned_mask == 0):
        cropped_image = crop_center(image, square_size,dim=3)
        cropped_mask = np.zeros((square_size, square_size, image.shape[2]), dtype=mask.dtype)
        return cropped_image, cropped_mask

    rows_most_min = 300
    rows_most_max = 0
    cols_most_min = 300
    cols_most_max = 0

    for i in range(cleaned_mask.shape[2]):
        rows = np.any(cleaned_mask[:, :, i] != 0, axis=1)
        cols = np.any(cleaned_mask[:, :, i] != 0, axis=0)
        rows_where = np.where(rows)
        cols_where = np.where(cols)
        if len(rows_where[0]) != 0:
            rows_most_min = min(rows_most_min, np.min(rows_where))
            rows_most_max = max(rows_most_max, np.max(rows_where))
        if len(cols_where[0]) != 0:
            cols_most_min = min(cols_most_min, np.min(cols_where))
            cols_most_max = max(cols_most_max, np.max(cols_where))
    if (rows_most_max - rows_most_min > 80) or (cols_most_max - cols_most_min > 80):
        print('error exceed 80')
    center_row = (rows_most_min + rows_most_max) // 2
    center_col = (cols_most_min + cols_most_max) // 2

    # 计算 80x80 正方形的边界
    half_size = square_size // 2
    square_top = center_row - half_size
    square_left = center_col - half_size
    square_bottom = center_row + half_size
    square_right = center_col + half_size
    
    # 创建一个空白的80x80图像和掩码
    cropped_image = np.zeros((square_size, square_size, image.shape[2]), dtype=image.dtype)
    cropped_mask = np.zeros((square_size, square_size, image.shape[2]), dtype=mask.dtype)
    # 计算裁剪的源坐标和目标坐标
    src_top = max(square_top, 0)
    src_left = max(square_left, 0)
    src_bottom = min(square_bottom, image.shape[0])
    src_right = min(square_right, image.shape[1])

    dst_top = src_top - square_top
    dst_left = src_left - square_left
    dst_bottom = dst_top + (src_bottom - src_top)
    dst_right = dst_left + (src_right - src_left)

    # 将图像和掩码的相关部分复制到新的80x80图像中
    cropped_image[dst_top:dst_bottom, dst_left:dst_right, :] = image[src_top:src_bottom, src_left:src_right, :]
    cropped_mask[dst_top:dst_bottom, dst_left:dst_right, :] = mask[src_top:src_bottom, src_left:src_right, :]
    return cropped_image, cropped_mask


def crop_image_and_mask(image, mask, structure_size=2, square_size=80):
    # 创建一个二维的结构元素
    structure = generate_binary_structure(2, structure_size)

    # 应用形态学开运算
    cleaned_mask = binary_opening(mask, structure=structure)

    if np.all(cleaned_mask == 0):
        print('all zero')
        cropped_image = crop_center(image, square_size,dim=2)
        cropped_mask = np.zeros((square_size, square_size), dtype=mask.dtype)
        return cropped_image, cropped_mask
    
    # 确定非零元素的行和列
    rows = np.any(cleaned_mask != 0, axis=1)
    cols = np.any(cleaned_mask != 0, axis=0)

    # 找到非零元素的最小和最大行索引
    rows_where = np.where(rows)
    row_min = np.min(rows_where)
    row_max = np.max(rows_where)

    # 找到非零元素的最小和最大列索引
    cols_where = np.where(cols)
    col_min = np.min(cols_where)
    col_max = np.max(cols_where)

    if (row_max - row_min > 80) or (col_max - col_min > 80):
        print('error')
    # 计算最小外接矩形的中心点
    center_row = (row_min + row_max) // 2
    center_col = (col_min + col_max) // 2

    # 计算 80x80 正方形的边界
    half_size = square_size // 2
    square_top = center_row - half_size
    square_left = center_col - half_size
    square_bottom = center_row + half_size
    square_right = center_col + half_size

    # 创建一个空白的80x80图像和掩码
    cropped_image = np.zeros((square_size, square_size), dtype=image.dtype)
    cropped_mask = np.zeros((square_size, square_size), dtype=mask.dtype)
    # 计算裁剪的源坐标和目标坐标
    src_top = max(square_top, 0)
    src_left = max(square_left, 0)
    src_bottom = min(square_bottom, image.shape[0])
    src_right = min(square_right, image.shape[1])

    dst_top = src_top - square_top
    dst_left = src_left - square_left
    dst_bottom = dst_top + (src_bottom - src_top)
    dst_right = dst_left + (src_right - src_left)

    # 将图像和掩码的相关部分复制到新的80x80图像中
    cropped_image[dst_top:dst_bottom, dst_left:dst_right] = image[src_top:src_bottom, src_left:src_right]
    cropped_mask[dst_top:dst_bottom, dst_left:dst_right] = mask[src_top:src_bottom, src_left:src_right]

    return cropped_image, cropped_mask




import pandas as pd
import nibabel as nib
import os
def test_wh(path):
    with open('output_val_1.txt', 'w') as f:
        csv = pd.read_csv(path)
        cmr = csv['20209_2_0']
        not_exist = []
        xiaodian = []
        value_error = []
        for i in cmr:
            cmr_es_path = os.path.join('/mnt/data/ukb_heartmri/ukb_20209', i.split('/')[0],'seg_sa_ES.nii.gz')
            cmr_ed_path = os.path.join('/mnt/data/ukb_heartmri/ukb_20209', i.split('/')[0],'seg_sa_ED.nii.gz')
            try:
                img_ed = nib.load(cmr_ed_path)
                img_es = nib.load(cmr_es_path)
                img_ed = img_ed.get_fdata()
                img_es = img_es.get_fdata()
                slice_index = find_slice_with_most_ones(img_es)
                
                print(f'i: {i}, slice index: {slice_index}',file=f)
                f.flush()
                if slice_index <=2 and img_ed.shape[2]>=5:
                    slice_index += 1
                clean_mask_ed ,_,_,width_ed,height_ed = find_min_enclosing_box_and_size(img_ed[:,:,slice_index])
                clean_mask_sd ,_,_,width_es,height_es = find_min_enclosing_box_and_size(img_es[:,:,slice_index])
                if width_ed > 80 or height_ed > 80 or width_es > 80 or height_es > 80:
                    # print(f'i: {i}')
                    xiaodian.append(i)
                    print(f'ed: {width_ed}, {height_ed}',file=f)
                    print(f'es: {width_es}, {height_es}',file=f)
                    f.flush()
            except FileNotFoundError as e:
                print(e, file=f)
                print(f'i: {i}',file=f)
                f.flush()
                not_exist.append(i)
                    
                continue
            except ValueError as e:
                print(e, file=f)
                print(f'i: {i}', file=f)
                f.flush()
                value_error.append(i)
                
    return not_exist, xiaodian, value_error
# not_exist,xiaodian,value_error = test_wh('/mnt/data/ukb_collation/ukb_ecg_cmr/data/val_v2.csv')
# with open('output_val_1.txt', 'a') as f:
#     print(not_exist,file=f)
#     print(xiaodian,file=f)
#     print(value_error,file=f)
#     f.flush()



# col = ['22420-2.0', '22421-2.0', '22422-2.0', '22423-2.0', '12697-2.0', '21000-2.0', '21003-2.0', 'SEX', '20117-2.0', '1558-2.0', '1618-2.0', '102-2.0', '12681-2.0', '22426-2.0', '1369-2.0', '23099-2.0', '23104-2.0', '21001-2.0', '22427-2.0', '22425-2.0', '12702-2.0', '22424-2.0', '12682-2.0', '12680-2.0', '12678-2.0', '12677-2.0', '1289-2.0', '1239-2.0', '2443-2.0', '4079-2.0', '12675-2.0', '1021-2.0', '894-2.0', '874-2.0', '981-2.0', '12683-2.0', '12684-2.0', '20160-2.0', '1269-2.0', '1279-2.0', '2296-2.0', '943-2.0', '971-2.0', '12673-2.0', '12144-2.0', '49-2.0', '23106-2.0', '1379-2.0', '12687-2.0', '12679-2.0', '884-2.0', '904-2.0', '864-2.0', '2178-2.0', '12338-2.0', '1249-2.0', '12676-2.0', '1389-2.0', '22334-2.0', '22330-2.0', '1349-2.0', '21021-2.0', '12340-2.0', '22333-2.0', '1299-2.0', '4717-2.0', '20015-2.0', '1160-2.0', '1200-2.0', '20116-2.0', '1259-2.0', '50-2.0', '12686-2.0', '4080-2.0', '12674-2.0', '1990-2.0', '1090-2.0', '1080-2.0', '1070-2.0', '23283-2.0', '924-2.0', '12336-2.0', '48-2.0', '23098-2.0', '21002-2.0', '2306-2.0', '23101-2.0', '23100-2.0', '23102-2.0', '1980-2.0']
#         self.train_tar = self.train_csv[col].astype('float64')
#         self.val_tar = self.val_csv[col].astype('float64')
#         self.test_tar = self.test_csv[col].astype('float64')

        
#         category_feats = [
#             "21000-2.0",
#             "SEX",
#             "20117-2.0",
#             "1558-2.0",
#             "1618-2.0",
#             "1369-2.0",
#             "1239-2.0",
#             "2443-2.0",
#             "1021-2.0",
#             "981-2.0",
#             "20160-2.0",
#             "2296-2.0",
#             "943-2.0",
#             "971-2.0",
#             "1379-2.0",
#             "884-2.0",
#             "904-2.0",
#             "864-2.0",
#             "2178-2.0",
#             "1249-2.0",
#             "1389-2.0",
#             "1349-2.0",
#             "4717-2.0",
#             "1200-2.0",
#             "20116-2.0",
#             "1259-2.0",
#             "1990-2.0",
#             "924-2.0",
#             "2306-2.0",
#             "1980-2.0"
#         ]
#         col_set = set(col)
#         category_feats_set = set(category_feats)
#         numeric_feats = col_set - category_feats_set
#         numeric_feats = list(numeric_feats)

#         for feat in category_feats:
#             self.train_tar[feat] = self.train_tar[feat].astype("category")
#             self.val_tar[feat] = self.val_tar[feat].astype("category")
#             self.test_tar[feat] = self.test_tar[feat].astype("category")
        
#         one_hot_data_train = pd.get_dummies(self.train_tar[category_feats], prefix=category_feats)
#         one_hot_data_val = pd.get_dummies(self.val_tar[category_feats], prefix=category_feats)
#         one_hot_data_test = pd.get_dummies(self.test_tar[category_feats], prefix=category_feats)

#         train_columns = set(one_hot_data_train.columns)
#         val_columns = set(one_hot_data_val.columns)
#         test_columns = set(one_hot_data_test.columns)

#         # 找出只在训练集中出现的列
#         train_val_columns = train_columns - val_columns
#         train_test_columns = train_columns - test_columns
        
#         for col in train_val_columns:
#             one_hot_data_val[col] = False
#         for col in train_test_columns:
#             one_hot_data_test[col] = False

#         self.train_tar[numeric_feats] = self.train_tar[numeric_feats].fillna(self.train_tar[numeric_feats].mode().iloc[0])
#         self.val_tar[numeric_feats] = self.val_tar[numeric_feats].fillna(self.val_tar[numeric_feats].mode().iloc[0])
#         self.test_tar[numeric_feats] = self.test_tar[numeric_feats].fillna(self.test_tar[numeric_feats].mode().iloc[0])

#         self.train_tar_ = pd.concat([one_hot_data_train, self.train_tar[numeric_feats]], axis=1)
#         self.val_tar_ = pd.concat([one_hot_data_val, self.val_tar[numeric_feats]], axis=1)
#         self.test_tar_ = pd.concat([one_hot_data_test, self.test_tar[numeric_feats]], axis=1)
        
#         self.train_tar_ = self.train_tar_.astype({col: int for col in self.train_tar_.select_dtypes(include=[bool]).columns})
#         self.val_tar_ = self.val_tar_.astype({col: int for col in self.val_tar_.select_dtypes(include=[bool]).columns})
#         self.test_tar_ = self.test_tar_.astype({col: int for col in self.test_tar_.select_dtypes(include=[bool]).columns})

