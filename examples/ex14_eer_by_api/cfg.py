# coding = utf-8
# @Time    : 2023-03-08  18:48:10
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: Cofig file.

VAD_TYPE = "vad_8k_en_phone_crdnns"
DEVICE = "gpu"

SR = 16000
LOAD_NPY = False
WORKERS = 20
ADD_NOISE = False

# CTI 未筛选过的数据: cti_mini_test_16k cti_mini_test_8k
# VoxCeleb数据:"voxcele12_test_data_16k"
# "voxcele12_test_data_16k" # voxceleb 16k
# "voxcele12_test_data_8k"
# "voxcele12_test_data_8k_16k" # voxceleb先降采样到8k，再上采样回16k
DATA_FOLDER = "/home/yuanqilong/dataset/newtest" 


# VoxCeleb ECAPA官方提供的模型:"../../models/ECAPATDNN_16k"
# VoxCeleb 降采样到8k再上采样回16k的模型:"../../models/ECAPATDNN_16k_2"
# VoxCeleb 降采样到8k再上采样回16k的模型 valid ErrorRate: 1.43e-02 :"../../models/ECAPATDNN_16k_3"
# VoxCeleb 8k电话信道编解码模型:"../../models/ECAPATDNN_8k"
MODEL_PATH = "../../models/ECAPATDNN-16k-phone_1"
# /home/zhaosheng/utils/models/ECAPATDNN-16k-phone_1 编解码降采样到8k再上采样回16k的模型 valid ErrorRate: 3.19%


# 干扰集
NOISE_PATH = "../../dataset/noise"


# NAME = f"{MODEL_NAME}-{SR}k-{DATA_NAME}"
MODEL_NAME = MODEL_PATH.split("/")[-1]
DATA_NAME = DATA_FOLDER.split("/")[-1]

if ADD_NOISE:
    NAME = f"{MODEL_NAME}-{SR}k-{DATA_NAME}-noise"
else:
    NAME = f"{MODEL_NAME}-{SR}k-{DATA_NAME}-clean_zero"
