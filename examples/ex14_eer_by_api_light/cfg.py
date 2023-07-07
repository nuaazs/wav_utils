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

DATA_FOLDER = "/home/zhaosheng/VAF_UTILS/utils/datasets/cti_test_dataset_16k_envad" 

# 干扰集
NOISE_PATH = "../../dataset/noise"


# NAME = f"{MODEL_NAME}-{SR}k-{DATA_NAME}"
MODEL_NAME = "ERES2NET,CAMPP,ECAPATDNN" # ,ECAPATDNN,CAMPP
DATA_NAME = DATA_FOLDER.split("/")[-1]

if ADD_NOISE:
    NAME = f"{MODEL_NAME}-{SR}k-{DATA_NAME}-noise"
else:
    NAME = f"{MODEL_NAME}-{SR}k-{DATA_NAME}-clean_zero"
