# coding = utf-8
# @Time    : 2023-03-08  18:48:10
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: Cofig file.

VAD_TYPE = "vad_8k_en_phone_crdnns"
DEVICE = "gpu" #"cuda:0"
NAME = "ecapa_speechbrain"
SR = 8000

RAW_DATA_FOLDER = '/ssd2/cti_test_dataset'
DATA_FOLDER_8k = RAW_DATA_FOLDER+'_8k'
DATA_FOLDER_8k_vad = RAW_DATA_FOLDER+'_8k_vad'
DATA_FOLDER_16k_vad = RAW_DATA_FOLDER+'_16k_vad'

LOAD_NPY = False
WORKERS = 20

MODEL_PATH = "../../models/ECAPATDNN"

if "ecapa" in NAME:
    config_file = "../../models/ECAPATDNN/ecapa-tdnn/config.yaml"
    weight = "/ssd2/online/speaker-verification/SI/tdnn_amsoftmax_epoch51_eer0.011.pdparams"
elif "resnet" in NAME:
    config_file = "/ssd2/online/speaker-verification/egs/resnet34/config.yaml"
    weight = "/ssd2/online/speaker-verification/SI/resnetse34_epoch92_eer0.00931.pdparams"
else:
    print("Please check your config file.")