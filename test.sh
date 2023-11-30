# 230918
# 3rd accred ## 5. ReID 정확도

# data ready ./utils/keti2msmt.ipynb
# setting keti data directory tree msmt style

# 5-1. model run
#python test_clipreid.py --config_file configs/person/vit_clipreid_KETI.yml MODEL.DIST_TRAIN False MODEL.DEVICE_ID "('0')"

# 5-2. image sorting run
# notebook ./utils/error_get.ipynb
# notebook ./utils/positive_get.ipynb

# out dir
# syh '/media/syh/hdd/Clip_ReID_KETI/positive_case_result_KETI'
# syh '/media/syh/hdd/Clip_ReID_KETI/error_case_result_KETI'
#



# 231004 Test trained model 230927
#
#python test_clipreid.py --config_file configs/person/vit_clipreid_msmt.yml MODEL.DIST_TRAIN False MODEL.DEVICE_ID "('0')"

# 231107 Test for GradCAM ClipReID
#python test_clipreid.py --config_file configs/person/vit_clipreid_msmt.yml MODEL.DIST_TRAIN False MODEL.DEVICE_ID "('0')"

# 231117 test with msmt, imkaagenet pretrained model

#python test.py --config_file configs/person/vit_clipreid_promptbase_msmt_server_run_dino_imagenet.yml MODEL.DIST_TRAIN False
python test_promptreid.py --config_file configs/person/vit_clipreid_promptbase_msmt_server_run_dino_msmt.yml MODEL.DIST_TRAIN False