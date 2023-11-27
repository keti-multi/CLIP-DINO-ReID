# 23.11.17
#python train_promptreid.py --config_file configs/person/vit_clipreid_promptbase_msmt_server_run_dino_msmt.yml MODEL.DIST_TRAIN False
#python train_promptreid.py --config_file configs/person/vit_clipreid_promptbase_msmt_server_run_dino_imagenet.yml MODEL.DIST_TRAIN False

# 23.11.17
# Dino msmt 80epoch test
#python train_promptreid.py --config_file configs/person/vit_clipreid_promptbase_msmt_server_run_dino_msmt.yml MODEL.DIST_TRAIN False

# Dino msmt 20epoch test dino visual
#python train_promptreid.py --config_file configs/person/vit_clipreid_promptbase_msmt_server_run_dino_msmt.yml MODEL.DIST_TRAIN False

# dino 140 ensemble with clip_reid
python train_promptreid.py --config_file configs/person/vit_clipreid_promptbase_msmt_server_run_dino_msmt_ensemble.yml MODEL.DIST_TRAIN False