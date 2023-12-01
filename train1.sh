# Clip Baseline
#python train.py --config_file configs/person/vit_clipreid_msmt.yml MODEL.DIST_TRAIN False
python train.py --config_file configs/person/vit_clipreid_msmt_dino_teacher_01.yml MODEL.DIST_TRAIN False
python train.py --config_file configs/person/vit_clipreid_msmt_dino_teacher_02.yml MODEL.DIST_TRAIN False
python train.py --config_file configs/person/vit_clipreid_msmt_dino_teacher_03.yml MODEL.DIST_TRAIN False

