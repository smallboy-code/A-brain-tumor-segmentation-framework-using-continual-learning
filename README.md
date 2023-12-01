
# A-brain-tumor-segmentation-framework-using-continual-learning
=======

# Requirements

You need to install the following libraries:
- Python (3.6)
- Pytorch (1.8.1+cu102)
- torchvision (0.9.1+cu102)
- tensorboardX (1.8)
- matplotlib (3.3.1)
- numpy (1.17.2)
- [inplace-abn](https://github.com/mapillary/inplace_abn) (1.0.7)

# How to perform training
The most important file is run.py, that is in charge to start the training or test procedure.
To run it, simpy use the following command:

> CUDA_VISIBLE_DEVICES=${GPU} python run.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 0 --lr 0.005 --epochs ${INITIAL_EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS} .. other args ..
