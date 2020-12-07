~/anaconda3/bin/python train.py --arch rgbd_resnet --outdir ./results
python predict.py --ckpt=./results/model_best_state.pth
