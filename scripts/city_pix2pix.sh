CUDA_VISIBLE_DEVICES=$1 python train.py --dataroot ./datasets/cityscapes --name pix2pix_city --model pix2pix --which_model_netG unet_256 --which_direction AtoB --lambda_A 100 --dataset_mode aligned --no_lsgan --norm batch --pool_size 0 --niter 200 --niter_decay 200 --loadSize 256 --batchSize 4

CUDA_VISIBLE_DEVICES=$1 python test.py --dataroot ./datasets/cityscapes --name pix2pix_city --model pix2pix --which_model_netG unet_256 --which_direction AtoB --dataset_mode aligned --norm batch --loadSize 256
