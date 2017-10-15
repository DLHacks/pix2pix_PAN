CUDA_VISIBLE_DEVICES=$1 python mytrain.py --dataroot ./datasets/maps --name pix2pix_map_10_14 --model pix2pix --which_model_netG unet_256 --which_direction AtoB --lambda_A 100 --dataset_mode aligned --no_lsgan --norm batch --pool_size 0 --niter 200 --niter_decay 200 --loadSize 256 --batchSize 4

CUDA_VISIBLE_DEVICES=$1 python mytest.py --dataroot ./datasets/maps --name pix2pix_map_10_14 --model pix2pix --which_model_netG unet_256 --which_direction AtoB --dataset_mode aligned --norm batch --loadSize 256
