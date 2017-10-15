CUDA_VISIBLE_DEVICES=$1 python mytrain.py --dataroot ./datasets/maps --name pan_map_10_15 --model pan --which_model_netG unet_256 --which_direction AtoB --lambda_A 100 --dataset_mode aligned --no_lsgan --norm batch --pool_size 0 --niter 200 --niter_decay 200 --loadSize 256 --batchSize 4 --pan_mergin_m 60

CUDA_VISIBLE_DEVICES=$1 python mytest.py --dataroot ./datasets/maps --name pan_map_10_15 --model pan --which_model_netG unet_256 --which_direction AtoB --dataset_mode aligned --norm batch --loadSize 256
