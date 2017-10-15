CUDA_VISIBLE_DEVICES=$1 python train.py --dataroot ./datasets/facades --name pix2pix_facade --model pix2pix --which_model_netG unet_256 --which_direction BtoA --lambda_A 100 --dataset_mode aligned --no_lsgan --norm batch --pool_size 0 --niter 200 --niter_decay 200 --loadSize 256 --batchSize 4 --pan_mergin_m 20

CUDA_VISIBLE_DEVICES=$1 python test.py --dataroot ./datasets/facades --name pix2pix_facade --model pix2pix --which_model_netG unet_256 --which_direction BtoA --dataset_mode aligned --norm batch --loadSize 256
