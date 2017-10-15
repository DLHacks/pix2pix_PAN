python train.py --dataroot ./datasets/tmp_facades --name tmp_pan --model pan --which_model_netG unet_256 --which_direction BtoA --lambda_A 100 --dataset_mode aligned --no_lsgan --norm batch --pool_size 0 --niter 10 --niter_decay 10 --save_latest_freq 200 --loadSize 256 --batchSize 10 --gpu_ids -1

python test.py --dataroot ./datasets/tmp_facades --name tmp_pan --model pan --which_model_netG unet_256 --which_direction BtoA --dataset_mode aligned --norm batch --loadSize 256 --gpu_ids -1
