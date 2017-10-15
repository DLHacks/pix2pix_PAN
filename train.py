import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader

opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

# read pix2pix/PAN moodel
if opt.model == 'pix2pix':
    assert(opt.dataset_mode == 'aligned')
    from models.pix2pix_model import Pix2PixModel
    model = Pix2PixModel()
    model.initialize(opt)
elif opt.model == 'pan':
    from models.pan_model import PanModel
    model = PanModel()
    model.initialize(opt)

total_steps = 0

batch_size = opt.batchSize
print_freq = opt.print_freq
epoch_count = opt.epoch_count
niter = opt.niter
niter_decay = opt.niter_decay
display_freq = opt.display_freq
save_latest_freq = opt.save_latest_freq
save_epoch_freq = opt.save_epoch_freq

for epoch in range(epoch_count, niter + niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0

    for i, data in enumerate(dataset):
        # data --> (1, 3, 256, 256)
        iter_start_time = time.time()
        total_steps += batch_size
        epoch_iter += batch_size
        model.set_input(data)
        model.optimize_parameters()

        if total_steps % print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / batch_size

            message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, epoch_iter, t)
            for k, v in errors.items():
                message += '%s: %.3f ' % (k, v)
            print(message)

        # save latest weights
        if total_steps % save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    # save weights periodicaly
    if epoch % save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, niter + niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()
