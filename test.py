import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from util.visualizer import Visualizer
from util import html

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()

# read pix2pix/PAN model
if opt.model == 'pix2pix':
    assert(opt.dataset_mode == 'aligned')
    from models.pix2pix_model import Pix2PixModel
    model = Pix2PixModel()
    model.initialize(opt)
elif opt.model == 'pan':
    from models.pan_model import PanModel
    model = PanModel()
    model.initialize(opt)

visualizer = Visualizer(opt)

# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

# test
for i, data in enumerate(dataset):
    if i >= opt.how_many: # default 50 images
        break
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    img_path = model.get_image_paths()
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)

webpage.save()
