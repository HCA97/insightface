from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "vit_h_clip"
config.resume = False
config.output = None
config.embedding_size = 1024
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9

# LR and WD for Head
config.weight_decay = 1e-5
config.lr = 3e-4
config.optimizer = 'adamw'
# Batch Size
config.batch_size = 8
config.gradient_acc = 4

config.verbose = 2000
config.dali = False

config.rec = "./train_tmp/product-10k-insightface"
config.num_classes = 9004
config.num_image = 139875
config.num_epoch = 10
config.warmup_step = 1000
config.val_targets = []
