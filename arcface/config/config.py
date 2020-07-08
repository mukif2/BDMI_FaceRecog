class Config(object):

    use_saved_model = True
    saved_model = 'checkpoints\\resnet18_49.pth'

    env = 'default'
    backbone = 'resnet18'
    classify = 'softmax'
    num_classes = 5699
    metric = 'arc_margin'
    easy_margin = False
    use_se = True
    loss = 'focal_loss'

    display = False
    finetune = False

    train_root = ''
    train_list = 'train.txt'
    val_list = '/data/Datasets/webface/val_data_13938.txt'

    test_root = 'F:\\mtcnnarcface\\mtcnn-pytorch-master\\mtcnn-pytorch-master\\zyf_aligned'
    test_list = 'zyf.txt'

    lfw_root = '/data/Datasets/lfw/lfw-align-128'
    lfw_test_list = '/data/Datasets/lfw/lfw_test_pair.txt'

    checkpoints_path = 'checkpoints'
    load_model_path = 'models/resnet50.pth'
    test_model_path = 'checkpoints/resnet18_110.pth'
    save_interval = 1

    train_batch_size = 16  # batch size
    test_batch_size = 1

    input_shape = (3, 112, 112)

    optimizer = ''

    use_gpu = True  # use GPU or not
    gpu_id = '0'
    num_workers = 4  # how many workers for loading data
    print_freq = 100  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 50
    lr = 1e-3  # initial learning rate
    lr_step = 50
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4
