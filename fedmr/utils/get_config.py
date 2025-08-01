import yaml
from utils.option import args_parser


def get_config(Train=True):
    args = args_parser()
    # with open('./config/{}_{}_{}.yaml'.format(args.attack, args.dataset, args.defence)) as f:
    if Train:
        with open('./config/config.yaml') as f:
            config = yaml.safe_load(f)
        config.update({k: v for k, v in args.__dict__.items() if v is not None})
        args.__dict__ = config
        return args
    else:
        with open('./config/config_eval.yaml') as f:
            config = yaml.safe_load(f)
        config.update({k: v for k, v in args.__dict__.items() if v is not None})
        args.__dict__ = config
        return args