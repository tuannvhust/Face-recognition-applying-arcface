import os 
def get_ckpt_folder(args):
    folder = f'ckpt/{args.model}_{args.metric}_{args.loss}/checkpoints/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder

def get_checkpoint_path(args):
    path = os.path.join(get_ckpt_folder(args), 'model.ckpt')
    return path

def get_log_folder(args):
    folder = get_ckpt_folder(args)
    folder = folder.replace('checkpoints', 'logs')
    return folder