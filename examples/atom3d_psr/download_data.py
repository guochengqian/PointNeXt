import atom3d.datasets as da
import os, sys, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from openpoints.utils import cfg

parser = argparse.ArgumentParser('S3DIS scene segmentation training')
parser.add_argument('--cfg', default='cfgs/atom3d_psr/default.yaml', type=str,  help='config file')
args, opts = parser.parse_known_args()
cfg.load(args.cfg, recursive=True)

da.download_dataset(cfg.dataset.NAME, cfg.dataset.common.data_dir, split='year')  # Download LBA dataset.