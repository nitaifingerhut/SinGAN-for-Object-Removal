from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions

def ParseNorm(s):
    norm_mode = int(s)
    if (norm_mode != 0 and norm_mode != 1):
        raise argparse.ArgumentTypeError("valid normalization modes are 0,1.")
    return norm_mode

def ParseCoords(s):
    mask_coords = s.replace('(', '').replace(')', '').split(',')
    mask_coords = [int(coord) for coord in mask_coords]
    if (len(mask_coords) != 4):
        raise argparse.ArgumentTypeError("Coordinates must be in format of: (y0,y1),(x0,x1), \n"
                                         "where: y0, y1 are longitude coordinates \n"
                                         "       x1, y1 are latitude coordinates. ")
    return mask_coords

if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='task to be done', default='train')
    parser.add_argument('--norm', default='0', type=ParseNorm, 
                        help="normalization mode: \n"
                        "0:none, 1:normalize. ")
    parser.add_argument('--mask_coords', type=ParseCoords,
                        help="Mask's coordinates in format of: (y0,y1),(x0,x1), \n"
                             "where: y0, y1 are longitude coordinates \n"
                             "       x1, y1 are latitude coordinates. ")
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    reals = []
    masks    = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)

    if (os.path.exists(dir2save)):
        print('trained model already exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        real = functions.read_image(opt)
        functions.adjust_scales2image(real, opt)
        train(opt, Gs, Zs, reals, masks, NoiseAmp)
        SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt)