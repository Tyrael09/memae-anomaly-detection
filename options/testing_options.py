import argparse
import re


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class TestOptions:
    def __init__(self):
        """Reset the class; indicates the class hasn't been initialized"""
        self.initialized = False

    def initialize(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--UseCUDA", help="Use CUDA?", type=str2bool, nargs="?", default=True)
        parser.add_argument("--ModelName", help="AE/MemAE", type=str, default="MemAE")
        parser.add_argument("--ModelSetting", help="Conv3D/Conv3DSpar", type=str, default="Conv3DSpar")
        parser.add_argument("--Seed", type=int, default=1)
        parser.add_argument("--Dataset", help="Dataset", type=str, default="Cataract")
        parser.add_argument("--ImgChnNum", help="image channel", type=int, default=1)  
        parser.add_argument("--FrameNum", help="frame num for VIDEO clip", type=int, default=1) 
        parser.add_argument("--BatchSize", help="BatchSize", type=int, default=1)
        parser.add_argument("--MemDim", help="Memory Dimension", type=int, default=1)  
        parser.add_argument("--EntropyLossWeight", help="EntropyLossWeight", type=float, default=0.0002)
        parser.add_argument("--ShrinkThres", help="ShrinkThres", type=float, default=0.0025)
        parser.add_argument("--ModelRoot", help="model dir", type=str, default="/local/scratch/hendrik/models/")
        parser.add_argument(
            "--ModelFilePath",
            help="pretrained model",
            type=str,
            default="/local/scratch/hendrik/models/model_MemAE_MemDim2000_FrameNum16_Overlap0.25/MemAE_MemDim2000_FrameNum16_Overlap0.25_epoch_0100_final.pt",
        )
        parser.add_argument("--OutRoot", help="Path for output", type=str, default="./results/6/") # TODO: adjust
        parser.add_argument("--Overlap", help="Overlap", type=float, default=1)

        self.initialized = True
        self.parser = parser
        return parser

    def extract_model_info(self, model_file_path):
        """Extract ImgChnNum, FrameNum, MemDim, and Overlap from the file path."""
        regex = r"MemDim(\d+)_FrameNum(\d+)_Overlap([\d\.]+)"
        match = re.search(regex, model_file_path)
        if match:
            memdim = int(match.group(1))
            framenum = int(match.group(2))
            overlap = float(match.group(3))
            return memdim, framenum, overlap
        else:
            raise ValueError(f"Could not parse model information from path: {model_file_path}")

    def parse(self, is_print):
        parser = self.initialize()
        opt = parser.parse_args()

        # Extract model-specific parameters from the file path
        memdim, framenum, overlap = self.extract_model_info(opt.ModelFilePath)
        opt.MemDim = memdim
        opt.FrameNum = framenum
        opt.Overlap = overlap

        if is_print:
            self.print_options(opt)
        self.opt = opt
        return self.opt

    def print_options(self, opt):
        message = ""
        message += "----------------- Options ---------------\n"
        for k, v in sorted(vars(opt).items()):
            v = v if v is not None else "None"  # Replace None with a string representation
            comment = ""
            default = self.parser.get_default(k)
            if v != default:
                comment = f"\t[default: {default}]"
            message += f"{k:>25}: {v:<30}{comment}\n"
        message += "----------------- End -------------------"
        print(message)
        self.message = message

