    #!/usr/bin/python3
#==================================================================#
# KoboldAI
# Version: 1.18.1
# By: KoboldAIDev and the KoboldAI Community
#==================================================================#

# External packages
import eventlet
from eventlet import tpool
eventlet.monkey_patch(all=True, thread=False, os=False)
#eventlet.monkey_patch(os=True, select=True, socket=True, thread=True, time=True, psycopg=True)
import os
os.system("")
__file__ = os.path.dirname(os.path.realpath(__file__))
os.chdir(__file__)
os.environ['EVENTLET_THREADPOOL_SIZE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import logging
logging.getLogger("urllib3").setLevel(logging.ERROR)

from os import path, getcwd
import time
import re
import json
import collections
import zipfile
import packaging
import packaging.version
import contextlib
import traceback
import threading
import markdown
import bleach
import itertools
import bisect
import functools
import traceback
from collections.abc import Iterable
from typing import Any, Callable, TypeVar, Tuple, Union, Dict, Set, List

import requests
import html
import argparse
import sys
import gc

import lupa
import importlib

# KoboldAI
import fileops
import gensettings
from utils import debounce
import utils
import koboldai_settings
import torch
from transformers import StoppingCriteria, GPT2TokenizerFast, GPT2LMHeadModel, GPTNeoForCausalLM, GPTNeoModel, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, modeling_utils
from transformers import __version__ as transformers_version
import transformers
try:
    from transformers.models.opt.modeling_opt import OPTDecoder
except:
    pass
import transformers.generation_utils
global tpu_mtj_backend


if lupa.LUA_VERSION[:2] != (5, 4):
    print(f"Please install lupa==1.10. You have lupa {lupa.__version__}.", file=sys.stderr)

patch_causallm_patched = False

# Make sure tqdm progress bars display properly in Colab
from tqdm.auto import tqdm
old_init = tqdm.__init__
def new_init(self, *args, **kwargs):
    old_init(self, *args, **kwargs)
    if(self.ncols == 0 and kwargs.get("ncols") != 0):
        self.ncols = 99
tqdm.__init__ = new_init

# Fix some issues with the OPT tokenizer
from transformers import PreTrainedTokenizerBase
old_pretrainedtokenizerbase_from_pretrained = PreTrainedTokenizerBase.from_pretrained.__func__
@classmethod
def new_pretrainedtokenizerbase_from_pretrained(cls, *args, **kwargs):
    tokenizer = old_pretrainedtokenizerbase_from_pretrained(cls, *args, **kwargs)
    tokenizer._koboldai_header = tokenizer.encode("")
    tokenizer.add_bos_token = False
    tokenizer.add_prefix_space = False
    return tokenizer
PreTrainedTokenizerBase.from_pretrained = new_pretrainedtokenizerbase_from_pretrained

#==================================================================#
# Variables & Storage
#==================================================================#

# Terminal tags for colored text
class colors:
    PURPLE    = '\033[95m'
    BLUE      = '\033[94m'
    CYAN      = '\033[96m'
    GREEN     = '\033[92m'
    YELLOW    = '\033[93m'
    RED       = '\033[91m'
    END       = '\033[0m'
    UNDERLINE = '\033[4m'

# AI models Menu
# This is a dict of lists where they key is the menu name, and the list is the menu items.
# Each item takes the 4 elements, 1: Text to display, 2: Model Name (koboldai_vars.model) or menu name (Key name for another menu),
# 3: the memory requirement for the model, 4: if the item is a menu or not (True/False)
model_menu = {
    'mainmenu': [
        ["Load a model from its directory", "NeoCustom", "", False],
        ["Load an old GPT-2 model (eg CloverEdition)", "GPT2Custom", "", False],
        ["Adventure Models", "adventurelist", "", True],
        ["Novel Models", "novellist", "", True],
        ["NSFW Models", "nsfwlist", "", True],
        ["Untuned GPT-Neo/J", "gptneolist", "", True],
        ["Untuned Fairseq Dense", "fsdlist", "", True],
        ["Untuned OPT", "optlist", "", True],
        ["Untuned XGLM", "xglmlist", "", True],
        ["Untuned GPT2", "gpt2list", "", True],
        ["Online Services", "apilist", "", True],
        ["Read Only (No AI)", "ReadOnly", "", False]
        ],
    'adventurelist': [
        ["Nerys FSD 13B (Hybrid)", "KoboldAI/fairseq-dense-13B-Nerys", "32GB", False],
        ["Skein 6B", "KoboldAI/GPT-J-6B-Skein", "16GB", False],
        ["Adventure 6B", "KoboldAI/GPT-J-6B-Adventure", "16GB", False],
        ["Nerys FSD 2.7B (Hybrid)", "KoboldAI/fairseq-dense-2.7B-Nerys", "8GB", False],
        ["Adventure 2.7B", "KoboldAI/GPT-Neo-2.7B-AID", "8GB", False],
        ["Adventure 1.3B", "KoboldAI/GPT-Neo-1.3B-Adventure", "6GB", False],
        ["Adventure 125M (Mia)", "Merry/AID-Neo-125M", "2GB", False],
        ["Return to Main Menu", "mainmenu", "", True],
        ],
    'novellist': [
        ["Nerys FSD 13B (Hybrid)", "KoboldAI/fairseq-dense-13B-Nerys", "32GB", False],
        ["Janeway FSD 13B", "KoboldAI/fairseq-dense-13B-Janeway", "32GB", False],
        ["Janeway FSD 6.7B", "KoboldAI/fairseq-dense-6.7B-Janeway", "16GB", False],
        ["Janeway Neo 6B", "KoboldAI/GPT-J-6B-Janeway", "16GB", False],
        ["Janeway Neo 2.7B", "KoboldAI/GPT-Neo-2.7B-Janeway", "8GB", False],
        ["Janeway FSD 2.7B", "KoboldAI/fairseq-dense-2.7B-Janeway", "8GB", False],
        ["Nerys FSD 2.7B (Hybrid)", "KoboldAI/fairseq-dense-2.7B-Nerys", "8GB", False],
        ["Horni-LN 2.7B", "KoboldAI/GPT-Neo-2.7B-Horni-LN", "8GB", False],
        ["Picard 2.7B (Older Janeway)", "KoboldAI/GPT-Neo-2.7B-Picard", "8GB", False],
        ["Return to Main Menu", "mainmenu", "", True],
        ],
    'nsfwlist': [
        ["Shinen FSD 13B (NSFW)", "KoboldAI/fairseq-dense-13B-Shinen", "32GB", False],
        ["Shinen FSD 6.7B (NSFW)", "KoboldAI/fairseq-dense-6.7B-Shinen", "16GB", False],
        ["Lit 6B (NSFW)", "hakurei/lit-6B", "16GB", False],
        ["Shinen 6B (NSFW)", "KoboldAI/GPT-J-6B-Shinen", "16GB", False],
        ["Horni 2.7B (NSFW)", "KoboldAI/GPT-Neo-2.7B-Horni", "8GB", False],
        ["Shinen 2.7B (NSFW)", "KoboldAI/GPT-Neo-2.7B-Shinen", "8GB", False],
        ["Return to Main Menu", "mainmenu", "", True],
        ],
    'chatlist': [
        ["Convo 6B (Chatbot)", "hitomi-team/convo-6B", "16GB", False],
        ["C1 6B (Chatbot)", "hakurei/c1-6B", "16GB", False],
        ["C1 1.3B (Chatbot)", "iokru/c1-1.3B", "6GB", False],
        ["Return to Main Menu", "mainmenu", "", True],
        ],
    'gptneolist': [
        ["GPT-J 6B", "EleutherAI/gpt-j-6B", "16GB", False],
        ["GPT-Neo 2.7B", "EleutherAI/gpt-neo-2.7B", "8GB", False],
        ["GPT-Neo 1.3B", "EleutherAI/gpt-neo-1.3B", "6GB", False],
        ["GPT-Neo 125M", "EleutherAI/gpt-neo-125M", "2GB", False],
        ["Return to Main Menu", "mainmenu", "", True],
        ],
    'gpt2list': [
        ["GPT-2 XL", "gpt2-xl", "6GB", False],
        ["GPT-2 Large", "gpt2-large", "4GB", False],
        ["GPT-2 Med", "gpt2-medium", "2GB", False],
        ["GPT-2", "gpt2", "2GB", False],
        ["Return to Main Menu", "mainmenu", "", True],
        ],
    'optlist': [
        ["OPT 30B", "facebook/opt-30b", "64GB", False],
        ["OPT 13B", "facebook/opt-13b", "32GB", False],
        ["OPT 6.7B", "facebook/opt-6.7b", "16GB", False],
        ["OPT 2.7B", "facebook/opt-2.7b", "8GB", False],
        ["OPT 1.3B", "facebook/opt-1.3b", "4GB", False],
        ["OPT 350M", "facebook/opt-350m", "2GB", False],
        ["OPT 125M", "facebook/opt-125m", "1GB", False],
        ["Return to Main Menu", "mainmenu", "", True],
        ],
    'fsdlist': [
        ["Fairseq Dense 13B", "KoboldAI/fairseq-dense-13B", "32GB", False],
        ["Fairseq Dense 6.7B", "KoboldAI/fairseq-dense-6.7B", "16GB", False],
        ["Fairseq Dense 2.7B", "KoboldAI/fairseq-dense-2.7B", "8GB", False],
        ["Fairseq Dense 1.3B", "KoboldAI/fairseq-dense-1.3B", "4GB", False],
        ["Fairseq Dense 355M", "KoboldAI/fairseq-dense-355M", "2GB", False],
        ["Fairseq Dense 125M", "KoboldAI/fairseq-dense-125M", "1GB", False],
        ["Return to Main Menu", "mainmenu", "", True],
        ],
    'xglmlist': [
        ["XGLM 4.5B (Larger Dataset)", "facebook/xglm-4.5B", "12GB", False],
        ["XGLM 7.5B", "facebook/xglm-7.5B", "18GB", False],
        ["XGLM 2.9B", "facebook/xglm-2.9B", "10GB", False],
        ["XGLM 1.7B", "facebook/xglm-1.7B", "6GB", False],
        ["XGLM 564M", "facebook/xglm-564M", "4GB", False],
        ["Return to Main Menu", "mainmenu", "", True],
        ],
    'apilist': [
        ["GooseAI API (requires API key)", "GooseAI", "", False],
        ["OpenAI API (requires API key)", "OAI", "", False],
        ["InferKit API (requires API key)", "InferKit", "", False],
        ["KoboldAI Server API (Old Google Colab)", "Colab", "", False],
        ["Return to Main Menu", "mainmenu", "", True],
    ]
    }

class Send_to_socketio(object):
    def write(self, bar):
        print(bar, end="")
        time.sleep(0.01)
        try:
            emit('from_server', {'cmd': 'model_load_status', 'data': bar.replace(" ", "&nbsp;")}, broadcast=True, room="UI_1")
        except:
            pass
                                
# Set logging level to reduce chatter from Flask
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Start flask & SocketIO
print("{0}Initializing Flask... {1}".format(colors.PURPLE, colors.END), end="")
from flask import Flask, render_template, Response, request, copy_current_request_context, send_from_directory, session, has_request_context
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_session import Session
import secrets
app = Flask(__name__, root_path=os.getcwd())
app.secret_key = secrets.token_hex()
app.config['SESSION_TYPE'] = 'filesystem'
app.config['TEMPLATES_AUTO_RELOAD'] = True
Session(app)
socketio = SocketIO(app, async_method="eventlet", manage_session=False)
#socketio = SocketIO(app, async_method="eventlet", logger=True, engineio_logger=True, manage_session=False)
koboldai_vars = koboldai_settings.koboldai_vars(session, socketio)

utils.koboldai_vars = koboldai_vars

print("{0}OK!{1}".format(colors.GREEN, colors.END))

#==================================================================#
# Function to get model selection at startup
#==================================================================#
def sendModelSelection(menu="mainmenu", folder="./models"):
    #If we send one of the manual load options, send back the list of model directories, otherwise send the menu
    if menu in ('NeoCustom', 'GPT2Custom'):
        (paths, breadcrumbs) = get_folder_path_info(folder)
        if koboldai_vars.host:
            breadcrumbs = []
        menu_list = [[folder, menu, "", False] for folder in paths]
        menu_list_ui_2 = [[folder[0], folder[1], "", False] for folder in paths]
        menu_list.append(["Return to Main Menu", "mainmenu", "", True])
        menu_list_ui_2.append(["Return to Main Menu", "mainmenu", "", True])
        if os.path.abspath("{}/models".format(os.getcwd())) == os.path.abspath(folder):
            showdelete=True
        else:
            showdelete=False
        emit('from_server', {'cmd': 'show_model_menu', 'data': menu_list, 'menu': menu, 'breadcrumbs': breadcrumbs, "showdelete": showdelete}, broadcast=True, room="UI_1")
        emit('show_model_menu', {'data': menu_list_ui_2, 'menu': menu, 'breadcrumbs': breadcrumbs, "showdelete": showdelete}, broadcast=False, room="UI_2")
    else:
        emit('from_server', {'cmd': 'show_model_menu', 'data': model_menu[menu], 'menu': menu, 'breadcrumbs': [], "showdelete": False}, broadcast=True, room="UI_1")
        emit('show_model_menu', {'data': model_menu[menu], 'menu': menu, 'breadcrumbs': [], "showdelete": False}, broadcast=False, room="UI_2")

def get_folder_path_info(base):
    if base == 'This PC':
        breadcrumbs = [['This PC', 'This PC']]
        paths = [["{}:\\".format(chr(i)), "{}:\\".format(chr(i))] for i in range(65, 91) if os.path.exists("{}:".format(chr(i)))]
    else:
        path = os.path.abspath(base)
        if path[-1] == "\\":
            path = path[:-1]
        breadcrumbs = []
        for i in range(len(path.replace("/", "\\").split("\\"))):
            breadcrumbs.append(["\\".join(path.replace("/", "\\").split("\\")[:i+1]),
                                 path.replace("/", "\\").split("\\")[i]])
        if len(breadcrumbs) == 1:
            breadcrumbs = [["{}:\\".format(chr(i)), "{}:\\".format(chr(i))] for i in range(65, 91) if os.path.exists("{}:".format(chr(i)))]
        else:
            if len([["{}:\\".format(chr(i)), "{}:\\".format(chr(i))] for i in range(65, 91) if os.path.exists("{}:".format(chr(i)))]) > 0:
                breadcrumbs.insert(0, ['This PC', 'This PC'])
        paths = []
        base_path = os.path.abspath(base)
        for item in os.listdir(base_path):
            if os.path.isdir(os.path.join(base_path, item)):
                paths.append([os.path.join(base_path, item), item])
    # Paths/breadcrumbs is a list of lists, where the first element in the sublist is the full path and the second is the folder name
    return (paths, breadcrumbs)


def getModelSelection(modellist):
    print("    #    Model\t\t\t\t\t\tVRAM\n    ========================================================")
    i = 1
    for m in modellist:
        print("    {0} - {1}\t\t\t{2}".format("{:<2}".format(i), m[0].ljust(25), m[2]))
        i += 1
    print(" ");
    modelsel = 0
    koboldai_vars.model = ''
    while(koboldai_vars.model == ''):
        modelsel = input("Model #> ")
        if(modelsel.isnumeric() and int(modelsel) > 0 and int(modelsel) <= len(modellist)):
            koboldai_vars.model = modellist[int(modelsel)-1][1]
        else:
            print("{0}Please enter a valid selection.{1}".format(colors.RED, colors.END))
    
    # Model Lists
    try:
        getModelSelection(eval(koboldai_vars.model))
    except Exception as e:
        if(koboldai_vars.model == "Return"):
            getModelSelection(mainmenu)
                
        # If custom model was selected, get the filesystem location and store it
        if(koboldai_vars.model == "NeoCustom" or koboldai_vars.model == "GPT2Custom"):
            print("{0}Please choose the folder where pytorch_model.bin is located:{1}\n".format(colors.CYAN, colors.END))
            modpath = fileops.getdirpath(getcwd() + "/models", "Select Model Folder")
        
            if(modpath):
                # Save directory to vars
                koboldai_vars.custmodpth = modpath
            else:
                # Print error and retry model selection
                print("{0}Model select cancelled!{1}".format(colors.RED, colors.END))
                print("{0}Select an AI model to continue:{1}\n".format(colors.CYAN, colors.END))
                getModelSelection(mainmenu)

def check_if_dir_is_model(path):
    if os.path.exists(path):
        try:
            from transformers import AutoConfig
            model_config = AutoConfig.from_pretrained(path)
        except:
            return False
        return True
    else:
        return False
    
#==================================================================#
# Return all keys in tokenizer dictionary containing char
#==================================================================#
#def gettokenids(char):
#    keys = []
#    for key in vocab_keys:
#        if(key.find(char) != -1):
#            keys.append(key)
#    return keys

#==================================================================#
# Return Model Name
#==================================================================#
def getmodelname():
    if(args.configname):
       modelname = args.configname
       return modelname
    if(koboldai_vars.model in ("NeoCustom", "GPT2Custom", "TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX")):
        modelname = os.path.basename(os.path.normpath(koboldai_vars.custmodpth))
        return modelname
    else:
        modelname = koboldai_vars.model
        return modelname

#==================================================================#
# Breakmodel configuration functions
#==================================================================#
def device_list(n_layers, primary=None, selected=None):
    device_count = torch.cuda.device_count()
    if(device_count < 2):
        primary = None
    gpu_blocks = breakmodel.gpu_blocks + (device_count - len(breakmodel.gpu_blocks))*[0]
    print(f"{colors.YELLOW}       DEVICE ID  |  LAYERS  |  DEVICE NAME{colors.END}")
    for i in range(device_count):
        name = torch.cuda.get_device_name(i)
        if(len(name) > 47):
            name = "..." + name[-44:]
        row_color = colors.END
        sep_color = colors.YELLOW
        print(f"{row_color}{colors.YELLOW + '->' + row_color if i == selected else '  '} {'(primary)' if i == primary else ' '*9} {i:3}  {sep_color}|{row_color}     {gpu_blocks[i]:3}  {sep_color}|{row_color}  {name}{colors.END}")
    row_color = colors.END
    sep_color = colors.YELLOW
    if(utils.HAS_ACCELERATE):
        print(f"{row_color}{colors.YELLOW + '->' + row_color if -1 == selected else '  '} {' '*9} N/A  {sep_color}|{row_color}     {breakmodel.disk_blocks:3}  {sep_color}|{row_color}  (Disk cache){colors.END}")
    print(f"{row_color}   {' '*9} N/A  {sep_color}|{row_color}     {n_layers:3}  {sep_color}|{row_color}  (CPU){colors.END}")

def device_config(config):
    global breakmodel, generator
    import breakmodel
    n_layers = utils.num_layers(config)
    if(args.breakmodel_gpulayers is not None or (utils.HAS_ACCELERATE and args.breakmodel_disklayers is not None)):
        try:
            if(not args.breakmodel_gpulayers):
                breakmodel.gpu_blocks = []
            else:
                breakmodel.gpu_blocks = list(map(int, args.breakmodel_gpulayers.split(',')))
            assert len(breakmodel.gpu_blocks) <= torch.cuda.device_count()
            s = n_layers
            for i in range(len(breakmodel.gpu_blocks)):
                if(breakmodel.gpu_blocks[i] <= -1):
                    breakmodel.gpu_blocks[i] = s
                    break
                else:
                    s -= breakmodel.gpu_blocks[i]
            assert sum(breakmodel.gpu_blocks) <= n_layers
            n_layers -= sum(breakmodel.gpu_blocks)
            if(args.breakmodel_disklayers is not None):
                assert args.breakmodel_disklayers <= n_layers
                breakmodel.disk_blocks = args.breakmodel_disklayers
                n_layers -= args.breakmodel_disklayers
        except:
            print("WARNING: --breakmodel_gpulayers is malformatted. Please use the --help option to see correct usage of --breakmodel_gpulayers. Defaulting to all layers on device 0.", file=sys.stderr)
            breakmodel.gpu_blocks = [n_layers]
            n_layers = 0
    elif(args.breakmodel_layers is not None):
        breakmodel.gpu_blocks = [n_layers - max(0, min(n_layers, args.breakmodel_layers))]
        n_layers -= sum(breakmodel.gpu_blocks)
    elif(args.model is not None):
        print("Breakmodel not specified, assuming GPU 0")
        breakmodel.gpu_blocks = [n_layers]
        n_layers = 0
    else:
        device_count = torch.cuda.device_count()
        if(device_count > 1):
            print(colors.CYAN + "\nPlease select one of your GPUs to be your primary GPU.")
            print("VRAM usage in your primary GPU will be higher than for your other ones.")
            print("It is recommended you make your fastest GPU your primary GPU.")
            device_list(n_layers)
            while(True):
                primaryselect = input("device ID> ")
                if(primaryselect.isnumeric() and 0 <= int(primaryselect) < device_count):
                    breakmodel.primary_device = int(primaryselect)
                    break
                else:
                    print(f"{colors.RED}Please enter an integer between 0 and {device_count-1}.{colors.END}")
        else:
            breakmodel.primary_device = 0

        print(colors.PURPLE + "\nIf you don't have enough VRAM to run the model on a single GPU")
        print("you can split the model between your CPU and your GPU(s), or between")
        print("multiple GPUs if you have more than one.")
        print("By putting more 'layers' on a GPU or CPU, more computations will be")
        print("done on that device and more VRAM or RAM will be required on that device")
        print("(roughly proportional to number of layers).")
        print("It should be noted that GPUs are orders of magnitude faster than the CPU.")
        print(f"This model has{colors.YELLOW} {n_layers} {colors.PURPLE}layers.{colors.END}\n")

        for i in range(device_count):
            device_list(n_layers, primary=breakmodel.primary_device, selected=i)
            print(f"{colors.CYAN}\nHow many of the remaining{colors.YELLOW} {n_layers} {colors.CYAN}layers would you like to put into device {i}?\nYou can also enter -1 to allocate all remaining layers to this device.{colors.END}\n")
            while(True):
                layerselect = input("# of layers> ")
                if((layerselect.isnumeric() or layerselect.strip() == '-1') and -1 <= int(layerselect) <= n_layers):
                    layerselect = int(layerselect)
                    layerselect = n_layers if layerselect == -1 else layerselect
                    breakmodel.gpu_blocks.append(layerselect)
                    n_layers -= layerselect
                    break
                else:
                    print(f"{colors.RED}Please enter an integer between -1 and {n_layers}.{colors.END}")
            if(n_layers == 0):
                break

        if(utils.HAS_ACCELERATE and n_layers > 0):
            device_list(n_layers, primary=breakmodel.primary_device, selected=-1)
            print(f"{colors.CYAN}\nHow many of the remaining{colors.YELLOW} {n_layers} {colors.CYAN}layers would you like to put into the disk cache?\nYou can also enter -1 to allocate all remaining layers to this device.{colors.END}\n")
            while(True):
                layerselect = input("# of layers> ")
                if((layerselect.isnumeric() or layerselect.strip() == '-1') and -1 <= int(layerselect) <= n_layers):
                    layerselect = int(layerselect)
                    layerselect = n_layers if layerselect == -1 else layerselect
                    breakmodel.disk_blocks = layerselect
                    n_layers -= layerselect
                    break
                else:
                    print(f"{colors.RED}Please enter an integer between -1 and {n_layers}.{colors.END}")

    print(colors.PURPLE + "\nFinal device configuration:")
    device_list(n_layers)

    # If all layers are on the same device, use the old GPU generation mode
    while(len(breakmodel.gpu_blocks) and breakmodel.gpu_blocks[-1] == 0):
        breakmodel.gpu_blocks.pop()
    if(len(breakmodel.gpu_blocks) and breakmodel.gpu_blocks[-1] in (-1, utils.num_layers(config))):
        koboldai_vars.breakmodel = False
        koboldai_vars.usegpu = True
        koboldai_vars.gpu_device = len(breakmodel.gpu_blocks)-1
        return

    if(not breakmodel.gpu_blocks):
        print("Nothing assigned to a GPU, reverting to CPU only mode")
        import breakmodel
        breakmodel.primary_device = "cpu"
        koboldai_vars.breakmodel = False
        koboldai_vars.usegpu = False
        return

def move_model_to_devices(model):
    global generator

    if(not utils.HAS_ACCELERATE and not koboldai_vars.breakmodel):
        if(koboldai_vars.usegpu):
            model = model.half().to(koboldai_vars.gpu_device)
        else:
            model = model.to('cpu').float()
        generator = model.generate
        return

    import breakmodel

    if(utils.HAS_ACCELERATE):
        disk_blocks = breakmodel.disk_blocks
        gpu_blocks = breakmodel.gpu_blocks
        ram_blocks = len(utils.layers_module_names) - sum(gpu_blocks)
        cumulative_gpu_blocks = tuple(itertools.accumulate(gpu_blocks))
        device_map = {}
        for name in utils.layers_module_names:
            layer = int(name.rsplit(".", 1)[1])
            device = ("disk" if layer < disk_blocks else "cpu") if layer < ram_blocks else bisect.bisect_right(cumulative_gpu_blocks, layer - ram_blocks)
            device_map[name] = device
        for name in utils.get_missing_module_names(model, list(device_map.keys())):
            device_map[name] = breakmodel.primary_device
        breakmodel.dispatch_model_ex(model, device_map, main_device=breakmodel.primary_device, offload_buffers=True, offload_dir="accelerate-disk-cache")
        gc.collect()
        generator = model.generate
        return

    model.half()
    gc.collect()

    if(hasattr(model, "transformer")):
        model.transformer.wte.to(breakmodel.primary_device)
        model.transformer.ln_f.to(breakmodel.primary_device)
        if(hasattr(model, 'lm_head')):
            model.lm_head.to(breakmodel.primary_device)
        if(hasattr(model.transformer, 'wpe')):
            model.transformer.wpe.to(breakmodel.primary_device)
    elif(not hasattr(model.model, "decoder")):
        model.model.embed_tokens.to(breakmodel.primary_device)
        model.model.layer_norm.to(breakmodel.primary_device)
        model.lm_head.to(breakmodel.primary_device)
        model.model.embed_positions.to(breakmodel.primary_device)
    else:
        model.model.decoder.embed_tokens.to(breakmodel.primary_device)
        if(model.model.decoder.project_in is not None):
            model.model.decoder.project_in.to(breakmodel.primary_device)
        if(model.model.decoder.project_out is not None):
            model.model.decoder.project_out.to(breakmodel.primary_device)
        model.model.decoder.embed_positions.to(breakmodel.primary_device)
    gc.collect()
    GPTNeoModel.forward = breakmodel.new_forward_neo
    if("GPTJModel" in globals()):
        GPTJModel.forward = breakmodel.new_forward_neo # type: ignore
    if("XGLMModel" in globals()):
        XGLMModel.forward = breakmodel.new_forward_xglm # type: ignore
    if("OPTDecoder" in globals()):
        OPTDecoder.forward = breakmodel.new_forward_opt # type: ignore
    generator = model.generate
    if(hasattr(model, "transformer")):
        breakmodel.move_hidden_layers(model.transformer)
    elif(not hasattr(model.model, "decoder")):
        breakmodel.move_hidden_layers(model.model, model.model.layers)
    else:
        breakmodel.move_hidden_layers(model.model.decoder, model.model.decoder.layers)

#==================================================================#
#  Allow the models to override some settings
#==================================================================#
def loadmodelsettings():
    try:
        js   = json.loads(str(model_config).partition(' ')[2])
    except Exception as e:
        try:
            try:
                js   = json.load(open(koboldai_vars.custmodpth + "/config.json", "r"))
            except Exception as e:
                js   = json.load(open(koboldai_vars.custmodpth.replace('/', '_') + "/config.json", "r"))            
        except Exception as e:
            js   = {}
    if koboldai_vars.model_type == "xglm" or js.get("compat", "j") == "fairseq_lm":
        koboldai_vars.newlinemode = "s"  # Default to </s> newline mode if using XGLM
    if koboldai_vars.model_type == "opt" or koboldai_vars.model_type == "bloom":
        koboldai_vars.newlinemode = "ns"  # Handle </s> but don't convert newlines if using Fairseq models that have newlines trained in them
    koboldai_vars.modelconfig = js
    if("badwordsids" in js):
        koboldai_vars.badwordsids = js["badwordsids"]
    if("nobreakmodel" in js):
        koboldai_vars.nobreakmodel = js["nobreakmodel"]
    if("sampler_order" in js):
        koboldai_vars.sampler_order = js["sampler_order"]
        koboldai_vars.default_preset['sampler_order'] = js["sampler_order"]
    if("temp" in js):
        koboldai_vars.temp       = js["temp"]
        koboldai_vars.default_preset['temp'] = js["temp"]
    if("top_p" in js):
        koboldai_vars.top_p      = js["top_p"]
        koboldai_vars.default_preset['top_p'] = js["top_p"]
    if("top_k" in js):
        koboldai_vars.top_k      = js["top_k"]
        koboldai_vars.default_preset['top_k'] = js["top_k"]
    if("tfs" in js):
        koboldai_vars.tfs        = js["tfs"]
        koboldai_vars.default_preset['tfs'] = js["tfs"]
    if("typical" in js):
        koboldai_vars.typical    = js["typical"]
        koboldai_vars.default_preset['typical'] = js["typical"]
    if("top_a" in js):
        koboldai_vars.top_a      = js["top_a"]
        koboldai_vars.default_preset['top_a'] = js["top_a"]
    if("rep_pen" in js):
        koboldai_vars.rep_pen    = js["rep_pen"]
        koboldai_vars.default_preset['rep_pen'] = js["rep_pen"]
    if("rep_pen_slope" in js):
        koboldai_vars.rep_pen_slope = js["rep_pen_slope"]
        koboldai_vars.default_preset['rep_pen_slope'] = js["rep_pen_slope"]
    if("rep_pen_range" in js):
        koboldai_vars.rep_pen_range = js["rep_pen_range"]
        koboldai_vars.default_preset['rep_pen_range'] = js["rep_pen_range"]
    if("adventure" in js):
        koboldai_vars.adventure = js["adventure"]
    if("chatmode" in js):
        koboldai_vars.chatmode = js["chatmode"]
    if("dynamicscan" in js):
        koboldai_vars.dynamicscan = js["dynamicscan"]
    if("formatoptns" in js):
        koboldai_vars.formatoptns = js["formatoptns"]
    if("welcome" in js):
        koboldai_vars.welcome = js["welcome"]
    if("newlinemode" in js):
        koboldai_vars.newlinemode = js["newlinemode"]
    if("antemplate" in js):
        koboldai_vars.setauthornotetemplate = js["antemplate"]
        if(not koboldai_vars.gamestarted):
            koboldai_vars.authornotetemplate = koboldai_vars.setauthornotetemplate

#==================================================================#
#  Take settings from vars and write them to client settings file
#==================================================================#
def savesettings():
     # Build json to write
    js = {}
    js["apikey"]      = koboldai_vars.apikey
    js["andepth"]     = koboldai_vars.andepth
    js["sampler_order"] = koboldai_vars.sampler_order
    js["temp"]        = koboldai_vars.temp
    js["top_p"]       = koboldai_vars.top_p
    js["top_k"]       = koboldai_vars.top_k
    js["tfs"]         = koboldai_vars.tfs
    js["typical"]     = koboldai_vars.typical
    js["top_a"]       = koboldai_vars.top_a
    js["rep_pen"]     = koboldai_vars.rep_pen
    js["rep_pen_slope"] = koboldai_vars.rep_pen_slope
    js["rep_pen_range"] = koboldai_vars.rep_pen_range
    js["genamt"]      = koboldai_vars.genamt
    js["max_length"]  = koboldai_vars.max_length
    js["ikgen"]       = koboldai_vars.ikgen
    js["formatoptns"] = koboldai_vars.formatoptns
    js["numseqs"]     = koboldai_vars.numseqs
    js["widepth"]     = koboldai_vars.widepth
    js["useprompt"]   = koboldai_vars.useprompt
    js["adventure"]   = koboldai_vars.adventure
    js["chatmode"]    = koboldai_vars.chatmode
    js["chatname"]    = koboldai_vars.chatname
    js["dynamicscan"] = koboldai_vars.dynamicscan
    js["nopromptgen"] = koboldai_vars.nopromptgen
    js["rngpersist"]  = koboldai_vars.rngpersist
    js["nogenmod"]    = koboldai_vars.nogenmod
    js["autosave"]    = koboldai_vars.autosave
    js["welcome"]     = koboldai_vars.welcome
    js["newlinemode"] = koboldai_vars.newlinemode
    js["output_streaming"] = koboldai_vars.output_streaming

    js["antemplate"]  = koboldai_vars.setauthornotetemplate

    js["userscripts"] = koboldai_vars.userscripts
    js["corescript"]  = koboldai_vars.corescript
    js["softprompt"]  = koboldai_vars.spfilename

    # Write it
    if not os.path.exists('settings'):
        os.mkdir('settings')
    file = open("settings/" + getmodelname().replace('/', '_') + ".settings", "w")
    try:
        file.write(json.dumps(js, indent=3))
    finally:
        file.close()

#==================================================================#
#  Don't save settings unless 2 seconds have passed without modification
#==================================================================#
@debounce(2)
def settingschanged():
    print("{0}Saving settings!{1}".format(colors.GREEN, colors.END))
    savesettings()

#==================================================================#
#  Read settings from client file JSON and send to vars
#==================================================================#

def loadsettings():
    if(path.exists("defaults/" + getmodelname().replace('/', '_') + ".settings")):
        # Read file contents into JSON object
        file = open("defaults/" + getmodelname().replace('/', '_') + ".settings", "r")
        js   = json.load(file)
        
        processsettings(js)
        file.close()
    if(path.exists("settings/" + getmodelname().replace('/', '_') + ".settings")):
        # Read file contents into JSON object
        file = open("settings/" + getmodelname().replace('/', '_') + ".settings", "r")
        js   = json.load(file)
        
        processsettings(js)
        file.close()
        
def processsettings(js):
# Copy file contents to vars
    if("apikey" in js):
        koboldai_vars.apikey     = js["apikey"]
    if("andepth" in js):
        koboldai_vars.andepth    = js["andepth"]
    if("sampler_order" in js):
        koboldai_vars.sampler_order = js["sampler_order"]
    if("temp" in js):
        koboldai_vars.temp       = js["temp"]
    if("top_p" in js):
        koboldai_vars.top_p      = js["top_p"]
    if("top_k" in js):
        koboldai_vars.top_k      = js["top_k"]
    if("tfs" in js):
        koboldai_vars.tfs        = js["tfs"]
    if("typical" in js):
        koboldai_vars.typical    = js["typical"]
    if("top_a" in js):
        koboldai_vars.top_a      = js["top_a"]
    if("rep_pen" in js):
        koboldai_vars.rep_pen    = js["rep_pen"]
    if("rep_pen_slope" in js):
        koboldai_vars.rep_pen_slope = js["rep_pen_slope"]
    if("rep_pen_range" in js):
        koboldai_vars.rep_pen_range = js["rep_pen_range"]
    if("genamt" in js):
        koboldai_vars.genamt     = js["genamt"]
    if("max_length" in js):
        koboldai_vars.max_length = js["max_length"]
    if("ikgen" in js):
        koboldai_vars.ikgen      = js["ikgen"]
    if("formatoptns" in js):
        koboldai_vars.formatoptns = js["formatoptns"]
    if("numseqs" in js):
        koboldai_vars.numseqs = js["numseqs"]
    if("widepth" in js):
        koboldai_vars.widepth = js["widepth"]
    if("useprompt" in js):
        koboldai_vars.useprompt = js["useprompt"]
    if("adventure" in js):
        koboldai_vars.adventure = js["adventure"]
    if("chatmode" in js):
        koboldai_vars.chatmode = js["chatmode"]
    if("chatname" in js):
        koboldai_vars.chatname = js["chatname"]
    if("dynamicscan" in js):
        koboldai_vars.dynamicscan = js["dynamicscan"]
    if("nopromptgen" in js):
        koboldai_vars.nopromptgen = js["nopromptgen"]
    if("rngpersist" in js):
        koboldai_vars.rngpersist = js["rngpersist"]
    if("nogenmod" in js):
        koboldai_vars.nogenmod = js["nogenmod"]
    if("autosave" in js):
        koboldai_vars.autosave = js["autosave"]
    if("newlinemode" in js):
        koboldai_vars.newlinemode = js["newlinemode"]
    if("welcome" in js):
        koboldai_vars.welcome = js["welcome"]
    if("output_streaming" in js):
        koboldai_vars.autosave = js["output_streaming"]

    if("antemplate" in js):
        koboldai_vars.setauthornotetemplate = js["antemplate"]
        if(not koboldai_vars.gamestarted):
            koboldai_vars.authornotetemplate = koboldai_vars.setauthornotetemplate
    
    if("userscripts" in js):
        koboldai_vars.userscripts = []
        for userscript in js["userscripts"]:
            if type(userscript) is not str:
                continue
            userscript = userscript.strip()
            if len(userscript) != 0 and all(q not in userscript for q in ("..", ":")) and all(userscript[0] not in q for q in ("/", "\\")) and os.path.exists(fileops.uspath(userscript)):
                koboldai_vars.userscripts.append(userscript)

    if("corescript" in js and type(js["corescript"]) is str and all(q not in js["corescript"] for q in ("..", ":")) and all(js["corescript"][0] not in q for q in ("/", "\\"))):
        koboldai_vars.corescript = js["corescript"]
    else:
        koboldai_vars.corescript = "default.lua"

#==================================================================#
#  Load a soft prompt from a file
#==================================================================#

def check_for_sp_change():
    while(True):
        time.sleep(0.1)
        if(koboldai_vars.sp_changed):
            with app.app_context():
                emit('from_server', {'cmd': 'spstatitems', 'data': {koboldai_vars.spfilename: koboldai_vars.spmeta} if koboldai_vars.allowsp and len(koboldai_vars.spfilename) else {}}, namespace=None, broadcast=True, room="UI_1")
            koboldai_vars.sp_changed = False

socketio.start_background_task(check_for_sp_change)

def spRequest(filename):
    if(not koboldai_vars.allowsp):
        raise RuntimeError("Soft prompts are not supported by your current model/backend")
    
    old_filename = koboldai_vars.spfilename

    koboldai_vars.spfilename = ""
    settingschanged()

    if(len(filename) == 0):
        koboldai_vars.sp = None
        koboldai_vars.sp_length = 0
        if(old_filename != filename):
            koboldai_vars.sp_changed = True
        return

    global np
    if 'np' not in globals():
        import numpy as np

    z, version, shape, fortran_order, dtype = fileops.checksp(filename, koboldai_vars.modeldim)
    if not isinstance(z, zipfile.ZipFile):
        raise RuntimeError(f"{repr(filename)} is not a valid soft prompt file")
    with z.open('meta.json') as f:
        koboldai_vars.spmeta = json.load(f)
    z.close()

    with np.load(fileops.sppath(filename), allow_pickle=False) as f:
        tensor = f['tensor.npy']

    # If the tensor is in bfloat16 format, convert it to float32
    if(tensor.dtype == 'V2'):
        tensor.dtype = np.uint16
        tensor = np.uint32(tensor) << 16
        tensor.dtype = np.float32

    if(tensor.dtype != np.float16):
        tensor = np.float32(tensor)
    assert not np.isinf(tensor).any() and not np.isnan(tensor).any()

    koboldai_vars.sp_length = tensor.shape[-2]
    koboldai_vars.spmeta["n_tokens"] = koboldai_vars.sp_length

    if(koboldai_vars.use_colab_tpu or koboldai_vars.model in ("TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX")):
        rows = tensor.shape[0]
        padding_amount = tpu_mtj_backend.params["seq"] - (tpu_mtj_backend.params["seq"] % -tpu_mtj_backend.params["cores_per_replica"]) - rows
        tensor = np.pad(tensor, ((0, padding_amount), (0, 0)))
        tensor = tensor.reshape(
            tpu_mtj_backend.params["cores_per_replica"],
            -1,
            tpu_mtj_backend.params.get("d_embed", tpu_mtj_backend.params["d_model"]),
        )
        koboldai_vars.sp = tpu_mtj_backend.shard_xmap(np.float32(tensor))
    else:
        koboldai_vars.sp = torch.from_numpy(tensor)

    koboldai_vars.spfilename = filename
    settingschanged()
    if(old_filename != filename):
            koboldai_vars.sp_changed = True

#==================================================================#
# Startup
#==================================================================#
def general_startup(override_args=None):
    global args
    # Parsing Parameters
    parser = argparse.ArgumentParser(description="KoboldAI Server")
    parser.add_argument("--remote", action='store_true', help="Optimizes KoboldAI for Remote Play")
    parser.add_argument("--noaimenu", action='store_true', help="Disables the ability to select the AI")
    parser.add_argument("--ngrok", action='store_true', help="Optimizes KoboldAI for Remote Play using Ngrok")
    parser.add_argument("--localtunnel", action='store_true', help="Optimizes KoboldAI for Remote Play using Localtunnel")
    parser.add_argument("--host", action='store_true', help="Optimizes KoboldAI for Remote Play without using a proxy service")
    parser.add_argument("--port", type=int, help="Specify the port on which the application will be joinable")
    parser.add_argument("--aria2_port", type=int, help="Specify the port on which aria2's RPC interface will be open if aria2 is installed (defaults to 6799)")
    parser.add_argument("--model", help="Specify the Model Type to skip the Menu")
    parser.add_argument("--path", help="Specify the Path for local models (For model NeoCustom or GPT2Custom)")
    parser.add_argument("--revision", help="Specify the model revision for huggingface models (can be a git branch/tag name or a git commit hash)")
    parser.add_argument("--cpu", action='store_true', help="By default unattended launches are on the GPU use this option to force CPU usage.")
    parser.add_argument("--breakmodel", action='store_true', help=argparse.SUPPRESS)
    parser.add_argument("--breakmodel_layers", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--breakmodel_gpulayers", type=str, help="If using a model that supports hybrid generation, this is a comma-separated list that specifies how many layers to put on each GPU device. For example to put 8 layers on device 0, 9 layers on device 1 and 11 layers on device 2, use --beakmodel_gpulayers 8,9,11")
    parser.add_argument("--breakmodel_disklayers", type=int, help="If using a model that supports hybrid generation, this is the number of layers to put in disk cache.")
    parser.add_argument("--override_delete", action='store_true', help="Deleting stories from inside the browser is disabled if you are using --remote and enabled otherwise. Using this option will instead allow deleting stories if using --remote and prevent deleting stories otherwise.")
    parser.add_argument("--override_rename", action='store_true', help="Renaming stories from inside the browser is disabled if you are using --remote and enabled otherwise. Using this option will instead allow renaming stories if using --remote and prevent renaming stories otherwise.")
    parser.add_argument("--configname", help="Force a fixed configuration name to aid with config management.")
    parser.add_argument("--colab", action='store_true', help="Optimize for Google Colab.")
    parser.add_argument("--nobreakmodel", action='store_true', help="Disables Breakmodel support completely.")
    parser.add_argument("--unblock", action='store_true', default=False, help="Unblocks the KoboldAI port to be accessible from other machines without optimizing for remote play (It is recommended to use --host instead)")
    parser.add_argument("--quiet", action='store_true', default=False, help="If present will suppress any story related text from showing on the console")
    parser.add_argument("--no_aria2", action='store_true', default=False, help="Prevents KoboldAI from using aria2 to download huggingface models more efficiently, in case aria2 is causing you issues")
    parser.add_argument("--lowmem", action='store_true', help="Extra Low Memory loading for the GPU, slower but memory does not peak to twice the usage")
    parser.add_argument("--savemodel", action='store_true', help="Saves the model to the models folder even if --colab is used (Allows you to save models to Google Drive)")
    #args: argparse.Namespace = None
    if "pytest" in sys.modules and override_args is None:
        args = parser.parse_args([])
        return
    if override_args is not None:
        import shlex
        args = parser.parse_args(shlex.split(override_args))
    elif(os.environ.get("KOBOLDAI_ARGS") is not None):
        import shlex
        args = parser.parse_args(shlex.split(os.environ["KOBOLDAI_ARGS"]))
    else:
        args = parser.parse_args()

    temp = [x for x in vars(args)]
    for arg in temp:
        if arg == "path":
            if "model_path" in os.environ:
                setattr(args, arg, os.environ["model_path"])
        else:
            if arg in os.environ:
                if isinstance(getattr(args, arg), bool):
                    if os.environ[arg].lower() == "true":
                        setattr(args, arg, True)
                    else:
                        setattr(args, arg, False)
                else:
                    setattr(args, arg, os.environ[arg])

   

    koboldai_vars.model = args.model;
    koboldai_vars.revision = args.revision

    if args.colab:
        args.remote = True;
        args.override_rename = True;
        args.override_delete = True;
        args.nobreakmodel = True;
        args.quiet = True;
        args.lowmem = True;
        args.noaimenu = True;

    if args.quiet:
        koboldai_vars.quiet = True

    if args.nobreakmodel:
        koboldai_vars.nobreakmodel = True;

    if args.remote:
        koboldai_vars.host = True;

    if args.ngrok:
        koboldai_vars.host = True;

    if args.localtunnel:
        koboldai_vars.host = True;

    if args.host:
        koboldai_vars.host = True;

    if args.cpu:
        koboldai_vars.use_colab_tpu = False

    koboldai_vars.smandelete = koboldai_vars.host == args.override_delete
    koboldai_vars.smanrename = koboldai_vars.host == args.override_rename

    koboldai_vars.aria2_port = args.aria2_port or 6799
    
    #Now let's look to see if we are going to force a load of a model from a user selected folder
    if(koboldai_vars.model == "selectfolder"):
        print("{0}Please choose the folder where pytorch_model.bin is located:{1}\n".format(colors.CYAN, colors.END))
        modpath = fileops.getdirpath(getcwd() + "/models", "Select Model Folder")
    
        if(modpath):
            # Save directory to vars
            koboldai_vars.model = "NeoCustom"
            koboldai_vars.custmodpth = modpath
    elif args.model:
        print("Welcome to KoboldAI!\nYou have selected the following Model:", koboldai_vars.model)
        if args.path:
            print("You have selected the following path for your Model :", args.path)
            koboldai_vars.custmodpth = args.path;
            koboldai_vars.colaburl = args.path + "/request"; # Lets just use the same parameter to keep it simple
#==================================================================#
# Load Model
#==================================================================# 

def tpumtjgetsofttokens():
    soft_tokens = None
    if(koboldai_vars.sp is None):
        global np
        if 'np' not in globals():
            import numpy as np
        tensor = np.zeros((1, tpu_mtj_backend.params.get("d_embed", tpu_mtj_backend.params["d_model"])), dtype=np.float32)
        rows = tensor.shape[0]
        padding_amount = tpu_mtj_backend.params["seq"] - (tpu_mtj_backend.params["seq"] % -tpu_mtj_backend.params["cores_per_replica"]) - rows
        tensor = np.pad(tensor, ((0, padding_amount), (0, 0)))
        tensor = tensor.reshape(
            tpu_mtj_backend.params["cores_per_replica"],
            -1,
            tpu_mtj_backend.params.get("d_embed", tpu_mtj_backend.params["d_model"]),
        )
        koboldai_vars.sp = tpu_mtj_backend.shard_xmap(tensor)
    soft_tokens = np.arange(
        tpu_mtj_backend.params["n_vocab"] + tpu_mtj_backend.params["n_vocab_padding"],
        tpu_mtj_backend.params["n_vocab"] + tpu_mtj_backend.params["n_vocab_padding"] + koboldai_vars.sp_length,
        dtype=np.uint32
    )
    return soft_tokens
 
def get_model_info(model, directory=""):
    # if the model is in the api list
    disk_blocks = 0
    key = False
    breakmodel = False
    gpu = False
    layer_count = None
    key_value = ""
    break_values = []
    url = False
    gpu_count = torch.cuda.device_count()
    gpu_names = []
    for i in range(gpu_count):
        gpu_names.append(torch.cuda.get_device_name(i))
    if model in [x[1] for x in model_menu['apilist']]:
        if path.exists("settings/{}.settings".format(model)):
            with open("settings/{}.settings".format(model), "r") as file:
                # Check if API key exists
                js = json.load(file)
                if("apikey" in js and js["apikey"] != ""):
                    # API key exists, grab it and close the file
                    key_value = js["apikey"]
                elif 'oaiapikey' in js and js['oaiapikey'] != "":
                    key_value = js["oaiapikey"]
                if model in ('GooseAI', 'OAI'): 
                    get_oai_models({'model': model, 'key': key_value})
        key = True
    elif model == 'ReadOnly':
        pass
    elif model == 'Colab':
        url = True
    elif not utils.HAS_ACCELERATE and not torch.cuda.is_available():
        pass
    else:
        layer_count = get_layer_count(model, directory=directory)
        if layer_count is None:
            breakmodel = False
        else:
            breakmodel = True
            if model in ["NeoCustom", "GPT2Custom"]:
                filename = os.path.basename(os.path.normpath(directory))
            else:
                filename = "settings/{}.breakmodel".format(model.replace("/", "_"))
            if path.exists(filename):
                with open(filename, "r") as file:
                    data = file.read().split("\n")[:2]
                    if len(data) < 2:
                        data.append("0")
                    break_values, disk_blocks = data
                    break_values = break_values.split(",")
            else:
                break_values = [layer_count]
            break_values = [int(x) for x in break_values]
            break_values += [0] * (gpu_count - len(break_values))
    emit('from_server', {'cmd': 'selected_model_info', 'key_value': key_value, 'key':key, 
                         'gpu':gpu, 'layer_count':layer_count, 'breakmodel':breakmodel, 
                         'disk_break_value': disk_blocks, 'accelerate': utils.HAS_ACCELERATE,
                         'break_values': break_values, 'gpu_count': gpu_count,
                         'url': url, 'gpu_names': gpu_names}, broadcast=True, room="UI_1")
    emit('selected_model_info', {'key_value': key_value, 'key':key, 
                         'gpu':gpu, 'layer_count':layer_count, 'breakmodel':breakmodel, 
                         'disk_break_value': disk_blocks, 'disk_break': utils.HAS_ACCELERATE,
                         'break_values': break_values, 'gpu_count': gpu_count,
                         'url': url, 'gpu_names': gpu_names}, broadcast=False, room="UI_2")
    
    

def get_layer_count(model, directory=""):
    if(model not in ["InferKit", "Colab", "OAI", "GooseAI" , "ReadOnly", "TPUMeshTransformerGPTJ"]):
        if(koboldai_vars.model == "GPT2Custom"):
            model_config = open(directory + "/config.json", "r")
        # Get the model_type from the config or assume a model type if it isn't present
        else:
            from transformers import AutoConfig
            if directory == "":
                model_config = AutoConfig.from_pretrained(model, cache_dir="cache")
            elif os.path.isdir(directory):
                model_config = AutoConfig.from_pretrained(directory, cache_dir="cache")
            else:
                assert "Selected Model directory doesn't exist"
        
        
        
        return utils.num_layers(model_config)
    else:
        return None

@socketio.on('OAI_Key_Update')
def get_oai_models(data):
    key = data['key']
    model = data['model']
    koboldai_vars.oaiapikey = key
    if model == 'OAI':
        url = "https://api.openai.com/v1/engines"
    elif model == 'GooseAI':
        url = "https://api.goose.ai/v1/engines"
    else:
        return
        
    # Get list of models from OAI
    print("{0}Retrieving engine list...{1}".format(colors.PURPLE, colors.END), end="")
    req = requests.get(
        url, 
        headers = {
            'Authorization': 'Bearer '+key
            }
        )
    if(req.status_code == 200):
        engines = req.json()["data"]
        try:
            engines = [[en["id"], "{} ({})".format(en['id'], "Ready" if en["ready"] == True else "Not Ready")] for en in engines]
        except:
            print(engines)
            raise
        
        online_model = ""
        changed=False
        
        #Save the key
        if not path.exists("settings"):
            # If the client settings file doesn't exist, create it
            # Write API key to file
            os.makedirs('settings', exist_ok=True)
        if path.exists("settings/{}.settings".format(model)):
            with open("settings/{}.settings".format(model), "r") as file:
                js = json.load(file)
                if 'online_model' in js:
                    online_model = js['online_model']
                if "apikey" in js:
                    if js['apikey'] != key:
                        changed=True
        if changed:
            with open("settings/{}.settings".format(model), "w") as file:
                js["apikey"] = key
                file.write(json.dumps(js, indent=3), room="UI_1")
            
        emit('from_server', {'cmd': 'oai_engines', 'data': engines, 'online_model': online_model}, broadcast=True, room="UI_1")
        emit('oai_engines', {'data': engines, 'online_model': online_model}, room="UI_2")
    else:
        # Something went wrong, print the message and quit since we can't initialize an engine
        print("{0}ERROR!{1}".format(colors.RED, colors.END), room="UI_1")
        print(req.json())
        emit('from_server', {'cmd': 'errmsg', 'data': req.json()})


# Function to patch transformers to use our soft prompt
def patch_causallm(model):
    from torch.nn import Embedding
    if(getattr(Embedding, "_koboldai_patch_causallm_model", None)):
        Embedding._koboldai_patch_causallm_model = model
        return model
    old_embedding_call = Embedding.__call__
    def new_embedding_call(self, input_ids, *args, **kwargs):
        if(Embedding._koboldai_patch_causallm_model.get_input_embeddings() is not self):
            return old_embedding_call(self, input_ids, *args, **kwargs)
        assert input_ids is not None
        if(koboldai_vars.sp is not None):
            shifted_input_ids = input_ids - model.config.vocab_size
        input_ids.clamp_(max=model.config.vocab_size-1)
        inputs_embeds = old_embedding_call(self, input_ids, *args, **kwargs)
        if(koboldai_vars.sp is not None):
            koboldai_vars.sp = koboldai_vars.sp.to(inputs_embeds.dtype).to(inputs_embeds.device)
            inputs_embeds = torch.where(
                (shifted_input_ids >= 0)[..., None],
                koboldai_vars.sp[shifted_input_ids.clamp(min=0)],
                inputs_embeds,
            )
        return inputs_embeds
    Embedding.__call__ = new_embedding_call
    Embedding._koboldai_patch_causallm_model = model
    return model

def patch_transformers_download():
    global transformers
    import copy, requests, tqdm, time
    class Send_to_socketio(object):
        def write(self, bar):
            bar = bar.replace("\r", "")
            try:
                emit('from_server', {'cmd': 'model_load_status', 'data': bar.replace(" ", "&nbsp;")}, broadcast=True, room="UI_1")
                eventlet.sleep(seconds=0)
            except:
                pass
    def http_get(
        url: str,
        temp_file: transformers.utils.hub.BinaryIO,
        proxies=None,
        resume_size=0,
        headers: transformers.utils.hub.Optional[transformers.utils.hub.Dict[str, str]] = None,
        file_name: transformers.utils.hub.Optional[str] = None,
    ):
        """
        Download remote file. Do not gobble up errors.
        """
        headers = copy.deepcopy(headers)
        if resume_size > 0:
            headers["Range"] = f"bytes={resume_size}-"
        r = requests.get(url, stream=True, proxies=proxies, headers=headers)
        transformers.utils.hub._raise_for_status(r)
        content_length = r.headers.get("Content-Length")
        total = resume_size + int(content_length) if content_length is not None else None
        # `tqdm` behavior is determined by `utils.logging.is_progress_bar_enabled()`
        # and can be set using `utils.logging.enable/disable_progress_bar()`
        koboldai_vars.total_download_chunks = total
        progress = tqdm.tqdm(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            total=total,
            initial=resume_size,
            desc=f"Downloading {file_name}" if file_name is not None else "Downloading",
            file=Send_to_socketio(),
        )
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                progress.update(len(chunk))
                koboldai_vars.downloaded_chunks += len(chunk)
                temp_file.write(chunk)
        progress.close()

    transformers.utils.hub.http_get = http_get

def patch_transformers():
    global transformers
    patch_transformers_download()
    old_from_pretrained = PreTrainedModel.from_pretrained.__func__
    @classmethod
    def new_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        koboldai_vars.fp32_model = False
        utils.num_shards = None
        utils.current_shard = 0
        utils.from_pretrained_model_name = pretrained_model_name_or_path
        utils.from_pretrained_index_filename = None
        utils.from_pretrained_kwargs = kwargs
        utils.bar = None
        if not args.no_aria2:
            utils.aria2_hook(pretrained_model_name_or_path, **kwargs)
        return old_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs)
    PreTrainedModel.from_pretrained = new_from_pretrained
    if(hasattr(modeling_utils, "get_checkpoint_shard_files")):
        old_get_checkpoint_shard_files = modeling_utils.get_checkpoint_shard_files
        def new_get_checkpoint_shard_files(pretrained_model_name_or_path, index_filename, *args, **kwargs):
            utils.num_shards = utils.get_num_shards(index_filename)
            utils.from_pretrained_index_filename = index_filename
            return old_get_checkpoint_shard_files(pretrained_model_name_or_path, index_filename, *args, **kwargs)
        modeling_utils.get_checkpoint_shard_files = new_get_checkpoint_shard_files
        
    # Some versions of transformers 4.17.0.dev0 are affected by
    # https://github.com/huggingface/transformers/issues/15736
    # This is a workaround for those versions of transformers.
    if(transformers_version == "4.17.0.dev0"):
        try:
            from transformers.models.xglm.modeling_xglm import XGLMSinusoidalPositionalEmbedding
        except ImportError:
            pass
        else:
            @torch.no_grad()
            def new_forward(self, input_ids: torch.Tensor = None, inputs_embeds: torch.Tensor = None, past_key_values_length: int = 0):
                bsz, seq_len = inputs_embeds.size()[:-1]
                input_shape = inputs_embeds.size()[:-1]
                sequence_length = input_shape[1]
                position_ids = torch.arange(
                    past_key_values_length + self.padding_idx + 1, past_key_values_length + sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
                ).unsqueeze(0).expand(input_shape).contiguous()
                max_pos = self.padding_idx + 1 + seq_len + past_key_values_length
                if max_pos > self.weights.size(0):
                    self.make_weights(max_pos + self.offset, self.embedding_dim, self.padding_idx)
                return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, -1).detach()
            XGLMSinusoidalPositionalEmbedding.forward = new_forward


    # Fix a bug in OPTForCausalLM where self.lm_head is the wrong size
    if(packaging.version.parse("4.19.0.dev0") <= packaging.version.parse(transformers_version) < packaging.version.parse("4.20.0")):
        try:
            from transformers import OPTForCausalLM, OPTModel
        except ImportError:
            pass
        else:
            # This is the same as the original __init__ but with
            # config.hidden_size
            # replaced with
            # config.word_embed_proj_dim
            def new_init(self, config):
                super(OPTForCausalLM, self).__init__(config)
                self.model = OPTModel(config)
                self.lm_head = torch.nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)
                self.post_init()
            OPTForCausalLM.__init__ = new_init


    # Patch transformers to use our custom logit warpers
    from transformers import LogitsProcessorList, LogitsWarper, LogitsProcessor, TopKLogitsWarper, TopPLogitsWarper, TemperatureLogitsWarper, RepetitionPenaltyLogitsProcessor
    from warpers import AdvancedRepetitionPenaltyLogitsProcessor, TailFreeLogitsWarper, TypicalLogitsWarper, TopALogitsWarper

    def dynamic_processor_wrap(cls, field_name, var_name, cond=None):
        old_call = cls.__call__
        def new_call(self, *args, **kwargs):
            if(not isinstance(field_name, str) and isinstance(field_name, Iterable)):
                conds = []
                for f, v in zip(field_name, var_name):
                    conds.append(getattr(koboldai_vars, v))
                    setattr(self, f, conds[-1])
            else:
                conds = getattr(koboldai_vars, var_name)
                setattr(self, field_name, conds)
            assert len(args) == 2
            if(cond is None or cond(conds)):
                return old_call(self, *args, **kwargs)
            return args[1]
        cls.__call__ = new_call
    dynamic_processor_wrap(AdvancedRepetitionPenaltyLogitsProcessor, ("penalty", "penalty_slope", "penalty_range"), ("rep_pen", "rep_pen_slope", "rep_pen_range"), cond=lambda x: x[0] != 1.0)
    dynamic_processor_wrap(TopKLogitsWarper, "top_k", "top_k", cond=lambda x: x > 0)
    dynamic_processor_wrap(TopALogitsWarper, "top_a", "top_a", cond=lambda x: x > 0.0)
    dynamic_processor_wrap(TopPLogitsWarper, "top_p", "top_p", cond=lambda x: x < 1.0)
    dynamic_processor_wrap(TailFreeLogitsWarper, "tfs", "tfs", cond=lambda x: x < 1.0)
    dynamic_processor_wrap(TypicalLogitsWarper, "typical", "typical", cond=lambda x: x < 1.0)
    dynamic_processor_wrap(TemperatureLogitsWarper, "temperature", "temp", cond=lambda x: x != 1.0)
    RepetitionPenaltyLogitsProcessor.__init__ = AdvancedRepetitionPenaltyLogitsProcessor.__init__
    RepetitionPenaltyLogitsProcessor.__call__ = AdvancedRepetitionPenaltyLogitsProcessor.__call__

    class LuaLogitsProcessor(LogitsProcessor):

        def __init__(self):
            pass

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
            assert scores.ndim == 2
            assert input_ids.ndim == 2
            self.regeneration_required = False
            self.halt = False

            scores_shape = scores.shape
            scores_list = scores.tolist()
            koboldai_vars.lua_koboldbridge.logits = koboldai_vars.lua_state.table()
            for r, row in enumerate(scores_list):
                koboldai_vars.lua_koboldbridge.logits[r+1] = koboldai_vars.lua_state.table(*row)
            koboldai_vars.lua_koboldbridge.vocab_size = scores_shape[-1]

            execute_genmod()

            scores = torch.tensor(
                tuple(tuple(row.values()) for row in koboldai_vars.lua_koboldbridge.logits.values()),
                device=scores.device,
                dtype=scores.dtype,
            )
            assert scores.shape == scores_shape

            return scores
    
    def new_get_logits_processor(*args, **kwargs) -> LogitsProcessorList:
        processors = new_get_logits_processor.old_get_logits_processor(*args, **kwargs)
        processors.insert(0, LuaLogitsProcessor())
        return processors
    new_get_logits_processor.old_get_logits_processor = transformers.generation_utils.GenerationMixin._get_logits_processor
    transformers.generation_utils.GenerationMixin._get_logits_processor = new_get_logits_processor

    class KoboldLogitsWarperList(LogitsProcessorList):
        def __init__(self, beams: int = 1, **kwargs):
            self.__warper_list: List[LogitsWarper] = []
            self.__warper_list.append(TopKLogitsWarper(top_k=1, min_tokens_to_keep=1 + (beams > 1)))
            self.__warper_list.append(TopALogitsWarper(top_a=0.5, min_tokens_to_keep=1 + (beams > 1)))
            self.__warper_list.append(TopPLogitsWarper(top_p=0.5, min_tokens_to_keep=1 + (beams > 1)))
            self.__warper_list.append(TailFreeLogitsWarper(tfs=0.5, min_tokens_to_keep=1 + (beams > 1)))
            self.__warper_list.append(TypicalLogitsWarper(typical=0.5, min_tokens_to_keep=1 + (beams > 1)))
            self.__warper_list.append(TemperatureLogitsWarper(temperature=0.5))

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, *args, **kwargs):
            for k in koboldai_vars.sampler_order:
                scores = self.__warper_list[k](input_ids, scores, *args, **kwargs)
            return scores

    def new_get_logits_warper(beams: int = 1,) -> LogitsProcessorList:
        return KoboldLogitsWarperList(beams=beams)
    
    def new_sample(self, *args, **kwargs):
        assert kwargs.pop("logits_warper", None) is not None
        kwargs["logits_warper"] = new_get_logits_warper(
            beams=1,
        )
        if(koboldai_vars.newlinemode == "s") or (koboldai_vars.newlinemode == "ns"):
            kwargs["eos_token_id"] = -1
            kwargs.setdefault("pad_token_id", 2)
        return new_sample.old_sample(self, *args, **kwargs)
    new_sample.old_sample = transformers.generation_utils.GenerationMixin.sample
    transformers.generation_utils.GenerationMixin.sample = new_sample


    # Allow bad words filter to ban <|endoftext|> token
    import transformers.generation_logits_process
    def new_init(self, bad_words_ids: List[List[int]], eos_token_id: int):
        return new_init.old_init(self, bad_words_ids, -1)
    new_init.old_init = transformers.generation_logits_process.NoBadWordsLogitsProcessor.__init__
    transformers.generation_logits_process.NoBadWordsLogitsProcessor.__init__ = new_init

    class TokenStreamer(StoppingCriteria):
        # A StoppingCriteria is used here because it seems to run after
        # everything has been evaluated score-wise.
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def __call__(
            self,
            input_ids: torch.LongTensor,
            scores: torch.FloatTensor,
            **kwargs,
        ) -> bool:
            if (not koboldai_vars.output_streaming):
                return False

            for batch, ids in enumerate(input_ids):
                tokenizer_text = utils.decodenewlines(tokenizer.decode(ids[-1]))
                koboldai_vars.actions.stream_token(tokenizer_text, batch=batch)
            return False


    # Sets up dynamic world info scanner
    class DynamicWorldInfoScanCriteria(StoppingCriteria):
        def __init__(
            self,
            tokenizer,
            excluded_world_info: List[Set],
        ):
            self.regeneration_required = False
            self.halt = False
            self.tokenizer = tokenizer
            self.excluded_world_info = excluded_world_info
        def __call__(
            self,
            input_ids: torch.LongTensor,
            scores: torch.FloatTensor,
            **kwargs,
        ) -> bool:
            koboldai_vars.generated_tkns += 1
            if(koboldai_vars.lua_koboldbridge.generated_cols and koboldai_vars.generated_tkns != koboldai_vars.lua_koboldbridge.generated_cols):
                raise RuntimeError(f"Inconsistency detected between KoboldAI Python and Lua backends ({koboldai_vars.generated_tkns} != {koboldai_vars.lua_koboldbridge.generated_cols})")
            if(koboldai_vars.abort or koboldai_vars.generated_tkns >= koboldai_vars.genamt):
                self.regeneration_required = False
                self.halt = False
                return True

            assert input_ids.ndim == 2
            assert len(self.excluded_world_info) == input_ids.shape[0]
            self.regeneration_required = koboldai_vars.lua_koboldbridge.regeneration_required
            self.halt = not koboldai_vars.lua_koboldbridge.generating
            koboldai_vars.lua_koboldbridge.regeneration_required = False

            for i in range(koboldai_vars.numseqs):
                koboldai_vars.lua_koboldbridge.generated[i+1][koboldai_vars.generated_tkns] = int(input_ids[i, -1].item())

            if(not koboldai_vars.dynamicscan):
                return self.regeneration_required or self.halt
            tail = input_ids[..., -koboldai_vars.generated_tkns:]
            for i, t in enumerate(tail):
                decoded = utils.decodenewlines(tokenizer.decode(t))
                _, found = checkworldinfo(decoded, force_use_txt=True, actions=koboldai_vars._actions)
                found -= self.excluded_world_info[i]
                if(len(found) != 0):
                    self.regeneration_required = True
                    break
            return self.regeneration_required or self.halt
    old_get_stopping_criteria = transformers.generation_utils.GenerationMixin._get_stopping_criteria
    def new_get_stopping_criteria(self, *args, **kwargs):
        stopping_criteria = old_get_stopping_criteria(self, *args, **kwargs)
        global tokenizer

        self.kai_scanner = DynamicWorldInfoScanCriteria(
            tokenizer=tokenizer,
            excluded_world_info=self.kai_scanner_excluded_world_info,
        )
        token_streamer = TokenStreamer(tokenizer=tokenizer)

        stopping_criteria.insert(0, self.kai_scanner)
        stopping_criteria.insert(0, token_streamer)
        return stopping_criteria
    transformers.generation_utils.GenerationMixin._get_stopping_criteria = new_get_stopping_criteria

def load_model(use_gpu=True, gpu_layers=None, disk_layers=None, initial_load=False, online_model=""):
    global model
    global generator
    global torch
    global model_config
    global GPT2TokenizerFast
    global tokenizer
    if not utils.HAS_ACCELERATE:
        disk_layers = None
    koboldai_vars.noai = False
    if not initial_load:
        set_aibusy(True)
        if koboldai_vars.model != 'ReadOnly':
            emit('from_server', {'cmd': 'model_load_status', 'data': "Loading {}".format(koboldai_vars.model)}, broadcast=True, room="UI_1")
            #Have to add a sleep so the server will send the emit for some reason
            time.sleep(0.1)
    if gpu_layers is not None:
        args.breakmodel_gpulayers = gpu_layers
    if disk_layers is not None:
        args.breakmodel_disklayers = int(disk_layers)
    
    #We need to wipe out the existing model and refresh the cuda cache
    model = None
    generator = None
    model_config = None
    for tensor in gc.get_objects():
        try:
            if torch.is_tensor(tensor):
                with torch.no_grad():
                    tensor.set_(torch.tensor((), device=tensor.device, dtype=tensor.dtype))
        except:
            pass
    gc.collect()
    try:
        with torch.no_grad():
            torch.cuda.empty_cache()
    except:
        pass
        
    #Reload our badwords
    koboldai_vars.badwordsids = koboldai_settings.badwordsids_default
    
    #Let's set the GooseAI or OpenAI server URLs if that's applicable
    if online_model != "":
        if path.exists("settings/{}.settings".format(koboldai_vars.model)):
            changed=False
            with open("settings/{}.settings".format(koboldai_vars.model), "r") as file:
                # Check if API key exists
                js = json.load(file)
                if 'online_model' in js:
                    if js['online_model'] != online_model:
                        changed=True
                        js['online_model'] = online_model
                else:
                    changed=True
                    js['online_model'] = online_model
            if changed:
                with open("settings/{}.settings".format(koboldai_vars.model), "w") as file:
                    file.write(json.dumps(js, indent=3))
        # Swap OAI Server if GooseAI was selected
        if(koboldai_vars.model == "GooseAI"):
            koboldai_vars.oaiengines = "https://api.goose.ai/v1/engines"
            koboldai_vars.model = "OAI"
            args.configname = "GooseAI" + "/" + online_model
        else:
            args.configname = koboldai_vars.model + "/" + online_model
        koboldai_vars.oaiurl = koboldai_vars.oaiengines + "/{0}/completions".format(online_model)
    
    
    # If transformers model was selected & GPU available, ask to use CPU or GPU
    if(koboldai_vars.model not in ["InferKit", "Colab", "OAI", "GooseAI" , "ReadOnly", "TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX"]):
        koboldai_vars.allowsp = True
        # Test for GPU support
        
        # Make model path the same as the model name to make this consistent with the other loading method if it isn't a known model type
        # This code is not just a workaround for below, it is also used to make the behavior consistent with other loading methods - Henk717
        if(not koboldai_vars.model in ["NeoCustom", "GPT2Custom"]):
            koboldai_vars.custmodpth = koboldai_vars.model
        elif(koboldai_vars.model == "NeoCustom"):
            koboldai_vars.model = os.path.basename(os.path.normpath(koboldai_vars.custmodpth))

        # Get the model_type from the config or assume a model type if it isn't present
        from transformers import AutoConfig
        if(os.path.isdir(koboldai_vars.custmodpth.replace('/', '_'))):
            try:
                model_config = AutoConfig.from_pretrained(koboldai_vars.custmodpth.replace('/', '_'), revision=koboldai_vars.revision, cache_dir="cache")
                koboldai_vars.model_type = model_config.model_type
            except ValueError as e:
                koboldai_vars.model_type = "not_found"
        elif(os.path.isdir("models/{}".format(koboldai_vars.custmodpth.replace('/', '_')))):
            try:
                model_config = AutoConfig.from_pretrained("models/{}".format(koboldai_vars.custmodpth.replace('/', '_')), revision=koboldai_vars.revision, cache_dir="cache")
                koboldai_vars.model_type = model_config.model_type
            except ValueError as e:
                koboldai_vars.model_type = "not_found"
        else:
            try:
                model_config = AutoConfig.from_pretrained(koboldai_vars.custmodpth, revision=koboldai_vars.revision, cache_dir="cache")
                koboldai_vars.model_type = model_config.model_type
            except ValueError as e:
                koboldai_vars.model_type = "not_found"
        if(koboldai_vars.model_type == "not_found" and koboldai_vars.model == "NeoCustom"):
            koboldai_vars.model_type = "gpt_neo"
        elif(koboldai_vars.model_type == "not_found" and koboldai_vars.model == "GPT2Custom"):
            koboldai_vars.model_type = "gpt2"
        elif(koboldai_vars.model_type == "not_found"):
            print("WARNING: No model type detected, assuming Neo (If this is a GPT2 model use the other menu option or --model GPT2Custom)")
            koboldai_vars.model_type = "gpt_neo"

    if(not koboldai_vars.use_colab_tpu and koboldai_vars.model not in ["InferKit", "Colab", "OAI", "GooseAI" , "ReadOnly", "TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX"]):
        loadmodelsettings()
        loadsettings()
        print("{0}Looking for GPU support...{1}".format(colors.PURPLE, colors.END), end="")
        koboldai_vars.hascuda = torch.cuda.is_available()
        koboldai_vars.bmsupported = (utils.HAS_ACCELERATE or koboldai_vars.model_type in ("gpt_neo", "gptj", "xglm", "opt")) and not koboldai_vars.nobreakmodel
        if(args.breakmodel is not None and args.breakmodel):
            print("WARNING: --breakmodel is no longer supported. Breakmodel mode is now automatically enabled when --breakmodel_gpulayers is used (see --help for details).", file=sys.stderr)
        if(args.breakmodel_layers is not None):
            print("WARNING: --breakmodel_layers is deprecated. Use --breakmodel_gpulayers instead (see --help for details).", file=sys.stderr)
        if(args.model and koboldai_vars.bmsupported and not args.breakmodel_gpulayers and not args.breakmodel_layers and (not utils.HAS_ACCELERATE or not args.breakmodel_disklayers)):
            print("WARNING: Model launched without the --breakmodel_gpulayers argument, defaulting to GPU only mode.", file=sys.stderr)
            koboldai_vars.bmsupported = False
        if(not koboldai_vars.bmsupported and (args.breakmodel_gpulayers is not None or args.breakmodel_layers is not None or args.breakmodel_disklayers is not None)):
            print("WARNING: This model does not support hybrid generation. --breakmodel_gpulayers will be ignored.", file=sys.stderr)
        if(koboldai_vars.hascuda):
            print("{0}FOUND!{1}".format(colors.GREEN, colors.END))
        else:
            print("{0}NOT FOUND!{1}".format(colors.YELLOW, colors.END))
        
        if args.model:
            if(koboldai_vars.hascuda):
                genselected = True
                koboldai_vars.usegpu = True
                koboldai_vars.breakmodel = utils.HAS_ACCELERATE
            if(koboldai_vars.bmsupported):
                koboldai_vars.usegpu = False
                koboldai_vars.breakmodel = True
            if(args.cpu):
                koboldai_vars.usegpu = False
                koboldai_vars.breakmodel = utils.HAS_ACCELERATE
        elif(koboldai_vars.hascuda):    
            if(koboldai_vars.bmsupported):
                genselected = True
                koboldai_vars.usegpu = False
                koboldai_vars.breakmodel = True
            else:
                genselected = False
        else:
            genselected = False

        if(koboldai_vars.hascuda):
            if(use_gpu):
                if(koboldai_vars.bmsupported):
                    koboldai_vars.breakmodel = True
                    koboldai_vars.usegpu = False
                    genselected = True
                else:
                    koboldai_vars.breakmodel = False
                    koboldai_vars.usegpu = True
                    genselected = True
            else:
                koboldai_vars.breakmodel = utils.HAS_ACCELERATE
                koboldai_vars.usegpu = False
                genselected = True

    # Ask for API key if InferKit was selected
    if(koboldai_vars.model == "InferKit"):
        koboldai_vars.apikey = koboldai_vars.oaiapikey
                    
    # Swap OAI Server if GooseAI was selected
    if(koboldai_vars.model == "GooseAI"):
        koboldai_vars.oaiengines = "https://api.goose.ai/v1/engines"
        koboldai_vars.model = "OAI"
        args.configname = "GooseAI"

    # Ask for API key if OpenAI was selected
    if(koboldai_vars.model == "OAI"):
        if not args.configname:
            args.configname = "OAI"
        
    if(koboldai_vars.model == "ReadOnly"):
        koboldai_vars.noai = True

    # Start transformers and create pipeline
    if(not koboldai_vars.use_colab_tpu and koboldai_vars.model not in ["InferKit", "Colab", "OAI", "GooseAI" , "ReadOnly", "TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX"]):
        if(not koboldai_vars.noai):
            print("{0}Initializing transformers, please wait...{1}".format(colors.PURPLE, colors.END))
            for m in ("GPTJModel", "XGLMModel"):
                try:
                    globals()[m] = getattr(__import__("transformers"), m)
                except:
                    pass

            # Lazy loader
            import torch_lazy_loader
            def get_lazy_load_callback(n_layers, convert_to_float16=True):
                if not koboldai_vars.lazy_load:
                    return

                from tqdm.auto import tqdm

                global breakmodel
                import breakmodel

                if utils.HAS_ACCELERATE:
                    import accelerate.utils

                if args.breakmodel_disklayers is not None:
                    breakmodel.disk_blocks = args.breakmodel_disklayers

                disk_blocks = breakmodel.disk_blocks
                gpu_blocks = breakmodel.gpu_blocks
                ram_blocks = ram_blocks = n_layers - sum(gpu_blocks)
                cumulative_gpu_blocks = tuple(itertools.accumulate(gpu_blocks))

                def lazy_load_callback(model_dict: Dict[str, Union[torch_lazy_loader.LazyTensor, torch.Tensor]], f, **_):
                    if lazy_load_callback.nested:
                        return
                    lazy_load_callback.nested = True

                    device_map: Dict[str, Union[str, int]] = {}

                    @functools.lru_cache(maxsize=None)
                    def get_original_key(key):
                        return max((original_key for original_key in utils.module_names if original_key.endswith(key)), key=len)

                    for key, value in model_dict.items():
                        original_key = get_original_key(key)
                        if isinstance(value, torch_lazy_loader.LazyTensor) and not any(original_key.startswith(n) for n in utils.layers_module_names):
                            device_map[key] = koboldai_vars.gpu_device if koboldai_vars.hascuda and koboldai_vars.usegpu else "cpu" if not koboldai_vars.hascuda or not koboldai_vars.breakmodel else breakmodel.primary_device
                        else:
                            layer = int(max((n for n in utils.layers_module_names if original_key.startswith(n)), key=len).rsplit(".", 1)[1])
                            device = koboldai_vars.gpu_device if koboldai_vars.hascuda and koboldai_vars.usegpu else "disk" if layer < disk_blocks and layer < ram_blocks else "cpu" if not koboldai_vars.hascuda or not koboldai_vars.breakmodel else "shared" if layer < ram_blocks else bisect.bisect_right(cumulative_gpu_blocks, layer - ram_blocks)
                            device_map[key] = device

                    if utils.num_shards is None or utils.current_shard == 0:
                        utils.offload_index = {}
                        if utils.HAS_ACCELERATE:
                            if os.path.isdir("accelerate-disk-cache"):
                                # Delete all of the files in the disk cache folder without deleting the folder itself to allow people to create symbolic links for this folder
                                # (the folder doesn't contain any subfolders so os.remove will do just fine)
                                for filename in os.listdir("accelerate-disk-cache"):
                                    try:
                                        os.remove(os.path.join("accelerate-disk-cache", filename))
                                    except OSError:
                                        pass
                            os.makedirs("accelerate-disk-cache", exist_ok=True)
                        if utils.num_shards is not None:
                            num_tensors = len(utils.get_sharded_checkpoint_num_tensors(utils.from_pretrained_model_name, utils.from_pretrained_index_filename, **utils.from_pretrained_kwargs))
                        else:
                            num_tensors = len(device_map)
                        print(flush=True)
                        koboldai_vars.total_layers = num_tensors
                        koboldai_vars.loaded_layers = 0
                        utils.bar = tqdm(total=num_tensors, desc="Loading model tensors", file=Send_to_socketio())

                    with zipfile.ZipFile(f, "r") as z:
                        try:
                            last_storage_key = None
                            f = None
                            current_offset = 0
                            able_to_pin_layers = True
                            if utils.num_shards is not None:
                                utils.current_shard += 1
                            for key in sorted(device_map.keys(), key=lambda k: (model_dict[k].key, model_dict[k].seek_offset)):
                                storage_key = model_dict[key].key
                                if storage_key != last_storage_key or model_dict[key].seek_offset < current_offset:
                                    last_storage_key = storage_key
                                    if isinstance(f, zipfile.ZipExtFile):
                                        f.close()
                                    f = z.open(f"archive/data/{storage_key}")
                                    current_offset = 0
                                if current_offset != model_dict[key].seek_offset:
                                    f.read(model_dict[key].seek_offset - current_offset)
                                    current_offset = model_dict[key].seek_offset
                                device = device_map[key]
                                size = functools.reduce(lambda x, y: x * y, model_dict[key].shape, 1)
                                dtype = model_dict[key].dtype
                                nbytes = size if dtype is torch.bool else size * ((torch.finfo if dtype.is_floating_point else torch.iinfo)(dtype).bits >> 3)
                                #print(f"Transferring <{key}>  to  {f'({device.upper()})' if isinstance(device, str) else '[device ' + str(device) + ']'} ... ", end="", flush=True)
                                model_dict[key] = model_dict[key].materialize(f, map_location="cpu")
                                if model_dict[key].dtype is torch.float32:
                                    koboldai_vars.fp32_model = True
                                if convert_to_float16 and breakmodel.primary_device != "cpu" and koboldai_vars.hascuda and (koboldai_vars.breakmodel or koboldai_vars.usegpu) and model_dict[key].dtype is torch.float32:
                                    model_dict[key] = model_dict[key].to(torch.float16)
                                if breakmodel.primary_device == "cpu" or (not koboldai_vars.usegpu and not koboldai_vars.breakmodel and model_dict[key].dtype is torch.float16):
                                    model_dict[key] = model_dict[key].to(torch.float32)
                                if device == "shared":
                                    model_dict[key] = model_dict[key].to("cpu").detach_()
                                    if able_to_pin_layers and utils.HAS_ACCELERATE:
                                        try:
                                            model_dict[key] = model_dict[key].pin_memory()
                                        except:
                                            able_to_pin_layers = False
                                elif device == "disk":
                                    accelerate.utils.offload_weight(model_dict[key], get_original_key(key), "accelerate-disk-cache", index=utils.offload_index)
                                    model_dict[key] = model_dict[key].to("meta")
                                else:
                                    model_dict[key] = model_dict[key].to(device)
                                #print("OK", flush=True)
                                current_offset += nbytes
                                utils.bar.update(1)
                                koboldai_vars.loaded_layers += 1
                        finally:
                            if utils.num_shards is None or utils.current_shard >= utils.num_shards:
                                if utils.offload_index:
                                    for name, tensor in utils.named_buffers:
                                        if name not in utils.offload_index:
                                            accelerate.utils.offload_weight(tensor, name, "accelerate-disk-cache", index=utils.offload_index)
                                    accelerate.utils.save_offload_index(utils.offload_index, "accelerate-disk-cache")
                                utils.bar.close()
                                utils.bar = None
                            lazy_load_callback.nested = False
                            if isinstance(f, zipfile.ZipExtFile):
                                f.close()

                lazy_load_callback.nested = False
                return lazy_load_callback


            def get_hidden_size_from_model(model):
                try:
                    return int(model.model.decoder.project_in.in_features)
                except:
                    try:
                        return int(model.model.decoder.embed_tokens.out_features)
                    except:
                        try:
                            return int(model.transformer.hidden_size)
                        except:
                            try:
                                return int(model.transformer.embed_dim)
                            except:
                                return int(model.lm_head.in_features)
            
            def maybe_low_cpu_mem_usage() -> Dict[str, Any]:
                if(packaging.version.parse(transformers_version) < packaging.version.parse("4.11.0")):
                    print(f"\nWARNING:  Please upgrade to transformers 4.11.0 for lower RAM usage.  You have transformers {transformers_version}.", file=sys.stderr)
                    return {}
                return {"low_cpu_mem_usage": True}
            
            @contextlib.contextmanager
            def maybe_use_float16(always_use=False):
                if(always_use or (koboldai_vars.hascuda and args.lowmem and (koboldai_vars.usegpu or koboldai_vars.breakmodel))):
                    original_dtype = torch.get_default_dtype()
                    torch.set_default_dtype(torch.float16)
                    yield True
                    torch.set_default_dtype(original_dtype)
                else:
                    yield False

            # If custom GPT2 model was chosen
            if(koboldai_vars.model == "GPT2Custom"):
                koboldai_vars.lazy_load = False
                model_config = open(koboldai_vars.custmodpth + "/config.json", "r")
                js   = json.load(model_config)
                with(maybe_use_float16()):
                    try:
                        model = GPT2LMHeadModel.from_pretrained(koboldai_vars.custmodpth, revision=koboldai_vars.revision, cache_dir="cache")
                    except Exception as e:
                        if("out of memory" in traceback.format_exc().lower()):
                            raise RuntimeError("One of your GPUs ran out of memory when KoboldAI tried to load your model.")
                        raise e
                tokenizer = GPT2TokenizerFast.from_pretrained(koboldai_vars.custmodpth, revision=koboldai_vars.revision, cache_dir="cache")
                koboldai_vars.modeldim = get_hidden_size_from_model(model)
                # Is CUDA available? If so, use GPU, otherwise fall back to CPU
                if(koboldai_vars.hascuda and koboldai_vars.usegpu):
                    model = model.half().to(koboldai_vars.gpu_device)
                    generator = model.generate
                else:
                    model = model.to('cpu').float()
                    generator = model.generate
                patch_causallm(model)
            # Use the Generic implementation
            else:
                lowmem = maybe_low_cpu_mem_usage()
                # We must disable low_cpu_mem_usage (by setting lowmem to {}) if
                # using a GPT-2 model because GPT-2 is not compatible with this
                # feature yet
                if(koboldai_vars.model_type == "gpt2"):
                    lowmem = {}
                    koboldai_vars.lazy_load = False  # Also, lazy loader doesn't support GPT-2 models
                
                # If we're using torch_lazy_loader, we need to get breakmodel config
                # early so that it knows where to load the individual model tensors
                if(utils.HAS_ACCELERATE or koboldai_vars.lazy_load and koboldai_vars.hascuda and koboldai_vars.breakmodel):
                    device_config(model_config)

                # Download model from Huggingface if it does not exist, otherwise load locally
                
                #If we specify a model and it's in the root directory, we need to move it to the models directory (legacy folder structure to new)
                if os.path.isdir(koboldai_vars.model.replace('/', '_')):
                    import shutil
                    shutil.move(koboldai_vars.model.replace('/', '_'), "models/{}".format(koboldai_vars.model.replace('/', '_')))
                print("\n", flush=True)
                if(koboldai_vars.lazy_load):  # If we're using lazy loader, we need to figure out what the model's hidden layers are called
                    with torch_lazy_loader.use_lazy_torch_load(dematerialized_modules=True, use_accelerate_init_empty_weights=True):
                        try:
                            metamodel = AutoModelForCausalLM.from_config(model_config)
                        except Exception as e:
                            metamodel = GPTNeoForCausalLM.from_config(model_config)
                        utils.layers_module_names = utils.get_layers_module_names(metamodel)
                        utils.module_names = list(metamodel.state_dict().keys())
                        utils.named_buffers = list(metamodel.named_buffers(recurse=True))
                with maybe_use_float16(), torch_lazy_loader.use_lazy_torch_load(enable=koboldai_vars.lazy_load, callback=get_lazy_load_callback(utils.num_layers(model_config)) if koboldai_vars.lazy_load else None, dematerialized_modules=True):
                    if(koboldai_vars.lazy_load):  # torch_lazy_loader.py and low_cpu_mem_usage can't be used at the same time
                        lowmem = {}
                    if(os.path.isdir(koboldai_vars.custmodpth)):
                        try:
                            tokenizer = AutoTokenizer.from_pretrained(koboldai_vars.custmodpth, revision=koboldai_vars.revision, cache_dir="cache")
                        except Exception as e:
                            pass
                        try:
                            tokenizer = AutoTokenizer.from_pretrained(koboldai_vars.custmodpth, revision=koboldai_vars.revision, cache_dir="cache", use_fast=False)
                        except Exception as e:
                            try:
                                tokenizer = GPT2TokenizerFast.from_pretrained(koboldai_vars.custmodpth, revision=koboldai_vars.revision, cache_dir="cache")
                            except Exception as e:
                                tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", revision=koboldai_vars.revision, cache_dir="cache")
                        try:
                            model     = AutoModelForCausalLM.from_pretrained(koboldai_vars.custmodpth, revision=koboldai_vars.revision, cache_dir="cache", **lowmem)
                        except Exception as e:
                            if("out of memory" in traceback.format_exc().lower()):
                                raise RuntimeError("One of your GPUs ran out of memory when KoboldAI tried to load your model.")
                            model     = GPTNeoForCausalLM.from_pretrained(koboldai_vars.custmodpth, revision=koboldai_vars.revision, cache_dir="cache", **lowmem)
                    elif(os.path.isdir("models/{}".format(koboldai_vars.model.replace('/', '_')))):
                        try:
                            tokenizer = AutoTokenizer.from_pretrained("models/{}".format(koboldai_vars.model.replace('/', '_')), revision=koboldai_vars.revision, cache_dir="cache")
                        except Exception as e:
                            pass
                        try:
                            tokenizer = AutoTokenizer.from_pretrained("models/{}".format(koboldai_vars.model.replace('/', '_')), revision=koboldai_vars.revision, cache_dir="cache", use_fast=False)
                        except Exception as e:
                            try:
                                tokenizer = GPT2TokenizerFast.from_pretrained("models/{}".format(koboldai_vars.model.replace('/', '_')), revision=koboldai_vars.revision, cache_dir="cache")
                            except Exception as e:
                                tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", revision=koboldai_vars.revision, cache_dir="cache")
                        try:
                            model     = AutoModelForCausalLM.from_pretrained("models/{}".format(koboldai_vars.model.replace('/', '_')), revision=koboldai_vars.revision, cache_dir="cache", **lowmem)
                        except Exception as e:
                            if("out of memory" in traceback.format_exc().lower()):
                                raise RuntimeError("One of your GPUs ran out of memory when KoboldAI tried to load your model.")
                            model     = GPTNeoForCausalLM.from_pretrained("models/{}".format(koboldai_vars.model.replace('/', '_')), revision=koboldai_vars.revision, cache_dir="cache", **lowmem)
                    else:
                        old_rebuild_tensor = torch._utils._rebuild_tensor
                        def new_rebuild_tensor(storage: Union[torch_lazy_loader.LazyTensor, torch.Storage], storage_offset, shape, stride):
                            if(not isinstance(storage, torch_lazy_loader.LazyTensor)):
                                dtype = storage.dtype
                            else:
                                dtype = storage.storage_type.dtype
                                if(not isinstance(dtype, torch.dtype)):
                                    dtype = storage.storage_type(0).dtype
                            if(dtype is torch.float32 and len(shape) >= 2):
                                koboldai_vars.fp32_model = True
                            return old_rebuild_tensor(storage, storage_offset, shape, stride)
                        torch._utils._rebuild_tensor = new_rebuild_tensor

                        try:
                            tokenizer = AutoTokenizer.from_pretrained(koboldai_vars.model, revision=koboldai_vars.revision, cache_dir="cache")
                        except Exception as e:
                            pass
                        try:
                            tokenizer = AutoTokenizer.from_pretrained(koboldai_vars.model, revision=koboldai_vars.revision, cache_dir="cache", use_fast=False)
                        except Exception as e:
                            try:
                                tokenizer = GPT2TokenizerFast.from_pretrained(koboldai_vars.model, revision=koboldai_vars.revision, cache_dir="cache")
                            except Exception as e:
                                tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", revision=koboldai_vars.revision, cache_dir="cache")
                        try:
                            model     = AutoModelForCausalLM.from_pretrained(koboldai_vars.model, revision=koboldai_vars.revision, cache_dir="cache", **lowmem)
                        except Exception as e:
                            if("out of memory" in traceback.format_exc().lower()):
                                raise RuntimeError("One of your GPUs ran out of memory when KoboldAI tried to load your model.")
                            model     = GPTNeoForCausalLM.from_pretrained(koboldai_vars.model, revision=koboldai_vars.revision, cache_dir="cache", **lowmem)

                        torch._utils._rebuild_tensor = old_rebuild_tensor

                        if not args.colab or args.savemodel:
                            import shutil
                            tokenizer.save_pretrained("models/{}".format(koboldai_vars.model.replace('/', '_')))
                            if(koboldai_vars.fp32_model):  # Use save_pretrained to convert fp32 models to fp16
                                model = model.half()
                                model.save_pretrained("models/{}".format(koboldai_vars.model.replace('/', '_')), max_shard_size="500MiB")
                            else:  # For fp16 models, we can just copy the model files directly
                                import transformers.configuration_utils
                                import transformers.modeling_utils
                                import transformers.file_utils
                                # Save the config.json
                                shutil.move(transformers.file_utils.get_from_cache(transformers.file_utils.hf_bucket_url(koboldai_vars.model, transformers.configuration_utils.CONFIG_NAME, revision=koboldai_vars.revision), cache_dir="cache", local_files_only=True), os.path.join("models/{}".format(koboldai_vars.model.replace('/', '_')), transformers.configuration_utils.CONFIG_NAME))
                                if(utils.num_shards is None):
                                    # Save the pytorch_model.bin of an unsharded model
                                    shutil.move(transformers.file_utils.get_from_cache(transformers.file_utils.hf_bucket_url(koboldai_vars.model, transformers.modeling_utils.WEIGHTS_NAME, revision=koboldai_vars.revision), cache_dir="cache", local_files_only=True), os.path.join("models/{}".format(koboldai_vars.model.replace('/', '_')), transformers.modeling_utils.WEIGHTS_NAME))
                                else:
                                    with open(utils.from_pretrained_index_filename) as f:
                                        map_data = json.load(f)
                                    filenames = set(map_data["weight_map"].values())
                                    # Save the pytorch_model.bin.index.json of a sharded model
                                    shutil.move(utils.from_pretrained_index_filename, os.path.join("models/{}".format(koboldai_vars.model.replace('/', '_')), transformers.modeling_utils.WEIGHTS_INDEX_NAME))
                                    # Then save the pytorch_model-#####-of-#####.bin files
                                    for filename in filenames:
                                        shutil.move(transformers.file_utils.get_from_cache(transformers.file_utils.hf_bucket_url(koboldai_vars.model, filename, revision=koboldai_vars.revision), cache_dir="cache", local_files_only=True), os.path.join("models/{}".format(koboldai_vars.model.replace('/', '_')), filename))
                            shutil.rmtree("cache/")

                if(koboldai_vars.badwordsids is koboldai_settings.badwordsids_default and koboldai_vars.model_type not in ("gpt2", "gpt_neo", "gptj")):
                    koboldai_vars.badwordsids = [[v] for k, v in tokenizer.get_vocab().items() if any(c in str(k) for c in "<>[]") if koboldai_vars.newlinemode != "s" or str(k) != "</s>"]

                patch_causallm(model)

                if(koboldai_vars.hascuda):
                    if(koboldai_vars.usegpu):
                        koboldai_vars.modeldim = get_hidden_size_from_model(model)
                        model = model.half().to(koboldai_vars.gpu_device)
                        generator = model.generate
                    elif(koboldai_vars.breakmodel):  # Use both RAM and VRAM (breakmodel)
                        koboldai_vars.modeldim = get_hidden_size_from_model(model)
                        if(not koboldai_vars.lazy_load):
                            device_config(model.config)
                        move_model_to_devices(model)
                    elif(utils.HAS_ACCELERATE and __import__("breakmodel").disk_blocks > 0):
                        move_model_to_devices(model)
                        koboldai_vars.modeldim = get_hidden_size_from_model(model)
                        generator = model.generate
                    else:
                        model = model.to('cpu').float()
                        koboldai_vars.modeldim = get_hidden_size_from_model(model)
                        generator = model.generate
                elif(utils.HAS_ACCELERATE and __import__("breakmodel").disk_blocks > 0):
                    move_model_to_devices(model)
                    koboldai_vars.modeldim = get_hidden_size_from_model(model)
                    generator = model.generate
                else:
                    model.to('cpu').float()
                    koboldai_vars.modeldim = get_hidden_size_from_model(model)
                    generator = model.generate
            
            # Suppress Author's Note by flagging square brackets (Old implementation)
            #vocab         = tokenizer.get_vocab()
            #vocab_keys    = vocab.keys()
            #vars.badwords = gettokenids("[")
            #for key in vars.badwords:
            #    koboldai_vars.badwordsids.append([vocab[key]])
            
            print("{0}OK! {1} pipeline created!{2}".format(colors.GREEN, koboldai_vars.model, colors.END))
        
        else:
            from transformers import GPT2TokenizerFast
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", revision=koboldai_vars.revision, cache_dir="cache")
    else:
        from transformers import PreTrainedModel
        from transformers import modeling_utils
        old_from_pretrained = PreTrainedModel.from_pretrained.__func__
        @classmethod
        def new_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
            koboldai_vars.fp32_model = False
            utils.num_shards = None
            utils.current_shard = 0
            utils.from_pretrained_model_name = pretrained_model_name_or_path
            utils.from_pretrained_index_filename = None
            utils.from_pretrained_kwargs = kwargs
            utils.bar = None
            if not args.no_aria2:
                utils.aria2_hook(pretrained_model_name_or_path, **kwargs)
            return old_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs)
        PreTrainedModel.from_pretrained = new_from_pretrained
        if(hasattr(modeling_utils, "get_checkpoint_shard_files")):
            old_get_checkpoint_shard_files = modeling_utils.get_checkpoint_shard_files
            def new_get_checkpoint_shard_files(pretrained_model_name_or_path, index_filename, *args, **kwargs):
                utils.num_shards = utils.get_num_shards(index_filename)
                utils.from_pretrained_index_filename = index_filename
                return old_get_checkpoint_shard_files(pretrained_model_name_or_path, index_filename, *args, **kwargs)
            modeling_utils.get_checkpoint_shard_files = new_get_checkpoint_shard_files


        def tpumtjgenerate_warper_callback(scores) -> "np.array":
            scores_shape = scores.shape
            scores_list = scores.tolist()
            koboldai_vars.lua_koboldbridge.logits = koboldai_vars.lua_state.table()
            for r, row in enumerate(scores_list):
                koboldai_vars.lua_koboldbridge.logits[r+1] = koboldai_vars.lua_state.table(*row)
            koboldai_vars.lua_koboldbridge.vocab_size = scores_shape[-1]

            execute_genmod()

            scores = np.array(
                tuple(tuple(row.values()) for row in koboldai_vars.lua_koboldbridge.logits.values()),
                dtype=scores.dtype,
            )
            assert scores.shape == scores_shape

            return scores
        
        def tpumtjgenerate_stopping_callback(generated, n_generated, excluded_world_info) -> Tuple[List[set], bool, bool]:
            koboldai_vars.generated_tkns += 1

            assert len(excluded_world_info) == len(generated)
            regeneration_required = koboldai_vars.lua_koboldbridge.regeneration_required
            halt = koboldai_vars.abort or not koboldai_vars.lua_koboldbridge.generating or koboldai_vars.generated_tkns >= koboldai_vars.genamt
            koboldai_vars.lua_koboldbridge.regeneration_required = False

            global past

            for i in range(koboldai_vars.numseqs):
                koboldai_vars.lua_koboldbridge.generated[i+1][koboldai_vars.generated_tkns] = int(generated[i, tpu_mtj_backend.params["seq"] + n_generated - 1].item())

            if(not koboldai_vars.dynamicscan or halt):
                return excluded_world_info, regeneration_required, halt

            for i, t in enumerate(generated):
                decoded = utils.decodenewlines(tokenizer.decode(past[i])) + utils.decodenewlines(tokenizer.decode(t[tpu_mtj_backend.params["seq"] : tpu_mtj_backend.params["seq"] + n_generated]))
                _, found = checkworldinfo(decoded, force_use_txt=True, actions=koboldai_vars._actions)
                found -= excluded_world_info[i]
                if(len(found) != 0):
                    regeneration_required = True
                    break
            return excluded_world_info, regeneration_required, halt

        def tpumtjgenerate_compiling_callback() -> None:
            print(colors.GREEN + "TPU backend compilation triggered" + colors.END)
            koboldai_vars.compiling = True

        def tpumtjgenerate_stopped_compiling_callback() -> None:
            koboldai_vars.compiling = False
        
        def tpumtjgenerate_settings_callback() -> dict:
            return {
                "sampler_order": koboldai_vars.sampler_order,
                "top_p": float(koboldai_vars.top_p),
                "temp": float(koboldai_vars.temp),
                "top_k": int(koboldai_vars.top_k),
                "tfs": float(koboldai_vars.tfs),
                "typical": float(koboldai_vars.typical),
                "top_a": float(koboldai_vars.top_a),
                "repetition_penalty": float(koboldai_vars.rep_pen),
                "rpslope": float(koboldai_vars.rep_pen_slope),
                "rprange": int(koboldai_vars.rep_pen_range),
            }

        # If we're running Colab or OAI, we still need a tokenizer.
        if(koboldai_vars.model == "Colab"):
            from transformers import GPT2TokenizerFast
            tokenizer = GPT2TokenizerFast.from_pretrained("EleutherAI/gpt-neo-2.7B", revision=koboldai_vars.revision, cache_dir="cache")
            loadsettings()
        elif(koboldai_vars.model == "OAI"):
            from transformers import GPT2TokenizerFast
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", revision=koboldai_vars.revision, cache_dir="cache")
            loadsettings()
        # Load the TPU backend if requested
        elif(koboldai_vars.use_colab_tpu or koboldai_vars.model in ("TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX")):
            global tpu_mtj_backend
            import tpu_mtj_backend
            if(koboldai_vars.model == "TPUMeshTransformerGPTNeoX"):
                koboldai_vars.badwordsids = koboldai_settings.badwordsids_neox
            print("{0}Initializing Mesh Transformer JAX, please wait...{1}".format(colors.PURPLE, colors.END))
            if koboldai_vars.model in ("TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX") and (not koboldai_vars.custmodpth or not os.path.isdir(koboldai_vars.custmodpth)):
                raise FileNotFoundError(f"The specified model path {repr(koboldai_vars.custmodpth)} is not the path to a valid folder")
            import tpu_mtj_backend
            if(koboldai_vars.model == "TPUMeshTransformerGPTNeoX"):
                tpu_mtj_backend.pad_token_id = 2
            tpu_mtj_backend.koboldai_vars = koboldai_vars
            tpu_mtj_backend.warper_callback = tpumtjgenerate_warper_callback
            tpu_mtj_backend.stopping_callback = tpumtjgenerate_stopping_callback
            tpu_mtj_backend.compiling_callback = tpumtjgenerate_compiling_callback
            tpu_mtj_backend.stopped_compiling_callback = tpumtjgenerate_stopped_compiling_callback
            tpu_mtj_backend.settings_callback = tpumtjgenerate_settings_callback
            koboldai_vars.allowsp = True
            loadmodelsettings()
            loadsettings()
            tpu_mtj_backend.load_model(koboldai_vars.custmodpth, hf_checkpoint=koboldai_vars.model not in ("TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX") and koboldai_vars.use_colab_tpu, **koboldai_vars.modelconfig)
            koboldai_vars.modeldim = int(tpu_mtj_backend.params.get("d_embed", tpu_mtj_backend.params["d_model"]))
            tokenizer = tpu_mtj_backend.tokenizer
            if(koboldai_vars.badwordsids is koboldai_settings.badwordsids_default and koboldai_vars.model_type not in ("gpt2", "gpt_neo", "gptj")):
                koboldai_vars.badwordsids = [[v] for k, v in tokenizer.get_vocab().items() if any(c in str(k) for c in "<>[]") if koboldai_vars.newlinemode != "s" or str(k) != "</s>"]
        else:
            loadsettings()
    
    lua_startup()
    # Load scripts
    load_lua_scripts()
    
    final_startup()
    if not initial_load:
        set_aibusy(False)
        emit('from_server', {'cmd': 'hide_model_name'}, broadcast=True, room="UI_1")
        time.sleep(0.1)
        
        if not koboldai_vars.gamestarted:
            setStartState()
            sendsettings()
            refresh_settings()
    
    #Saving the tokenizer to the KoboldStoryRegister class so we can do token counting on the story data
    if 'tokenizer' in [x for x in globals()]:
        koboldai_vars.tokenizer = tokenizer
    
    #Let's load the presets
    with open('settings/preset/official.presets') as f:
        presets = json.load(f)
        to_use = {"Recommended": {"Default": koboldai_vars.default_preset}}
        #Check for 6B in title
        if '6B' in koboldai_vars.model or '6.7B' in koboldai_vars.model or '1.3B' in koboldai_vars.model:
            to_use['Recommended'].update(presets['6B'])
            for key in presets:
                if key != '6B':
                    to_use[key] = presets[key]
        elif '13B' in koboldai_vars.model:
            to_use['Recommended'].update(presets['13B'])
            for key in presets:
                if key != '13B':
                    to_use[key] = presets[key]
        else:
            for key in presets:
                to_use[key] = presets[key]
        koboldai_vars.presets = to_use

# Set up Flask routes
@app.route('/')
@app.route('/index')
def index():
    if 'story' in session:
        if session['story'] not in koboldai_vars.story_list():
            session['story'] = 'default'
    return render_template('index.html', hide_ai_menu=args.noaimenu, flaskwebgui=koboldai_vars.flaskwebgui)
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(app.root_path,
                                   'koboldai.ico', mimetype='image/vnd.microsoft.icon')    
@app.route('/download')
def download():
    save_format = request.args.get("format", "json").strip().lower()

    if(save_format == "plaintext"):
        txt = koboldai_vars.prompt + "".join(koboldai_vars.actions.values())
        save = Response(txt)
        filename = path.basename(koboldai_vars.savedir)
        if filename[-5:] == ".json":
            filename = filename[:-5]
        save.headers.set('Content-Disposition', 'attachment', filename='%s.txt' % filename)
        return(save)

    # Build json to write
    js = {}
    js["gamestarted"] = koboldai_vars.gamestarted
    js["prompt"]      = koboldai_vars.prompt
    js["memory"]      = koboldai_vars.memory
    js["authorsnote"] = koboldai_vars.authornote
    js["anotetemplate"] = koboldai_vars.authornotetemplate
    js["actions"]     = koboldai_vars.actions.to_json()
    js["worldinfo"]   = []
        
    # Extract only the important bits of WI
    for wi in koboldai_vars.worldinfo:
        if(wi["constant"] or wi["key"] != ""):
            js["worldinfo"].append({
                "key": wi["key"],
                "keysecondary": wi["keysecondary"],
                "content": wi["content"],
                "comment": wi["comment"],
                "folder": wi["folder"],
                "selective": wi["selective"],
                "constant": wi["constant"]
            })
    
    save = Response(json.dumps(js, indent=3))
    filename = path.basename(koboldai_vars.savedir)
    if filename[-5:] == ".json":
        filename = filename[:-5]
    save.headers.set('Content-Disposition', 'attachment', filename='%s.json' % filename)
    return(save)


#============================ LUA API =============================#
_bridged = {}
F = TypeVar("F", bound=Callable)
def lua_startup():
    global _bridged
    global F
    global bridged
    if(path.exists("settings/" + getmodelname().replace('/', '_') + ".settings")):
        file = open("settings/" + getmodelname().replace('/', '_') + ".settings", "r")
        js   = json.load(file)
        if("userscripts" in js):
            koboldai_vars.userscripts = []
            for userscript in js["userscripts"]:
                if type(userscript) is not str:
                    continue
                userscript = userscript.strip()
                if len(userscript) != 0 and all(q not in userscript for q in ("..", ":")) and all(userscript[0] not in q for q in ("/", "\\")) and os.path.exists(fileops.uspath(userscript)):
                    koboldai_vars.userscripts.append(userscript)
        if("corescript" in js and type(js["corescript"]) is str and all(q not in js["corescript"] for q in ("..", ":")) and all(js["corescript"][0] not in q for q in ("/", "\\"))):
            koboldai_vars.corescript = js["corescript"]
        else:
            koboldai_vars.corescript = "default.lua"
        file.close()
        
    #==================================================================#
    #  Lua runtime startup
    #==================================================================#

    print("", end="", flush=True)
    print(colors.PURPLE + "Initializing Lua Bridge... " + colors.END, end="", flush=True)

    # Set up Lua state
    koboldai_vars.lua_state = lupa.LuaRuntime(unpack_returned_tuples=True)

    # Load bridge.lua
    bridged = {
        "corescript_path": "cores",
        "userscript_path": "userscripts",
        "config_path": "userscripts",
        "lib_paths": koboldai_vars.lua_state.table("lualibs", os.path.join("extern", "lualibs")),
        "koboldai_vars": koboldai_vars
    }
    for kwarg in _bridged:
        bridged[kwarg] = _bridged[kwarg]
    try:
        koboldai_vars.lua_kobold, koboldai_vars.lua_koboldcore, koboldai_vars.lua_koboldbridge = koboldai_vars.lua_state.globals().dofile("bridge.lua")(
            koboldai_vars.lua_state.globals().python,
            bridged,
        )
    except lupa.LuaError as e:
        print(colors.RED + "ERROR!" + colors.END)
        koboldai_vars.lua_koboldbridge.obliterate_multiverse()
        print("{0}{1}{2}".format(colors.RED, "***LUA ERROR***: ", colors.END), end="", file=sys.stderr)
        print("{0}{1}{2}".format(colors.RED, str(e).replace("\033", ""), colors.END), file=sys.stderr)
        exit(1)
    print(colors.GREEN + "OK!" + colors.END)


def lua_log_format_name(name):
    return f"[{name}]" if type(name) is str else "CORE"


def bridged_kwarg(name=None):
    def _bridged_kwarg(f: F):
        _bridged[name if name is not None else f.__name__[4:] if f.__name__[:4] == "lua_" else f.__name__] = f
        return f
    return _bridged_kwarg

#==================================================================#
#  Event triggered when a userscript is loaded
#==================================================================#
@bridged_kwarg()
def load_callback(filename, modulename):
    print(colors.GREEN + f"Loading Userscript [{modulename}] <{filename}>" + colors.END)

#==================================================================#
#  Load all Lua scripts
#==================================================================#
def load_lua_scripts():
    print(colors.GREEN + "Loading Core Script" + colors.END)

    filenames = []
    modulenames = []
    descriptions = []

    lst = fileops.getusfiles(long_desc=True)
    filenames_dict = {ob["filename"]: i for i, ob in enumerate(lst)}

    for filename in koboldai_vars.userscripts:
        if filename in filenames_dict:
            i = filenames_dict[filename]
            filenames.append(filename)
            modulenames.append(lst[i]["modulename"])
            descriptions.append(lst[i]["description"])

    koboldai_vars.has_genmod = False

    try:
        koboldai_vars.lua_koboldbridge.obliterate_multiverse()
        tpool.execute(koboldai_vars.lua_koboldbridge.load_corescript, koboldai_vars.corescript)
        koboldai_vars.has_genmod = tpool.execute(koboldai_vars.lua_koboldbridge.load_userscripts, filenames, modulenames, descriptions)
        koboldai_vars.lua_running = True
    except lupa.LuaError as e:
        try:
            koboldai_vars.lua_koboldbridge.obliterate_multiverse()
        except:
            pass
        koboldai_vars.lua_running = False
        if(koboldai_vars.serverstarted):
            emit('from_server', {'cmd': 'errmsg', 'data': 'Lua script error; please check console.'}, broadcast=True, room="UI_1")
            sendUSStatItems()
        print("{0}{1}{2}".format(colors.RED, "***LUA ERROR***: ", colors.END), end="", file=sys.stderr)
        print("{0}{1}{2}".format(colors.RED, str(e).replace("\033", ""), colors.END), file=sys.stderr)
        print("{0}{1}{2}".format(colors.YELLOW, "Lua engine stopped; please open 'Userscripts' and press Load to reinitialize scripts.", colors.END), file=sys.stderr)
        if(koboldai_vars.serverstarted):
            set_aibusy(0)

#==================================================================#
#  Print message that originates from the userscript with the given name
#==================================================================#
@bridged_kwarg()
def lua_print(msg):
    if(koboldai_vars.lua_logname != koboldai_vars.lua_koboldbridge.logging_name):
        koboldai_vars.lua_logname = koboldai_vars.lua_koboldbridge.logging_name
        print(colors.BLUE + lua_log_format_name(koboldai_vars.lua_logname) + ":" + colors.END, file=sys.stderr)
    print(colors.PURPLE + msg.replace("\033", "") + colors.END)

#==================================================================#
#  Print warning that originates from the userscript with the given name
#==================================================================#
@bridged_kwarg()
def lua_warn(msg):
    if(koboldai_vars.lua_logname != koboldai_vars.lua_koboldbridge.logging_name):
        koboldai_vars.lua_logname = koboldai_vars.lua_koboldbridge.logging_name
        print(colors.BLUE + lua_log_format_name(koboldai_vars.lua_logname) + ":" + colors.END, file=sys.stderr)
    print(colors.YELLOW + msg.replace("\033", "") + colors.END)

#==================================================================#
#  Decode tokens into a string using current tokenizer
#==================================================================#
@bridged_kwarg()
def lua_decode(tokens):
    tokens = list(tokens.values())
    assert type(tokens) is list
    if("tokenizer" not in globals()):
        from transformers import GPT2TokenizerFast
        global tokenizer
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", revision=koboldai_vars.revision, cache_dir="cache")
    return utils.decodenewlines(tokenizer.decode(tokens))

#==================================================================#
#  Encode string into list of token IDs using current tokenizer
#==================================================================#
@bridged_kwarg()
def lua_encode(string):
    assert type(string) is str
    if("tokenizer" not in globals()):
        from transformers import GPT2TokenizerFast
        global tokenizer
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", revision=koboldai_vars.revision, cache_dir="cache")
    return tokenizer.encode(utils.encodenewlines(string), max_length=int(4e9), truncation=True)

#==================================================================#
#  Computes context given a submission, Lua array of entry UIDs and a Lua array
#  of folder UIDs
#==================================================================#
@bridged_kwarg()
def lua_compute_context(submission, entries, folders, kwargs):
    assert type(submission) is str
    if(kwargs is None):
        kwargs = koboldai_vars.lua_state.table()
    actions = koboldai_vars._actions if koboldai_vars.lua_koboldbridge.userstate == "genmod" else koboldai_vars.actions
    allowed_entries = None
    allowed_folders = None
    if(entries is not None):
        allowed_entries = set()
        i = 1
        while(entries[i] is not None):
            allowed_entries.add(int(entries[i]))
            i += 1
    if(folders is not None):
        allowed_folders = set()
        i = 1
        while(folders[i] is not None):
            allowed_folders.add(int(folders[i]))
            i += 1
    winfo, mem, anotetxt, _ = calcsubmitbudgetheader(
        submission,
        allowed_entries=allowed_entries,
        allowed_folders=allowed_folders,
        force_use_txt=True,
        scan_story=kwargs["scan_story"] if kwargs["scan_story"] != None else True,
    )
    txt, _, _ = calcsubmitbudget(
        len(actions),
        winfo,
        mem,
        anotetxt,
        actions,
    )
    return utils.decodenewlines(tokenizer.decode(txt))

#==================================================================#
#  Get property of a world info entry given its UID and property name
#==================================================================#
@bridged_kwarg()
def lua_get_attr(uid, k):
    assert type(uid) is int and type(k) is str
    if(uid in koboldai_vars.worldinfo_u and k in (
        "key",
        "keysecondary",
        "content",
        "comment",
        "folder",
        "num",
        "selective",
        "constant",
        "uid",
    )):
        return koboldai_vars.worldinfo_u[uid][k]

#==================================================================#
#  Set property of a world info entry given its UID, property name and new value
#==================================================================#
@bridged_kwarg()
def lua_set_attr(uid, k, v):
    assert type(uid) is int and type(k) is str
    assert uid in koboldai_vars.worldinfo_u and k in (
        "key",
        "keysecondary",
        "content",
        "comment",
        "selective",
        "constant",
    )
    if(type(koboldai_vars.worldinfo_u[uid][k]) is int and type(v) is float):
        v = int(v)
    assert type(koboldai_vars.worldinfo_u[uid][k]) is type(v)
    koboldai_vars.worldinfo_u[uid][k] = v
    print(colors.GREEN + f"{lua_log_format_name(koboldai_vars.lua_koboldbridge.logging_name)} set {k} of world info entry {uid} to {v}" + colors.END)

#==================================================================#
#  Get property of a world info folder given its UID and property name
#==================================================================#
@bridged_kwarg()
def lua_folder_get_attr(uid, k):
    assert type(uid) is int and type(k) is str
    if(uid in koboldai_vars.wifolders_d and k in (
        "name",
    )):
        return koboldai_vars.wifolders_d[uid][k]

#==================================================================#
#  Set property of a world info folder given its UID, property name and new value
#==================================================================#
@bridged_kwarg()
def lua_folder_set_attr(uid, k, v):
    assert type(uid) is int and type(k) is str
    assert uid in koboldai_vars.wifolders_d and k in (
        "name",
    )
    if(type(koboldai_vars.wifolders_d[uid][k]) is int and type(v) is float):
        v = int(v)
    assert type(koboldai_vars.wifolders_d[uid][k]) is type(v)
    koboldai_vars.wifolders_d[uid][k] = v
    print(colors.GREEN + f"{lua_log_format_name(koboldai_vars.lua_koboldbridge.logging_name)} set {k} of world info folder {uid} to {v}" + colors.END)

#==================================================================#
#  Get the "Amount to Generate"
#==================================================================#
@bridged_kwarg()
def lua_get_genamt():
    return koboldai_vars.genamt

#==================================================================#
#  Set the "Amount to Generate"
#==================================================================#
@bridged_kwarg()
def lua_set_genamt(genamt):
    assert koboldai_vars.lua_koboldbridge.userstate != "genmod" and type(genamt) in (int, float) and genamt >= 0
    print(colors.GREEN + f"{lua_log_format_name(koboldai_vars.lua_koboldbridge.logging_name)} set genamt to {int(genamt)}" + colors.END)
    koboldai_vars.genamt = int(genamt)

#==================================================================#
#  Get the "Gens Per Action"
#==================================================================#
@bridged_kwarg()
def lua_get_numseqs():
    return koboldai_vars.numseqs

#==================================================================#
#  Set the "Gens Per Action"
#==================================================================#
@bridged_kwarg()
def lua_set_numseqs(numseqs):
    assert type(numseqs) in (int, float) and numseqs >= 1
    print(colors.GREEN + f"{lua_log_format_name(koboldai_vars.lua_koboldbridge.logging_name)} set numseqs to {int(numseqs)}" + colors.END)
    koboldai_vars.numseqs = int(numseqs)

#==================================================================#
#  Check if a setting exists with the given name
#==================================================================#
@bridged_kwarg()
def lua_has_setting(setting):
    return setting in (
        "anotedepth",
        "settemp",
        "settopp",
        "settopk",
        "settfs",
        "settypical",
        "settopa",
        "setreppen",
        "setreppenslope",
        "setreppenrange",
        "settknmax",
        "setwidepth",
        "setuseprompt",
        "setadventure",
        "setchatmode",
        "setdynamicscan",
        "setnopromptgen",
        "autosave",
        "setrngpersist",
        "temp",
        "topp",
        "top_p",
        "topk",
        "top_k",
        "tfs",
        "typical",
        "topa",
        "reppen",
        "reppenslope",
        "reppenrange",
        "tknmax",
        "widepth",
        "useprompt",
        "chatmode",
        "chatname",
        "adventure",
        "dynamicscan",
        "nopromptgen",
        "rngpersist",
        "frmttriminc",
        "frmtrmblln",
        "frmtrmspch",
        "frmtadsnsp",
        "frmtsingleline",
        "triminc",
        "rmblln",
        "rmspch",
        "adsnsp",
        "singleline",
        "output_streaming",
    )

#==================================================================#
#  Return the setting with the given name if it exists
#==================================================================#
@bridged_kwarg()
def lua_get_setting(setting):
    if(setting in ("settemp", "temp")): return koboldai_vars.temp
    if(setting in ("settopp", "topp", "top_p")): return koboldai_vars.top_p
    if(setting in ("settopk", "topk", "top_k")): return koboldai_vars.top_k
    if(setting in ("settfs", "tfs")): return koboldai_vars.tfs
    if(setting in ("settypical", "typical")): return koboldai_vars.typical
    if(setting in ("settopa", "topa")): return koboldai_vars.top_a
    if(setting in ("setreppen", "reppen")): return koboldai_vars.rep_pen
    if(setting in ("setreppenslope", "reppenslope")): return koboldai_vars.rep_pen_slope
    if(setting in ("setreppenrange", "reppenrange")): return koboldai_vars.rep_pen_range
    if(setting in ("settknmax", "tknmax")): return koboldai_vars.max_length
    if(setting == "anotedepth"): return koboldai_vars.andepth
    if(setting in ("setwidepth", "widepth")): return koboldai_vars.widepth
    if(setting in ("setuseprompt", "useprompt")): return koboldai_vars.useprompt
    if(setting in ("setadventure", "adventure")): return koboldai_vars.adventure
    if(setting in ("setchatmode", "chatmode")): return koboldai_vars.chatmode
    if(setting in ("setdynamicscan", "dynamicscan")): return koboldai_vars.dynamicscan
    if(setting in ("setnopromptgen", "nopromptgen")): return koboldai_vars.nopromptgen
    if(setting in ("autosave", "autosave")): return koboldai_vars.autosave
    if(setting in ("setrngpersist", "rngpersist")): return koboldai_vars.rngpersist
    if(setting in ("frmttriminc", "triminc")): return koboldai_vars.formatoptns["frmttriminc"]
    if(setting in ("frmtrmblln", "rmblln")): return koboldai_vars.formatoptns["frmttrmblln"]
    if(setting in ("frmtrmspch", "rmspch")): return koboldai_vars.formatoptns["frmttrmspch"]
    if(setting in ("frmtadsnsp", "adsnsp")): return koboldai_vars.formatoptns["frmtadsnsp"]
    if(setting in ("frmtsingleline", "singleline")): return koboldai_vars.formatoptns["singleline"]
    if(setting == "outputstreaming"): return koboldai_vars.output_streaming

#==================================================================#
#  Set the setting with the given name if it exists
#==================================================================#
@bridged_kwarg()
def lua_set_setting(setting, v):
    actual_type = type(lua_get_setting(setting))
    assert v is not None and (actual_type is type(v) or (actual_type is int and type(v) is float))
    v = actual_type(v)
    print(colors.GREEN + f"{lua_log_format_name(koboldai_vars.lua_koboldbridge.logging_name)} set {setting} to {v}" + colors.END)
    if(setting in ("setadventure", "adventure") and v):
        koboldai_vars.actionmode = 1
    if(setting in ("settemp", "temp")): koboldai_vars.temp = v
    if(setting in ("settopp", "topp")): koboldai_vars.top_p = v
    if(setting in ("settopk", "topk")): koboldai_vars.top_k = v
    if(setting in ("settfs", "tfs")): koboldai_vars.tfs = v
    if(setting in ("settypical", "typical")): koboldai_vars.typical = v
    if(setting in ("settopa", "topa")): koboldai_vars.top_a = v
    if(setting in ("setreppen", "reppen")): koboldai_vars.rep_pen = v
    if(setting in ("setreppenslope", "reppenslope")): koboldai_vars.rep_pen_slope = v
    if(setting in ("setreppenrange", "reppenrange")): koboldai_vars.rep_pen_range = v
    if(setting in ("settknmax", "tknmax")): koboldai_vars.max_length = v; return True
    if(setting == "anotedepth"): koboldai_vars.andepth = v; return True
    if(setting in ("setwidepth", "widepth")): koboldai_vars.widepth = v; return True
    if(setting in ("setuseprompt", "useprompt")): koboldai_vars.useprompt = v; return True
    if(setting in ("setadventure", "adventure")): koboldai_vars.adventure = v
    if(setting in ("setdynamicscan", "dynamicscan")): koboldai_vars.dynamicscan = v
    if(setting in ("setnopromptgen", "nopromptgen")): koboldai_vars.nopromptgen = v
    if(setting in ("autosave", "noautosave")): koboldai_vars.autosave = v
    if(setting in ("setrngpersist", "rngpersist")): koboldai_vars.rngpersist = v
    if(setting in ("setchatmode", "chatmode")): koboldai_vars.chatmode = v
    if(setting in ("frmttriminc", "triminc")): koboldai_vars.formatoptns["frmttriminc"] = v
    if(setting in ("frmtrmblln", "rmblln")): koboldai_vars.formatoptns["frmttrmblln"] = v
    if(setting in ("frmtrmspch", "rmspch")): koboldai_vars.formatoptns["frmttrmspch"] = v
    if(setting in ("frmtadsnsp", "adsnsp")): koboldai_vars.formatoptns["frmtadsnsp"] = v
    if(setting in ("frmtsingleline", "singleline")): koboldai_vars.formatoptns["singleline"] = v
    if(setting == "outputstreaming"): koboldai_vars.output_streaming = v

#==================================================================#
#  Get contents of memory
#==================================================================#
@bridged_kwarg()
def lua_get_memory():
    return koboldai_vars.memory

#==================================================================#
#  Set contents of memory
#==================================================================#
@bridged_kwarg()
def lua_set_memory(m):
    assert type(m) is str
    koboldai_vars.memory = m

#==================================================================#
#  Get contents of author's note
#==================================================================#
@bridged_kwarg()
def lua_get_authorsnote():
    return koboldai_vars.authornote

#==================================================================#
#  Set contents of author's note
#==================================================================#
@bridged_kwarg()
def lua_set_authorsnote(m):
    assert type(m) is str
    koboldai_vars.authornote = m

#==================================================================#
#  Get contents of author's note template
#==================================================================#
@bridged_kwarg()
def lua_get_authorsnotetemplate():
    return koboldai_vars.authornotetemplate

#==================================================================#
#  Set contents of author's note template
#==================================================================#
@bridged_kwarg()
def lua_set_authorsnotetemplate(m):
    assert type(m) is str
    koboldai_vars.authornotetemplate = m

#==================================================================#
#  Save settings and send them to client
#==================================================================#
@bridged_kwarg()
def lua_resend_settings():
    print("lua_resend_settings")
    settingschanged()
    refresh_settings()

#==================================================================#
#  Set story chunk text and delete the chunk if the new chunk is empty
#==================================================================#
@bridged_kwarg()
def lua_set_chunk(k, v):
    assert type(k) in (int, None) and type(v) is str
    assert k >= 0
    assert k != 0 or len(v) != 0
    if(len(v) == 0):
        print(colors.GREEN + f"{lua_log_format_name(koboldai_vars.lua_koboldbridge.logging_name)} deleted story chunk {k}" + colors.END)
        chunk = int(k)
        if(koboldai_vars.lua_koboldbridge.userstate == "genmod"):
            del koboldai_vars._actions[chunk-1]
        koboldai_vars.lua_deleted.add(chunk)
        if(not hasattr(koboldai_vars, "_actions") or koboldai_vars._actions is not koboldai_vars.actions):
            #Instead of deleting we'll blank out the text. This way our actions and actions_metadata stay in sync and we can restore the chunk on an undo
            koboldai_vars.actions[chunk-1] = ""
            send_debug()
    else:
        if(k == 0):
            print(colors.GREEN + f"{lua_log_format_name(koboldai_vars.lua_koboldbridge.logging_name)} edited prompt chunk" + colors.END)
        else:
            print(colors.GREEN + f"{lua_log_format_name(koboldai_vars.lua_koboldbridge.logging_name)} edited story chunk {k}" + colors.END)
        chunk = int(k)
        if(chunk == 0):
            if(koboldai_vars.lua_koboldbridge.userstate == "genmod"):
                koboldai_vars._prompt = v
            koboldai_vars.lua_edited.add(chunk)
            koboldai_vars.prompt = v
        else:
            if(koboldai_vars.lua_koboldbridge.userstate == "genmod"):
                koboldai_vars._actions[chunk-1] = v
            koboldai_vars.lua_edited.add(chunk)
            koboldai_vars.actions[chunk-1] = v
            send_debug()

#==================================================================#
#  Get model type as "gpt-2-xl", "gpt-neo-2.7B", etc.
#==================================================================#
@bridged_kwarg()
def lua_get_modeltype():
    if(koboldai_vars.noai):
        return "readonly"
    if(koboldai_vars.model in ("Colab", "OAI", "InferKit")):
        return "api"
    if(not koboldai_vars.use_colab_tpu and koboldai_vars.model not in ("TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX") and (koboldai_vars.model in ("GPT2Custom", "NeoCustom") or koboldai_vars.model_type in ("gpt2", "gpt_neo", "gptj"))):
        hidden_size = get_hidden_size_from_model(model)
    if(koboldai_vars.model in ("gpt2",) or (koboldai_vars.model_type == "gpt2" and hidden_size == 768)):
        return "gpt2"
    if(koboldai_vars.model in ("gpt2-medium",) or (koboldai_vars.model_type == "gpt2" and hidden_size == 1024)):
        return "gpt2-medium"
    if(koboldai_vars.model in ("gpt2-large",) or (koboldai_vars.model_type == "gpt2" and hidden_size == 1280)):
        return "gpt2-large"
    if(koboldai_vars.model in ("gpt2-xl",) or (koboldai_vars.model_type == "gpt2" and hidden_size == 1600)):
        return "gpt2-xl"
    if(koboldai_vars.model_type == "gpt_neo" and hidden_size == 768):
        return "gpt-neo-125M"
    if(koboldai_vars.model in ("EleutherAI/gpt-neo-1.3B",) or (koboldai_vars.model_type == "gpt_neo" and hidden_size == 2048)):
        return "gpt-neo-1.3B"
    if(koboldai_vars.model in ("EleutherAI/gpt-neo-2.7B",) or (koboldai_vars.model_type == "gpt_neo" and hidden_size == 2560)):
        return "gpt-neo-2.7B"
    if(koboldai_vars.model in ("EleutherAI/gpt-j-6B",) or ((koboldai_vars.use_colab_tpu or koboldai_vars.model == "TPUMeshTransformerGPTJ") and tpu_mtj_backend.params["d_model"] == 4096) or (koboldai_vars.model_type in ("gpt_neo", "gptj") and hidden_size == 4096)):
        return "gpt-j-6B"
    return "unknown"

#==================================================================#
#  Get model backend as "transformers" or "mtj"
#==================================================================#
@bridged_kwarg()
def lua_get_modelbackend():
    if(koboldai_vars.noai):
        return "readonly"
    if(koboldai_vars.model in ("Colab", "OAI", "InferKit")):
        return "api"
    if(koboldai_vars.use_colab_tpu or koboldai_vars.model in ("TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX")):
        return "mtj"
    return "transformers"

#==================================================================#
#  Check whether model is loaded from a custom path
#==================================================================#
@bridged_kwarg()
def lua_is_custommodel():
    return koboldai_vars.model in ("GPT2Custom", "NeoCustom", "TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX")

#==================================================================#
#  Return the filename (as a string) of the current soft prompt, or
#  None if no soft prompt is loaded
#==================================================================#
@bridged_kwarg()
def lua_get_spfilename():
    return koboldai_vars.spfilename.strip() or None

#==================================================================#
#  When called with a string as argument, sets the current soft prompt;
#  when called with None as argument, uses no soft prompt.
#  Returns True if soft prompt changed, False otherwise.
#==================================================================#
@bridged_kwarg()
def lua_set_spfilename(filename: Union[str, None]):
    if(filename is None):
        filename = ""
    filename = str(filename).strip()
    changed = lua_get_spfilename() != filename
    assert all(q not in filename for q in ("/", "\\"))
    spRequest(filename)
    return changed

#==================================================================#
#  
#==================================================================#
def execute_inmod():
    setgamesaved(False)
    koboldai_vars.lua_logname = ...
    koboldai_vars.lua_edited = set()
    koboldai_vars.lua_deleted = set()
    try:
        tpool.execute(koboldai_vars.lua_koboldbridge.execute_inmod)
    except lupa.LuaError as e:
        koboldai_vars.lua_koboldbridge.obliterate_multiverse()
        koboldai_vars.lua_running = False
        emit('from_server', {'cmd': 'errmsg', 'data': 'Lua script error; please check console.'}, broadcast=True, room="UI_1")
        sendUSStatItems()
        print("{0}{1}{2}".format(colors.RED, "***LUA ERROR***: ", colors.END), end="", file=sys.stderr)
        print("{0}{1}{2}".format(colors.RED, str(e).replace("\033", ""), colors.END), file=sys.stderr)
        print("{0}{1}{2}".format(colors.YELLOW, "Lua engine stopped; please open 'Userscripts' and press Load to reinitialize scripts.", colors.END), file=sys.stderr)
        set_aibusy(0)

def execute_genmod():
    koboldai_vars.lua_koboldbridge.execute_genmod()

def execute_outmod():
    setgamesaved(False)
    emit('from_server', {'cmd': 'hidemsg', 'data': ''}, broadcast=True, room="UI_1")
    try:
        tpool.execute(koboldai_vars.lua_koboldbridge.execute_outmod)
    except lupa.LuaError as e:
        koboldai_vars.lua_koboldbridge.obliterate_multiverse()
        koboldai_vars.lua_running = False
        emit('from_server', {'cmd': 'errmsg', 'data': 'Lua script error; please check console.'}, broadcast=True, room="UI_1")
        sendUSStatItems()
        print("{0}{1}{2}".format(colors.RED, "***LUA ERROR***: ", colors.END), end="", file=sys.stderr)
        print("{0}{1}{2}".format(colors.RED, str(e).replace("\033", ""), colors.END), file=sys.stderr)
        print("{0}{1}{2}".format(colors.YELLOW, "Lua engine stopped; please open 'Userscripts' and press Load to reinitialize scripts.", colors.END), file=sys.stderr)
        set_aibusy(0)
    if(koboldai_vars.lua_koboldbridge.resend_settings_required):
        koboldai_vars.lua_koboldbridge.resend_settings_required = False
        lua_resend_settings()
    for k in koboldai_vars.lua_edited:
        inlineedit(k, koboldai_vars.actions[k])
    for k in koboldai_vars.lua_deleted:
        inlinedelete(k)




#============================ METHODS =============================#    

#==================================================================#
# Event triggered when browser SocketIO is loaded and connects to server
#==================================================================#
@socketio.on('connect')
def do_connect():
    if request.args.get("rely") == "true":
        return
    join_room("UI_{}".format(request.args.get('ui')))
    print("Joining Room UI_{}".format(request.args.get('ui')))
    if request.args.get("ui") == "2":
        ui2_connect()
        return
    print("{0}Client connected!{1}".format(colors.GREEN, colors.END))
    emit('from_server', {'cmd': 'setchatname', 'data': koboldai_vars.chatname}, room="UI_1")
    emit('from_server', {'cmd': 'setanotetemplate', 'data': koboldai_vars.authornotetemplate}, room="UI_1")
    emit('from_server', {'cmd': 'connected', 'smandelete': koboldai_vars.smandelete, 'smanrename': koboldai_vars.smanrename, 'modelname': getmodelname()}, room="UI_1")
    if(koboldai_vars.host):
        emit('from_server', {'cmd': 'runs_remotely'}, room="UI_1")
    if(koboldai_vars.flaskwebgui):
        emit('from_server', {'cmd': 'flaskwebgui'}, room="UI_1")
    if(koboldai_vars.allowsp):
        emit('from_server', {'cmd': 'allowsp', 'data': koboldai_vars.allowsp}, room="UI_1")

    sendUSStatItems()
    emit('from_server', {'cmd': 'spstatitems', 'data': {koboldai_vars.spfilename: koboldai_vars.spmeta} if koboldai_vars.allowsp and len(koboldai_vars.spfilename) else {}}, broadcast=True, room="UI_1")

    if(not koboldai_vars.gamestarted):
        setStartState()
        sendsettings()
        refresh_settings()
        koboldai_vars.laststory = None
        emit('from_server', {'cmd': 'setstoryname', 'data': koboldai_vars.laststory}, room="UI_1")
        sendwi()
        emit('from_server', {'cmd': 'setmemory', 'data': koboldai_vars.memory}, room="UI_1")
        emit('from_server', {'cmd': 'setanote', 'data': koboldai_vars.authornote}, room="UI_1")
        koboldai_vars.mode = "play"
    else:
        # Game in session, send current game data and ready state to browser
        refresh_story()
        sendsettings()
        refresh_settings()
        emit('from_server', {'cmd': 'setstoryname', 'data': koboldai_vars.laststory}, room="UI_1")
        sendwi()
        emit('from_server', {'cmd': 'setmemory', 'data': koboldai_vars.memory}, room="UI_1")
        emit('from_server', {'cmd': 'setanote', 'data': koboldai_vars.authornote}, room="UI_1")
        if(koboldai_vars.mode == "play"):
            if(not koboldai_vars.aibusy):
                emit('from_server', {'cmd': 'setgamestate', 'data': 'ready'}, room="UI_1")
            else:
                emit('from_server', {'cmd': 'setgamestate', 'data': 'wait'}, room="UI_1")
        elif(koboldai_vars.mode == "edit"):
            emit('from_server', {'cmd': 'editmode', 'data': 'true'}, room="UI_1")
        elif(koboldai_vars.mode == "memory"):
            emit('from_server', {'cmd': 'memmode', 'data': 'true'}, room="UI_1")
        elif(koboldai_vars.mode == "wi"):
            emit('from_server', {'cmd': 'wimode', 'data': 'true'}, room="UI_1")

    emit('from_server', {'cmd': 'gamesaved', 'data': koboldai_vars.gamesaved}, broadcast=True, room="UI_1")

#==================================================================#
# Event triggered when browser SocketIO sends data to the server
#==================================================================#
@socketio.on('message')
def get_message(msg):
    if not koboldai_vars.quiet:
        print("{0}Data received:{1}{2}".format(colors.GREEN, msg, colors.END))
    # Submit action
    if(msg['cmd'] == 'submit'):
        if(koboldai_vars.mode == "play"):
            if(koboldai_vars.aibusy):
                if(msg.get('allowabort', False)):
                    koboldai_vars.abort = True
                return
            koboldai_vars.abort = False
            koboldai_vars.lua_koboldbridge.feedback = None
            if(koboldai_vars.chatmode):
                if(type(msg['chatname']) is not str):
                    raise ValueError("Chatname must be a string")
                koboldai_vars.chatname = msg['chatname']
                settingschanged()
                emit('from_server', {'cmd': 'setchatname', 'data': koboldai_vars.chatname}, room="UI_1")
            koboldai_vars.recentrng = koboldai_vars.recentrngm = None
            actionsubmit(msg['data'], actionmode=msg['actionmode'])
        elif(koboldai_vars.mode == "edit"):
            editsubmit(msg['data'])
        elif(koboldai_vars.mode == "memory"):
            memsubmit(msg['data'])
    # Retry Action
    elif(msg['cmd'] == 'retry'):
        if(koboldai_vars.aibusy):
            if(msg.get('allowabort', False)):
                koboldai_vars.abort = True
            return
        koboldai_vars.abort = False
        if(koboldai_vars.chatmode):
            if(type(msg['chatname']) is not str):
                raise ValueError("Chatname must be a string")
            koboldai_vars.chatname = msg['chatname']
            settingschanged()
            emit('from_server', {'cmd': 'setchatname', 'data': koboldai_vars.chatname}, room="UI_1")
        actionretry(msg['data'])
    # Back/Undo Action
    elif(msg['cmd'] == 'back'):
        ignore = actionback()
    # Forward/Redo Action
    elif(msg['cmd'] == 'redo'):
        actionredo()
    # EditMode Action (old)
    elif(msg['cmd'] == 'edit'):
        if(koboldai_vars.mode == "play"):
            koboldai_vars.mode = "edit"
            emit('from_server', {'cmd': 'editmode', 'data': 'true'}, broadcast=True, room="UI_1")
        elif(koboldai_vars.mode == "edit"):
            koboldai_vars.mode = "play"
            emit('from_server', {'cmd': 'editmode', 'data': 'false'}, broadcast=True, room="UI_1")
    # EditLine Action (old)
    elif(msg['cmd'] == 'editline'):
        editrequest(int(msg['data']))
    # Inline edit
    elif(msg['cmd'] == 'inlineedit'):
        inlineedit(msg['chunk'], msg['data'])
    elif(msg['cmd'] == 'inlinedelete'):
        inlinedelete(msg['data'])
    # DeleteLine Action (old)
    elif(msg['cmd'] == 'delete'):
        deleterequest()
    elif(msg['cmd'] == 'memory'):
        togglememorymode()
    elif(not koboldai_vars.host and msg['cmd'] == 'savetofile'):
        savetofile()
    elif(not koboldai_vars.host and msg['cmd'] == 'loadfromfile'):
        loadfromfile()
    elif(msg['cmd'] == 'loadfromstring'):
        loadRequest(json.loads(msg['data']), filename=msg['filename'])
    elif(not koboldai_vars.host and msg['cmd'] == 'import'):
        importRequest()
    elif(msg['cmd'] == 'newgame'):
        newGameRequest()
    elif(msg['cmd'] == 'rndgame'):
        randomGameRequest(msg['data'], memory=msg['memory'])
    elif(msg['cmd'] == 'settemp'):
        koboldai_vars.temp = float(msg['data'])
        emit('from_server', {'cmd': 'setlabeltemp', 'data': msg['data']}, broadcast=True, room="UI_1")
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'settopp'):
        koboldai_vars.top_p = float(msg['data'])
        emit('from_server', {'cmd': 'setlabeltopp', 'data': msg['data']}, broadcast=True, room="UI_1")
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'settopk'):
        koboldai_vars.top_k = int(msg['data'])
        emit('from_server', {'cmd': 'setlabeltopk', 'data': msg['data']}, broadcast=True, room="UI_1")
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'settfs'):
        koboldai_vars.tfs = float(msg['data'])
        emit('from_server', {'cmd': 'setlabeltfs', 'data': msg['data']}, broadcast=True, room="UI_1")
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'settypical'):
        koboldai_vars.typical = float(msg['data'])
        emit('from_server', {'cmd': 'setlabeltypical', 'data': msg['data']}, broadcast=True, room="UI_1")
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'settopa'):
        koboldai_vars.top_a = float(msg['data'])
        emit('from_server', {'cmd': 'setlabeltopa', 'data': msg['data']}, broadcast=True, room="UI_1")
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setreppen'):
        koboldai_vars.rep_pen = float(msg['data'])
        emit('from_server', {'cmd': 'setlabelreppen', 'data': msg['data']}, broadcast=True, room="UI_1")
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setreppenslope'):
        koboldai_vars.rep_pen_slope = float(msg['data'])
        emit('from_server', {'cmd': 'setlabelreppenslope', 'data': msg['data']}, broadcast=True, room="UI_1")
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setreppenrange'):
        koboldai_vars.rep_pen_range = float(msg['data'])
        emit('from_server', {'cmd': 'setlabelreppenrange', 'data': msg['data']}, broadcast=True, room="UI_1")
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setoutput'):
        koboldai_vars.genamt = int(msg['data'])
        emit('from_server', {'cmd': 'setlabeloutput', 'data': msg['data']}, broadcast=True, room="UI_1")
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'settknmax'):
        koboldai_vars.max_length = int(msg['data'])
        emit('from_server', {'cmd': 'setlabeltknmax', 'data': msg['data']}, broadcast=True, room="UI_1")
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setikgen'):
        koboldai_vars.ikgen = int(msg['data'])
        emit('from_server', {'cmd': 'setlabelikgen', 'data': msg['data']}, broadcast=True, room="UI_1")
        settingschanged()
        refresh_settings()
    # Author's Note field update
    elif(msg['cmd'] == 'anote'):
        anotesubmit(msg['data'], template=msg['template'])
    # Author's Note depth update
    elif(msg['cmd'] == 'anotedepth'):
        koboldai_vars.andepth = int(msg['data'])
        emit('from_server', {'cmd': 'setlabelanotedepth', 'data': msg['data']}, broadcast=True, room="UI_1")
        settingschanged()
        refresh_settings()
    # Format - Trim incomplete sentences
    elif(msg['cmd'] == 'frmttriminc'):
        if('frmttriminc' in koboldai_vars.formatoptns):
            koboldai_vars.formatoptns["frmttriminc"] = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'frmtrmblln'):
        if('frmtrmblln' in koboldai_vars.formatoptns):
            koboldai_vars.formatoptns["frmtrmblln"] = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'frmtrmspch'):
        if('frmtrmspch' in koboldai_vars.formatoptns):
            koboldai_vars.formatoptns["frmtrmspch"] = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'frmtadsnsp'):
        if('frmtadsnsp' in koboldai_vars.formatoptns):
            koboldai_vars.formatoptns["frmtadsnsp"] = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'singleline'):
        if('singleline' in koboldai_vars.formatoptns):
            koboldai_vars.formatoptns["singleline"] = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'importselect'):
        koboldai_vars.importnum = int(msg["data"].replace("import", ""))
    elif(msg['cmd'] == 'importcancel'):
        emit('from_server', {'cmd': 'popupshow', 'data': False}, room="UI_1")
        koboldai_vars.importjs  = {}
    elif(msg['cmd'] == 'importaccept'):
        emit('from_server', {'cmd': 'popupshow', 'data': False}, room="UI_1")
        importgame()
    elif(msg['cmd'] == 'wi'):
        togglewimode()
    elif(msg['cmd'] == 'wiinit'):
        if(int(msg['data']) < len(koboldai_vars.worldinfo)):
            setgamesaved(False)
            koboldai_vars.worldinfo[msg['data']]["init"] = True
            addwiitem(folder_uid=msg['folder'])
    elif(msg['cmd'] == 'wifolderinit'):
        addwifolder()
    elif(msg['cmd'] == 'wimoveitem'):
        movewiitem(msg['destination'], msg['data'])
    elif(msg['cmd'] == 'wimovefolder'):
        movewifolder(msg['destination'], msg['data'])
    elif(msg['cmd'] == 'widelete'):
        deletewi(msg['data'])
    elif(msg['cmd'] == 'wifolderdelete'):
        deletewifolder(msg['data'])
    elif(msg['cmd'] == 'wiexpand'):
        assert 0 <= int(msg['data']) < len(koboldai_vars.worldinfo)
        setgamesaved(False)
        emit('from_server', {'cmd': 'wiexpand', 'data': msg['data']}, broadcast=True, room="UI_1")
    elif(msg['cmd'] == 'wiexpandfolder'):
        assert 0 <= int(msg['data']) < len(koboldai_vars.worldinfo)
        setgamesaved(False)
        emit('from_server', {'cmd': 'wiexpandfolder', 'data': msg['data']}, broadcast=True, room="UI_1")
    elif(msg['cmd'] == 'wifoldercollapsecontent'):
        setgamesaved(False)
        koboldai_vars.wifolders_d[msg['data']]['collapsed'] = True
        emit('from_server', {'cmd': 'wifoldercollapsecontent', 'data': msg['data']}, broadcast=True, room="UI_1")
    elif(msg['cmd'] == 'wifolderexpandcontent'):
        setgamesaved(False)
        koboldai_vars.wifolders_d[msg['data']]['collapsed'] = False
        emit('from_server', {'cmd': 'wifolderexpandcontent', 'data': msg['data']}, broadcast=True, room="UI_1")
    elif(msg['cmd'] == 'wiupdate'):
        setgamesaved(False)
        num = int(msg['num'])
        fields = ("key", "keysecondary", "content", "comment")
        for field in fields:
            if(field in msg['data'] and type(msg['data'][field]) is str):
                koboldai_vars.worldinfo[num][field] = msg['data'][field]
        emit('from_server', {'cmd': 'wiupdate', 'num': msg['num'], 'data': {field: koboldai_vars.worldinfo[num][field] for field in fields}}, broadcast=True, room="UI_1")
    elif(msg['cmd'] == 'wifolderupdate'):
        setgamesaved(False)
        uid = int(msg['uid'])
        fields = ("name", "collapsed")
        for field in fields:
            if(field in msg['data'] and type(msg['data'][field]) is (str if field != "collapsed" else bool)):
                koboldai_vars.wifolders_d[uid][field] = msg['data'][field]
        emit('from_server', {'cmd': 'wifolderupdate', 'uid': msg['uid'], 'data': {field: koboldai_vars.wifolders_d[uid][field] for field in fields}}, broadcast=True, room="UI_1")
    elif(msg['cmd'] == 'wiselon'):
        setgamesaved(False)
        koboldai_vars.worldinfo[msg['data']]["selective"] = True
        emit('from_server', {'cmd': 'wiselon', 'data': msg['data']}, broadcast=True, room="UI_1")
    elif(msg['cmd'] == 'wiseloff'):
        setgamesaved(False)
        koboldai_vars.worldinfo[msg['data']]["selective"] = False
        emit('from_server', {'cmd': 'wiseloff', 'data': msg['data']}, broadcast=True, room="UI_1")
    elif(msg['cmd'] == 'wiconstanton'):
        setgamesaved(False)
        koboldai_vars.worldinfo[msg['data']]["constant"] = True
        emit('from_server', {'cmd': 'wiconstanton', 'data': msg['data']}, broadcast=True, room="UI_1")
    elif(msg['cmd'] == 'wiconstantoff'):
        setgamesaved(False)
        koboldai_vars.worldinfo[msg['data']]["constant"] = False
        emit('from_server', {'cmd': 'wiconstantoff', 'data': msg['data']}, broadcast=True, room="UI_1")
    elif(msg['cmd'] == 'sendwilist'):
        commitwi(msg['data'])
    elif(msg['cmd'] == 'aidgimport'):
        importAidgRequest(msg['data'])
    elif(msg['cmd'] == 'saveasrequest'):
        saveas(msg['data'])
    elif(msg['cmd'] == 'saverequest'):
        save()
    elif(msg['cmd'] == 'loadlistrequest'):
        getloadlist()
    elif(msg['cmd'] == 'splistrequest'):
        getsplist()
    elif(msg['cmd'] == 'uslistrequest'):
        unloaded, loaded = getuslist()
        emit('from_server', {'cmd': 'buildus', 'data': {"unloaded": unloaded, "loaded": loaded}}, room="UI_1")
    elif(msg['cmd'] == 'samplerlistrequest'):
        emit('from_server', {'cmd': 'buildsamplers', 'data': koboldai_vars.sampler_order}, room="UI_1")
    elif(msg['cmd'] == 'usloaded'):
        koboldai_vars.userscripts = []
        for userscript in msg['data']:
            if type(userscript) is not str:
                continue
            userscript = userscript.strip()
            if len(userscript) != 0 and all(q not in userscript for q in ("..", ":")) and all(userscript[0] not in q for q in ("/", "\\")) and os.path.exists(fileops.uspath(userscript)):
                koboldai_vars.userscripts.append(userscript)
        settingschanged()
    elif(msg['cmd'] == 'usload'):
        load_lua_scripts()
        unloaded, loaded = getuslist()
        sendUSStatItems()
    elif(msg['cmd'] == 'samplers'):
        sampler_order = msg["data"]
        if(not isinstance(sampler_order, list)):
            raise ValueError(f"Sampler order must be a list, but got a {type(sampler_order)}")
        if(len(sampler_order) != len(koboldai_vars.sampler_order)):
            raise ValueError(f"Sampler order must be a list of length {len(koboldai_vars.sampler_order)}, but got a list of length {len(sampler_order)}")
        if(not all(isinstance(e, int) for e in sampler_order)):
            raise ValueError(f"Sampler order must be a list of ints, but got a list with at least one non-int element")
        koboldai_vars.sampler_order = sampler_order
        settingschanged()
    elif(msg['cmd'] == 'list_model'):
        sendModelSelection(menu=msg['data'])
    elif(msg['cmd'] == 'load_model'):
        if not os.path.exists("settings/"):
            os.mkdir("settings")
        changed = True
        if not utils.HAS_ACCELERATE:
            msg['disk_layers'] = "0"
        if os.path.exists("settings/" + koboldai_vars.model.replace('/', '_') + ".breakmodel"):
            with open("settings/" + koboldai_vars.model.replace('/', '_') + ".breakmodel", "r") as file:
                data = file.read().split('\n')[:2]
                if len(data) < 2:
                    data.append("0")
                gpu_layers, disk_layers = data
                if gpu_layers == msg['gpu_layers'] and disk_layers == msg['disk_layers']:
                    changed = False
        if changed:
            f = open("settings/" + koboldai_vars.model.replace('/', '_') + ".breakmodel", "w")
            f.write(msg['gpu_layers'] + '\n' + msg['disk_layers'])
            f.close()
        koboldai_vars.colaburl = msg['url'] + "/request"
        load_model(use_gpu=msg['use_gpu'], gpu_layers=msg['gpu_layers'], disk_layers=msg['disk_layers'], online_model=msg['online_model'])
    elif(msg['cmd'] == 'show_model'):
        print("Model Name: {}".format(getmodelname()))
        emit('from_server', {'cmd': 'show_model_name', 'data': getmodelname()}, broadcast=True, room="UI_1")
    elif(msg['cmd'] == 'selectmodel'):
        # This is run when a model line is selected from the UI (line from the model_menu variable) that is tagged as not a menu
        # otherwise we should be running the msg['cmd'] == 'list_model'
        
        # We have to do a bit of processing though, if we select a custom path, we need to list out the contents of folders
        # But if we select something else, we need to potentially show model layers for each GPU
        # We might also need to show key input. All of that happens here
        
        # The data variable will contain the model name. But our Custom lines need a bit more processing
        # If we're on a custom line that we have selected a model for, the path variable will be in msg
        # so if that's missing we need to run the menu to show the model folders in the models folder
        if msg['data'] in ('NeoCustom', 'GPT2Custom') and 'path' not in msg and 'path_modelname' not in msg:
            if 'folder' not in msg or koboldai_vars.host:
                folder = "./models"
            else:
                folder = msg['folder']
            sendModelSelection(menu=msg['data'], folder=folder)
        elif msg['data'] in ('NeoCustom', 'GPT2Custom') and 'path_modelname' in msg:
            #Here the user entered custom text in the text box. This could be either a model name or a path.
            if check_if_dir_is_model(msg['path_modelname']):
                koboldai_vars.model = msg['data']
                koboldai_vars.custmodpth = msg['path_modelname']
                get_model_info(msg['data'], directory=msg['path'])
            else:
                koboldai_vars.model = msg['path_modelname']
                try:
                    get_model_info(koboldai_vars.model)
                except:
                    emit('from_server', {'cmd': 'errmsg', 'data': "The model entered doesn't exist."}, room="UI_1")
        elif msg['data'] in ('NeoCustom', 'GPT2Custom'):
            if check_if_dir_is_model(msg['path']):
                koboldai_vars.model = msg['data']
                koboldai_vars.custmodpth = msg['path']
                get_model_info(msg['data'], directory=msg['path'])
            else:
                if koboldai_vars.host:
                    sendModelSelection(menu=msg['data'], folder="./models")
                else:
                    sendModelSelection(menu=msg['data'], folder=msg['path'])
        else:
            koboldai_vars.model = msg['data']
            if 'path' in msg:
                koboldai_vars.custmodpth = msg['path']
                get_model_info(msg['data'], directory=msg['path'])
            else:
                get_model_info(koboldai_vars.model)
    elif(msg['cmd'] == 'delete_model'):
        if "{}/models".format(os.getcwd()) in os.path.abspath(msg['data']) or "{}\\models".format(os.getcwd()) in os.path.abspath(msg['data']):
            if check_if_dir_is_model(msg['data']):
                print(colors.YELLOW + "WARNING: Someone deleted " + msg['data'])
                import shutil
                shutil.rmtree(msg['data'])
                sendModelSelection(menu=msg['menu'])
            else:
                print(colors.RED + "ERROR: Someone attempted to delete " + msg['data'] + " but this is not a valid model")
        else:
            print(colors.RED + "WARNING!!: Someone maliciously attempted to delete " + msg['data'] + " the attempt has been blocked.")
    elif(msg['cmd'] == 'OAI_Key_Update'):
        get_oai_models({'model': koboldai_vars.model, 'key': msg['key']})
    elif(msg['cmd'] == 'loadselect'):
        koboldai_vars.loadselect = msg["data"]
    elif(msg['cmd'] == 'spselect'):
        koboldai_vars.spselect = msg["data"]
    elif(msg['cmd'] == 'loadrequest'):
        loadRequest(fileops.storypath(koboldai_vars.loadselect))
    elif(msg['cmd'] == 'sprequest'):
        spRequest(koboldai_vars.spselect)
    elif(msg['cmd'] == 'deletestory'):
        deletesave(msg['data'])
    elif(msg['cmd'] == 'renamestory'):
        renamesave(msg['data'], msg['newname'])
    elif(msg['cmd'] == 'clearoverwrite'):    
        koboldai_vars.svowname = ""
        koboldai_vars.saveow   = False
    elif(msg['cmd'] == 'seqsel'):
        selectsequence(msg['data'])
    elif(msg['cmd'] == 'seqpin'):
        pinsequence(msg['data'])
    elif(msg['cmd'] == 'setnumseq'):
        koboldai_vars.numseqs = int(msg['data'])
        emit('from_server', {'cmd': 'setlabelnumseq', 'data': msg['data']}, room="UI_1")
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setwidepth'):
        koboldai_vars.widepth = int(msg['data'])
        emit('from_server', {'cmd': 'setlabelwidepth', 'data': msg['data']}, room="UI_1")
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setuseprompt'):
        koboldai_vars.useprompt = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setadventure'):
        koboldai_vars.adventure = msg['data']
        koboldai_vars.chatmode = False
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'autosave'):
        koboldai_vars.autosave = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setchatmode'):
        koboldai_vars.chatmode = msg['data']
        koboldai_vars.adventure = False
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setdynamicscan'):
        koboldai_vars.dynamicscan = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setnopromptgen'):
        koboldai_vars.nopromptgen = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setrngpersist'):
        koboldai_vars.rngpersist = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setnogenmod'):
        koboldai_vars.nogenmod = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setoutputstreaming'):
        koboldai_vars.output_streaming = msg['data']
        settingschanged()
        refresh_settings()
    elif(not koboldai_vars.host and msg['cmd'] == 'importwi'):
        wiimportrequest()
    elif(msg['cmd'] == 'debug'):
        koboldai_vars.debug = msg['data']
        emit('from_server', {'cmd': 'set_debug', 'data': msg['data']}, broadcast=True, room="UI_1")
        if koboldai_vars.debug:
            send_debug()

#==================================================================#
#  Send userscripts list to client
#==================================================================#
def sendUSStatItems():
    _, loaded = getuslist()
    loaded = loaded if koboldai_vars.lua_running else []
    last_userscripts = [e["filename"] for e in loaded]
    emit('from_server', {'cmd': 'usstatitems', 'data': loaded, 'flash': last_userscripts != koboldai_vars.last_userscripts}, broadcast=True, room="UI_1")
    koboldai_vars.last_userscripts = last_userscripts

#==================================================================#
#  KoboldAI Markup Formatting (Mixture of Markdown and sanitized html)
#==================================================================#
def kml(txt):
   txt = txt.replace('>', '&gt;')
   txt = bleach.clean(markdown.markdown(txt), tags = ['p', 'em', 'strong', 'code', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'ul', 'b', 'i', 'a', 'span', 'button'], styles = ['color', 'font-weight'], attributes=['id', 'class', 'style', 'href'])
   return txt

#==================================================================#
#  Send start message and tell Javascript to set UI state
#==================================================================#
def setStartState():
    if(koboldai_vars.welcome):
        txt = kml(koboldai_vars.welcome) + "<br/>"
    else:
        txt = "<span>Welcome to <span class=\"color_cyan\">KoboldAI</span>! You are running <span class=\"color_green\">"+getmodelname()+"</span>.<br/>"
    if(not koboldai_vars.noai and not koboldai_vars.welcome):
        txt = txt + "Please load a game or enter a prompt below to begin!</span>"
    if(koboldai_vars.noai):
        txt = txt + "Please load or import a story to read. There is no AI in this mode."
    emit('from_server', {'cmd': 'updatescreen', 'gamestarted': koboldai_vars.gamestarted, 'data': txt}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'setgamestate', 'data': 'start'}, broadcast=True, room="UI_1")

#==================================================================#
#  Transmit applicable settings to SocketIO to build UI sliders/toggles
#==================================================================#
def sendsettings():
    # Send settings for selected AI type
    emit('from_server', {'cmd': 'reset_menus'}, room="UI_1")
    if(koboldai_vars.model != "InferKit"):
        for set in gensettings.gensettingstf:
            emit('from_server', {'cmd': 'addsetting', 'data': set}, room="UI_1")
    else:
        for set in gensettings.gensettingsik:
            emit('from_server', {'cmd': 'addsetting', 'data': set}, room="UI_1")
    
    # Send formatting options
    for frm in gensettings.formatcontrols:
        emit('from_server', {'cmd': 'addformat', 'data': frm}, room="UI_1")
        # Add format key to vars if it wasn't loaded with client.settings
        if(not frm["id"] in koboldai_vars.formatoptns):
            koboldai_vars.formatoptns[frm["id"]] = False;

#==================================================================#
#  Set value of gamesaved
#==================================================================#
def setgamesaved(gamesaved):
    assert type(gamesaved) is bool
    if(gamesaved != koboldai_vars.gamesaved):
        emit('from_server', {'cmd': 'gamesaved', 'data': gamesaved}, broadcast=True, room="UI_1")
    koboldai_vars.gamesaved = gamesaved

#==================================================================#
#  Take input text from SocketIO and decide what to do with it
#==================================================================#

def check_for_backend_compilation():
    if(koboldai_vars.checking):
        return
    koboldai_vars.checking = True
    for _ in range(31):
        time.sleep(0.06276680299820175)
        if(koboldai_vars.compiling):
            emit('from_server', {'cmd': 'warnmsg', 'data': 'Compiling TPU backend&mdash;this usually takes 1&ndash;2 minutes...'}, broadcast=True, room="UI_1")
            break
    koboldai_vars.checking = False

def actionsubmit(data, actionmode=0, force_submit=False, force_prompt_gen=False, disable_recentrng=False):
    # Ignore new submissions if the AI is currently busy
    if(koboldai_vars.aibusy):
        return
    
    while(True):
        set_aibusy(1)

        if(disable_recentrng):
            koboldai_vars.recentrng = koboldai_vars.recentrngm = None

        koboldai_vars.recentback = False
        koboldai_vars.recentedit = False
        koboldai_vars.actionmode = actionmode

        # "Action" mode
        if(actionmode == 1):
            data = data.strip().lstrip('>')
            data = re.sub(r'\n+', ' ', data)
            if(len(data)):
                data = f"\n\n> {data}\n"
        
        # "Chat" mode
        if(koboldai_vars.chatmode and koboldai_vars.gamestarted):
            data = re.sub(r'\n+', ' ', data)
            if(len(data)):
                data = f"\n{koboldai_vars.chatname}: {data}\n"
        
        # If we're not continuing, store a copy of the raw input
        if(data != ""):
            koboldai_vars.lastact = data
        
        if(not koboldai_vars.gamestarted):
            koboldai_vars.submission = data
            execute_inmod()
            data = koboldai_vars.submission
            if(not force_submit and len(data.strip()) == 0):
                assert False
            # Start the game
            koboldai_vars.gamestarted = True
            if(not koboldai_vars.noai and koboldai_vars.lua_koboldbridge.generating and (not koboldai_vars.nopromptgen or force_prompt_gen)):
                # Save this first action as the prompt
                koboldai_vars.prompt = data
                # Clear the startup text from game screen
                emit('from_server', {'cmd': 'updatescreen', 'gamestarted': False, 'data': 'Please wait, generating story...'}, broadcast=True, room="UI_1")
                calcsubmit(data) # Run the first action through the generator
                if(not koboldai_vars.abort and koboldai_vars.lua_koboldbridge.restart_sequence is not None and len(koboldai_vars.genseqs) == 0):
                    data = ""
                    force_submit = True
                    disable_recentrng = True
                    continue
                emit('from_server', {'cmd': 'scrolldown', 'data': ''}, broadcast=True, room="UI_1")
                break
            else:
                # Save this first action as the prompt
                koboldai_vars.prompt = data if len(data) > 0 else '"'
                for i in range(koboldai_vars.numseqs):
                    koboldai_vars.lua_koboldbridge.outputs[i+1] = ""
                execute_outmod()
                koboldai_vars.lua_koboldbridge.regeneration_required = False
                genout = []
                for i in range(koboldai_vars.numseqs):
                    genout.append({"generated_text": koboldai_vars.lua_koboldbridge.outputs[i+1]})
                    assert type(genout[-1]["generated_text"]) is str
                koboldai_vars.actions.clear_unused_options()
                koboldai_vars.actions.append_options([x["generated_text"] for x in genout])
                genout = [{"generated_text": x['text']} for x in koboldai_vars.actions.get_current_options()]
                if(len(genout) == 1):
                    genresult(genout[0]["generated_text"], flash=False)
                    refresh_story()
                    if(len(koboldai_vars.actions) > 0):
                        emit('from_server', {'cmd': 'texteffect', 'data': koboldai_vars.actions.get_last_key() + 1}, broadcast=True, room="UI_1")
                    if(not koboldai_vars.abort and koboldai_vars.lua_koboldbridge.restart_sequence is not None):
                        data = ""
                        force_submit = True
                        disable_recentrng = True
                        continue
                else:
                    if(not koboldai_vars.abort and koboldai_vars.lua_koboldbridge.restart_sequence is not None and koboldai_vars.lua_koboldbridge.restart_sequence > 0):
                        genresult(genout[koboldai_vars.lua_koboldbridge.restart_sequence-1]["generated_text"], flash=False)
                        refresh_story()
                        data = ""
                        force_submit = True
                        disable_recentrng = True
                        continue
                    genselect(genout)
                    refresh_story()
                set_aibusy(0)
                emit('from_server', {'cmd': 'scrolldown', 'data': ''}, broadcast=True, room="UI_1")
                break
        else:
            # Apply input formatting & scripts before sending to tokenizer
            if(koboldai_vars.actionmode == 0):
                data = applyinputformatting(data)
            koboldai_vars.submission = data
            execute_inmod()
            data = koboldai_vars.submission
            # Dont append submission if it's a blank/continue action
            if(data != ""):
                # Store the result in the Action log
                if(len(koboldai_vars.prompt.strip()) == 0):
                    koboldai_vars.prompt = data
                else:
                    koboldai_vars.actions.append(data)
                update_story_chunk('last')
                send_debug()

            if(not koboldai_vars.noai and koboldai_vars.lua_koboldbridge.generating):
                # Off to the tokenizer!
                calcsubmit(data)
                if(not koboldai_vars.abort and koboldai_vars.lua_koboldbridge.restart_sequence is not None and len(koboldai_vars.genseqs) == 0):
                    data = ""
                    force_submit = True
                    disable_recentrng = True
                    continue
                emit('from_server', {'cmd': 'scrolldown', 'data': ''}, broadcast=True, room="UI_1")
                break
            else:
                for i in range(koboldai_vars.numseqs):
                    koboldai_vars.lua_koboldbridge.outputs[i+1] = ""
                execute_outmod()
                koboldai_vars.lua_koboldbridge.regeneration_required = False
                genout = []
                for i in range(koboldai_vars.numseqs):
                    genout.append({"generated_text": koboldai_vars.lua_koboldbridge.outputs[i+1]})
                    assert type(genout[-1]["generated_text"]) is str
                koboldai_vars.actions.clear_unused_options()
                koboldai_vars.actions.append_options([x["generated_text"] for x in genout])
                genout = [{"generated_text": x['text']} for x in koboldai_vars.actions.get_current_options()]
                if(len(genout) == 1):
                    genresult(genout[0]["generated_text"])
                    if(not koboldai_vars.abort and koboldai_vars.lua_koboldbridge.restart_sequence is not None):
                        data = ""
                        force_submit = True
                        disable_recentrng = True
                        continue
                else:
                    if(not koboldai_vars.abort and koboldai_vars.lua_koboldbridge.restart_sequence is not None and koboldai_vars.lua_koboldbridge.restart_sequence > 0):
                        genresult(genout[koboldai_vars.lua_koboldbridge.restart_sequence-1]["generated_text"])
                        data = ""
                        force_submit = True
                        disable_recentrng = True
                        continue
                    genselect(genout)
                set_aibusy(0)
                emit('from_server', {'cmd': 'scrolldown', 'data': ''}, broadcast=True, room="UI_1")
                break

#==================================================================#
#  
#==================================================================#
def actionretry(data):
    if(koboldai_vars.noai):
        emit('from_server', {'cmd': 'errmsg', 'data': "Retry function unavailable in Read Only mode."}, room="UI_1")
        return
    if(koboldai_vars.recentrng is not None):
        if(not koboldai_vars.aibusy):
            randomGameRequest(koboldai_vars.recentrng, memory=koboldai_vars.recentrngm)
        return
    if actionback():
        actionsubmit("", actionmode=koboldai_vars.actionmode, force_submit=True)
        send_debug()
    elif(not koboldai_vars.useprompt):
        emit('from_server', {'cmd': 'errmsg', 'data': "Please enable \"Always Add Prompt\" to retry with your prompt."}, room="UI_1")

#==================================================================#
#  
#==================================================================#
def actionback():
    if(koboldai_vars.aibusy):
        return
    # Remove last index of actions and refresh game screen
    if(len(koboldai_vars.genseqs) == 0 and len(koboldai_vars.actions) > 0):
        last_key = koboldai_vars.actions.get_last_key()
        koboldai_vars.actions.pop()
        koboldai_vars.recentback = True
        remove_story_chunk(last_key + 1)
        success = True
    elif(len(koboldai_vars.genseqs) == 0):
        emit('from_server', {'cmd': 'errmsg', 'data': "Cannot delete the prompt."}, room="UI_1")
        success =  False
    else:
        koboldai_vars.genseqs = []
        success = True
    send_debug()
    return success
        
def actionredo():
    genout = [[x['text'], "redo" if x['Previous Selection'] else "pinned" if x['Pinned'] else "normal"] for x in koboldai_vars.actions.get_redo_options()]
    if len(genout) == 0:
        emit('from_server', {'cmd': 'popuperror', 'data': "There's nothing to redo"}, broadcast=True, room="UI_1")
    elif len(genout) == 1:
        genresult(genout[0][0], flash=True, ignore_formatting=True)
    else:
        koboldai_vars.genseqs = [{"generated_text": x[0]} for x in genout]
        emit('from_server', {'cmd': 'genseqs', 'data': genout}, broadcast=True, room="UI_1")
    send_debug()

#==================================================================#
#  
#==================================================================#
def calcsubmitbudgetheader(txt, **kwargs):
    # Scan for WorldInfo matches
    winfo, found_entries = checkworldinfo(txt, **kwargs)

    # Add a newline to the end of memory
    if(koboldai_vars.memory != "" and koboldai_vars.memory[-1] != "\n"):
        mem = koboldai_vars.memory + "\n"
    else:
        mem = koboldai_vars.memory

    # Build Author's Note if set
    if(koboldai_vars.authornote != ""):
        anotetxt  = ("\n" + koboldai_vars.authornotetemplate + "\n").replace("<|>", koboldai_vars.authornote)
    else:
        anotetxt = ""

    return winfo, mem, anotetxt, found_entries

def calcsubmitbudget(actionlen, winfo, mem, anotetxt, actions, submission=None, budget_deduction=0):
    forceanote   = False # In case we don't have enough actions to hit A.N. depth
    anoteadded   = False # In case our budget runs out before we hit A.N. depth
    anotetkns    = []  # Placeholder for Author's Note tokens
    lnanote      = 0   # Placeholder for Author's Note length

    lnsp = koboldai_vars.sp_length

    if("tokenizer" not in globals()):
        from transformers import GPT2TokenizerFast
        global tokenizer
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", revision=koboldai_vars.revision, cache_dir="cache")

    lnheader = len(tokenizer._koboldai_header)

    # Calculate token budget
    prompttkns = tokenizer.encode(utils.encodenewlines(koboldai_vars.comregex_ai.sub('', koboldai_vars.prompt)), max_length=int(2e9), truncation=True)
    lnprompt   = len(prompttkns)

    memtokens = tokenizer.encode(utils.encodenewlines(mem), max_length=int(2e9), truncation=True)
    lnmem     = len(memtokens)
    if(lnmem > koboldai_vars.max_length - lnheader - lnsp - koboldai_vars.genamt - budget_deduction):
        raise OverflowError("The memory in your story is too long. Please either write a shorter memory text or increase the Max Tokens setting. If you are using a soft prompt, additionally consider using a smaller soft prompt.")

    witokens  = tokenizer.encode(utils.encodenewlines(winfo), max_length=int(2e9), truncation=True)
    lnwi      = len(witokens)
    if(lnmem + lnwi > koboldai_vars.max_length - lnheader - lnsp - koboldai_vars.genamt - budget_deduction):
        raise OverflowError("The current active world info keys take up too many tokens. Please either write shorter world info, decrease World Info Depth or increase the Max Tokens setting. If you are using a soft prompt, additionally consider using a smaller soft prompt.")

    if(anotetxt != ""):
        anotetkns = tokenizer.encode(utils.encodenewlines(anotetxt), max_length=int(2e9), truncation=True)
        lnanote   = len(anotetkns)
        if(lnmem + lnwi + lnanote > koboldai_vars.max_length - lnheader - lnsp - koboldai_vars.genamt - budget_deduction):
            raise OverflowError("The author's note in your story is too long. Please either write a shorter author's note or increase the Max Tokens setting. If you are using a soft prompt, additionally consider using a smaller soft prompt.")

    if(koboldai_vars.useprompt):
        budget = koboldai_vars.max_length - lnsp - lnprompt - lnmem - lnanote - lnwi - koboldai_vars.genamt - budget_deduction
    else:
        budget = koboldai_vars.max_length - lnsp - lnmem - lnanote - lnwi - koboldai_vars.genamt - budget_deduction

    lnsubmission = len(tokenizer.encode(utils.encodenewlines(koboldai_vars.comregex_ai.sub('', submission)), max_length=int(2e9), truncation=True)) if submission is not None else 0
    maybe_lnprompt = lnprompt if koboldai_vars.useprompt and actionlen > 0 else 0

    if(lnmem + lnwi + lnanote + maybe_lnprompt + lnsubmission > koboldai_vars.max_length - lnheader - lnsp - koboldai_vars.genamt - budget_deduction):
        raise OverflowError("Your submission is too long. Please either write a shorter submission or increase the Max Tokens setting. If you are using a soft prompt, additionally consider using a smaller soft prompt. If you are using the Always Add Prompt setting, turning it off may help.")

    assert budget >= 0

    if(actionlen == 0):
        # First/Prompt action
        tokens = tokenizer._koboldai_header + memtokens + witokens + anotetkns + prompttkns
        assert len(tokens) <= koboldai_vars.max_length - lnsp - koboldai_vars.genamt - budget_deduction
        ln = len(tokens) + lnsp
        return tokens, ln+1, ln+koboldai_vars.genamt
    else:
        tokens     = []
        
        # Check if we have the action depth to hit our A.N. depth
        if(anotetxt != "" and actionlen < koboldai_vars.andepth):
            forceanote = True
        
        # Get most recent action tokens up to our budget
        n = 0
        for key in reversed(actions):
            chunk = koboldai_vars.comregex_ai.sub('', actions[key])
            
            assert budget >= 0
            if(budget <= 0):
                break
            acttkns = tokenizer.encode(utils.encodenewlines(chunk), max_length=int(2e9), truncation=True)
            tknlen = len(acttkns)
            if(tknlen < budget):
                tokens = acttkns + tokens
                budget -= tknlen
            else:
                count = budget * -1
                tokens = acttkns[count:] + tokens
                budget = 0
                break
            
            # Inject Author's Note if we've reached the desired depth
            if(n == koboldai_vars.andepth-1):
                if(anotetxt != ""):
                    tokens = anotetkns + tokens # A.N. len already taken from bdgt
                    anoteadded = True
            n += 1
        
        # If we're not using the prompt every time and there's still budget left,
        # add some prompt.
        if(not koboldai_vars.useprompt):
            if(budget > 0):
                prompttkns = prompttkns[-budget:]
            else:
                prompttkns = []

        # Did we get to add the A.N.? If not, do it here
        if(anotetxt != ""):
            if((not anoteadded) or forceanote):
                tokens = tokenizer._koboldai_header + memtokens + witokens + anotetkns + prompttkns + tokens
            else:
                tokens = tokenizer._koboldai_header + memtokens + witokens + prompttkns + tokens
        else:
            # Prepend Memory, WI, and Prompt before action tokens
            tokens = tokenizer._koboldai_header + memtokens + witokens + prompttkns + tokens

        # Send completed bundle to generator
        assert len(tokens) <= koboldai_vars.max_length - lnsp - koboldai_vars.genamt - budget_deduction
        ln = len(tokens) + lnsp
        return tokens, ln+1, ln+koboldai_vars.genamt

#==================================================================#
# Take submitted text and build the text to be given to generator
#==================================================================#
def calcsubmit(txt):
    anotetxt     = ""    # Placeholder for Author's Note text
    forceanote   = False # In case we don't have enough actions to hit A.N. depth
    anoteadded   = False # In case our budget runs out before we hit A.N. depth
    actionlen    = len(koboldai_vars.actions)

    winfo, mem, anotetxt, found_entries = calcsubmitbudgetheader(txt)
 
    # For all transformers models
    if(koboldai_vars.model != "InferKit"):
        subtxt, min, max = calcsubmitbudget(actionlen, winfo, mem, anotetxt, koboldai_vars.actions, submission=txt)
        if(actionlen == 0):
            if(not koboldai_vars.use_colab_tpu and koboldai_vars.model not in ["Colab", "OAI", "TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX"]):
                generate(subtxt, min, max, found_entries=found_entries)
            elif(koboldai_vars.model == "Colab"):
                sendtocolab(utils.decodenewlines(tokenizer.decode(subtxt)), min, max)
            elif(koboldai_vars.model == "OAI"):
                oairequest(utils.decodenewlines(tokenizer.decode(subtxt)), min, max)
            elif(koboldai_vars.use_colab_tpu or koboldai_vars.model in ("TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX")):
                tpumtjgenerate(subtxt, min, max, found_entries=found_entries)
        else:
            if(not koboldai_vars.use_colab_tpu and koboldai_vars.model not in ["Colab", "OAI", "TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX"]):
                generate(subtxt, min, max, found_entries=found_entries)
            elif(koboldai_vars.model == "Colab"):
                sendtocolab(utils.decodenewlines(tokenizer.decode(subtxt)), min, max)
            elif(koboldai_vars.model == "OAI"):
                oairequest(utils.decodenewlines(tokenizer.decode(subtxt)), min, max)
            elif(koboldai_vars.use_colab_tpu or koboldai_vars.model in ("TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX")):
                tpumtjgenerate(subtxt, min, max, found_entries=found_entries)
                    
    # For InferKit web API
    else:
        # Check if we have the action depth to hit our A.N. depth
        if(anotetxt != "" and actionlen < koboldai_vars.andepth):
            forceanote = True
        
        if(koboldai_vars.useprompt):
            budget = koboldai_vars.ikmax - len(koboldai_vars.comregex_ai.sub('', koboldai_vars.prompt)) - len(anotetxt) - len(mem) - len(winfo) - 1
        else:
            budget = koboldai_vars.ikmax - len(anotetxt) - len(mem) - len(winfo) - 1
            
        subtxt = ""
        prompt = koboldai_vars.comregex_ai.sub('', koboldai_vars.prompt)
        n = 0
        for key in reversed(koboldai_vars.actions):
            chunk = koboldai_vars.actions[key]
            
            if(budget <= 0):
                    break
            actlen = len(chunk)
            if(actlen < budget):
                subtxt = chunk + subtxt
                budget -= actlen
            else:
                count = budget * -1
                subtxt = chunk[count:] + subtxt
                budget = 0
                break
            
            # If we're not using the prompt every time and there's still budget left,
            # add some prompt.
            if(not koboldai_vars.useprompt):
                if(budget > 0):
                    prompt = koboldai_vars.comregex_ai.sub('', koboldai_vars.prompt)[-budget:]
                else:
                    prompt = ""
            
            # Inject Author's Note if we've reached the desired depth
            if(n == koboldai_vars.andepth-1):
                if(anotetxt != ""):
                    subtxt = anotetxt + subtxt # A.N. len already taken from bdgt
                    anoteadded = True
            n += 1
        
        # Did we get to add the A.N.? If not, do it here
        if(anotetxt != ""):
            if((not anoteadded) or forceanote):
                subtxt = mem + winfo + anotetxt + prompt + subtxt
            else:
                subtxt = mem + winfo + prompt + subtxt
        else:
            subtxt = mem + winfo + prompt + subtxt
        
        # Send it!
        ikrequest(subtxt)

#==================================================================#
# Send text to generator and deal with output
#==================================================================#

def _generate(txt, minimum, maximum, found_entries):
    gen_in = torch.tensor(txt, dtype=torch.long)[None]
    if(koboldai_vars.sp is not None):
        soft_tokens = torch.arange(
            model.config.vocab_size,
            model.config.vocab_size + koboldai_vars.sp.shape[0],
        )
        gen_in = torch.cat((soft_tokens[None], gen_in), dim=-1)
    assert gen_in.shape[-1] + koboldai_vars.genamt <= koboldai_vars.max_length

    if(koboldai_vars.hascuda and koboldai_vars.usegpu):
        gen_in = gen_in.to(koboldai_vars.gpu_device)
    elif(koboldai_vars.hascuda and koboldai_vars.breakmodel):
        gen_in = gen_in.to(breakmodel.primary_device)
    else:
        gen_in = gen_in.to('cpu')

    model.kai_scanner_excluded_world_info = found_entries

    koboldai_vars._actions = koboldai_vars.actions
    koboldai_vars._prompt = koboldai_vars.prompt
    if(koboldai_vars.dynamicscan):
        koboldai_vars._actions = koboldai_vars._actions.copy()

    with torch.no_grad():
        already_generated = 0
        numseqs = koboldai_vars.numseqs
        while True:
            genout = generator(
                gen_in, 
                do_sample=True, 
                max_length=int(2e9),
                repetition_penalty=1.1,
                bad_words_ids=koboldai_vars.badwordsids,
                use_cache=True,
                num_return_sequences=numseqs
                )
            already_generated += len(genout[0]) - len(gen_in[0])
            assert already_generated <= koboldai_vars.genamt
            if(model.kai_scanner.halt or not model.kai_scanner.regeneration_required):
                break
            assert genout.ndim >= 2
            assert genout.shape[0] == koboldai_vars.numseqs
            if(koboldai_vars.lua_koboldbridge.generated_cols and koboldai_vars.generated_tkns != koboldai_vars.lua_koboldbridge.generated_cols):
                raise RuntimeError("Inconsistency detected between KoboldAI Python and Lua backends")
            if(already_generated != koboldai_vars.generated_tkns):
                raise RuntimeError("WI scanning error")
            for r in range(koboldai_vars.numseqs):
                for c in range(already_generated):
                    assert koboldai_vars.lua_koboldbridge.generated[r+1][c+1] is not None
                    genout[r][genout.shape[-1] - already_generated + c] = koboldai_vars.lua_koboldbridge.generated[r+1][c+1]
            encoded = []
            for i in range(koboldai_vars.numseqs):
                txt = utils.decodenewlines(tokenizer.decode(genout[i, -already_generated:]))
                winfo, mem, anotetxt, _found_entries = calcsubmitbudgetheader(txt, force_use_txt=True, actions=koboldai_vars._actions)
                found_entries[i].update(_found_entries)
                txt, _, _ = calcsubmitbudget(len(koboldai_vars._actions), winfo, mem, anotetxt, koboldai_vars._actions, submission=txt)
                encoded.append(torch.tensor(txt, dtype=torch.long, device=genout.device))
            max_length = len(max(encoded, key=len))
            encoded = torch.stack(tuple(torch.nn.functional.pad(e, (max_length - len(e), 0), value=model.config.pad_token_id or model.config.eos_token_id) for e in encoded))
            genout = torch.cat(
                (
                    encoded,
                    genout[..., -already_generated:],
                ),
                dim=-1
            )
            if(koboldai_vars.sp is not None):
                soft_tokens = torch.arange(
                    model.config.vocab_size,
                    model.config.vocab_size + koboldai_vars.sp.shape[0],
                    device=genout.device,
                )
                genout = torch.cat((soft_tokens.tile(koboldai_vars.numseqs, 1), genout), dim=-1)
            assert genout.shape[-1] + koboldai_vars.genamt - already_generated <= koboldai_vars.max_length
            diff = genout.shape[-1] - gen_in.shape[-1]
            minimum += diff
            maximum += diff
            gen_in = genout
            numseqs = 1
    
    return genout, already_generated
    

def generate(txt, minimum, maximum, found_entries=None):    
    koboldai_vars.generated_tkns = 0

    if(found_entries is None):
        found_entries = set()
    found_entries = tuple(found_entries.copy() for _ in range(koboldai_vars.numseqs))

    if not koboldai_vars.quiet:
        print("{0}Min:{1}, Max:{2}, Txt:{3}{4}".format(colors.YELLOW, minimum, maximum, utils.decodenewlines(tokenizer.decode(txt)), colors.END))

    # Store context in memory to use it for comparison with generated content
    koboldai_vars.lastctx = utils.decodenewlines(tokenizer.decode(txt))

    # Clear CUDA cache if using GPU
    if(koboldai_vars.hascuda and (koboldai_vars.usegpu or koboldai_vars.breakmodel)):
        gc.collect()
        torch.cuda.empty_cache()

    # Submit input text to generator
    try:
        genout, already_generated = tpool.execute(_generate, txt, minimum, maximum, found_entries)
    except Exception as e:
        if(issubclass(type(e), lupa.LuaError)):
            koboldai_vars.lua_koboldbridge.obliterate_multiverse()
            koboldai_vars.lua_running = False
            emit('from_server', {'cmd': 'errmsg', 'data': 'Lua script error; please check console.'}, broadcast=True, room="UI_1")
            sendUSStatItems()
            print("{0}{1}{2}".format(colors.RED, "***LUA ERROR***: ", colors.END), end="", file=sys.stderr)
            print("{0}{1}{2}".format(colors.RED, str(e).replace("\033", ""), colors.END), file=sys.stderr)
            print("{0}{1}{2}".format(colors.YELLOW, "Lua engine stopped; please open 'Userscripts' and press Load to reinitialize scripts.", colors.END), file=sys.stderr)
        else:
            emit('from_server', {'cmd': 'errmsg', 'data': 'Error occurred during generator call; please check console.'}, broadcast=True, room="UI_1")
            print("{0}{1}{2}".format(colors.RED, traceback.format_exc().replace("\033", ""), colors.END), file=sys.stderr)
        set_aibusy(0)
        return

    for i in range(koboldai_vars.numseqs):
        koboldai_vars.lua_koboldbridge.generated[i+1][koboldai_vars.generated_tkns] = int(genout[i, -1].item())
        koboldai_vars.lua_koboldbridge.outputs[i+1] = utils.decodenewlines(tokenizer.decode(genout[i, -already_generated:]))

    execute_outmod()
    if(koboldai_vars.lua_koboldbridge.regeneration_required):
        koboldai_vars.lua_koboldbridge.regeneration_required = False
        genout = []
        for i in range(koboldai_vars.numseqs):
            genout.append({"generated_text": koboldai_vars.lua_koboldbridge.outputs[i+1]})
            assert type(genout[-1]["generated_text"]) is str
    else:
        genout = [{"generated_text": utils.decodenewlines(tokenizer.decode(tokens[-already_generated:]))} for tokens in genout]
    
    koboldai_vars.actions.clear_unused_options()
    koboldai_vars.actions.append_options([x["generated_text"] for x in genout])
    genout = [{"generated_text": x['text']} for x in koboldai_vars.actions.get_current_options()]
    if(len(genout) == 1):
        genresult(genout[0]["generated_text"])
    else:
        if(koboldai_vars.lua_koboldbridge.restart_sequence is not None and koboldai_vars.lua_koboldbridge.restart_sequence > 0):
            genresult(genout[koboldai_vars.lua_koboldbridge.restart_sequence-1]["generated_text"])
        else:
            genselect(genout)
    
    # Clear CUDA cache again if using GPU
    if(koboldai_vars.hascuda and (koboldai_vars.usegpu or koboldai_vars.breakmodel)):
        del genout
        gc.collect()
        torch.cuda.empty_cache()
    
    set_aibusy(0)

#==================================================================#
#  Deal with a single return sequence from generate()
#==================================================================#
def genresult(genout, flash=True, ignore_formatting=False):
    if not koboldai_vars.quiet:
        print("{0}{1}{2}".format(colors.CYAN, genout, colors.END))
    
    # Format output before continuing
    if not ignore_formatting:
        genout = applyoutputformatting(genout)

    koboldai_vars.lua_koboldbridge.feedback = genout

    if(len(genout) == 0):
        return
    
    # Add formatted text to Actions array and refresh the game screen
    if(len(koboldai_vars.prompt.strip()) == 0):
        koboldai_vars.prompt = genout
    else:
        koboldai_vars.actions.append(genout)
    update_story_chunk('last')
    if(flash):
        emit('from_server', {'cmd': 'texteffect', 'data': koboldai_vars.actions.get_last_key() + 1 if len(koboldai_vars.actions) else 0}, broadcast=True, room="UI_1")
    send_debug()

#==================================================================#
#  Send generator sequences to the UI for selection
#==================================================================#
def genselect(genout):
    i = 0
    for result in genout:
        # Apply output formatting rules to sequences
        result["generated_text"] = applyoutputformatting(result["generated_text"])
        if not koboldai_vars.quiet:
            print("{0}[Result {1}]\n{2}{3}".format(colors.CYAN, i, result["generated_text"], colors.END))
        i += 1
    
    
    # Store sequences in memory until selection is made
    koboldai_vars.genseqs = genout
    
    
    genout = koboldai_vars.actions.get_current_options_no_edits(ui=1)

    # Send sequences to UI for selection
    emit('from_server', {'cmd': 'genseqs', 'data': genout}, broadcast=True, room="UI_1")
    send_debug()

#==================================================================#
#  Send selected sequence to action log and refresh UI
#==================================================================#
def selectsequence(n):
    if(len(koboldai_vars.genseqs) == 0):
        return
    koboldai_vars.lua_koboldbridge.feedback = koboldai_vars.genseqs[int(n)]["generated_text"]
    if(len(koboldai_vars.lua_koboldbridge.feedback) != 0):
        koboldai_vars.actions.append(koboldai_vars.lua_koboldbridge.feedback)
        update_story_chunk('last')
        emit('from_server', {'cmd': 'texteffect', 'data': koboldai_vars.actions.get_last_key() + 1 if len(koboldai_vars.actions) else 0}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'hidegenseqs', 'data': ''}, broadcast=True, room="UI_1")
    koboldai_vars.genseqs = []

    if(koboldai_vars.lua_koboldbridge.restart_sequence is not None):
        actionsubmit("", actionmode=koboldai_vars.actionmode, force_submit=True, disable_recentrng=True)
    send_debug()

#==================================================================#
#  Pin/Unpin the selected sequence
#==================================================================#
def pinsequence(n):
    if n.isnumeric():
        koboldai_vars.actions.toggle_pin(koboldai_vars.actions.get_last_key()+1, int(n))
        text = koboldai_vars.genseqs[int(n)]['generated_text']
    send_debug()


#==================================================================#
#  Send transformers-style request to ngrok/colab host
#==================================================================#
def sendtocolab(txt, min, max):
    # Log request to console
    if not koboldai_vars.quiet:
        print("{0}Tokens:{1}, Txt:{2}{3}".format(colors.YELLOW, min-1, txt, colors.END))
    
    # Store context in memory to use it for comparison with generated content
    koboldai_vars.lastctx = txt
    
    # Build request JSON data
    reqdata = {
        'text': txt,
        'min': min,
        'max': max,
        'rep_pen': koboldai_vars.rep_pen,
        'rep_pen_slope': koboldai_vars.rep_pen_slope,
        'rep_pen_range': koboldai_vars.rep_pen_range,
        'temperature': koboldai_vars.temp,
        'top_p': koboldai_vars.top_p,
        'top_k': koboldai_vars.top_k,
        'tfs': koboldai_vars.tfs,
        'typical': koboldai_vars.typical,
        'topa': koboldai_vars.top_a,
        'numseqs': koboldai_vars.numseqs,
        'retfultxt': False
    }
    
    # Create request
    req = requests.post(
        koboldai_vars.colaburl, 
        json = reqdata
        )
    
    # Deal with the response
    if(req.status_code == 200):
        js = req.json()["data"]
        
        # Try to be backwards compatible with outdated colab
        if("text" in js):
            genout = [getnewcontent(js["text"])]
        else:
            genout = js["seqs"]
        
        for i in range(koboldai_vars.numseqs):
            koboldai_vars.lua_koboldbridge.outputs[i+1] = genout[i]

        execute_outmod()
        if(koboldai_vars.lua_koboldbridge.regeneration_required):
            koboldai_vars.lua_koboldbridge.regeneration_required = False
            genout = []
            for i in range(koboldai_vars.numseqs):
                genout.append(koboldai_vars.lua_koboldbridge.outputs[i+1])
                assert type(genout[-1]) is str

        koboldai_vars.actions.clear_unused_options()
        koboldai_vars.actions.append_options([x["generated_text"] for x in genout])
        genout = [{"generated_text": x['text']} for x in koboldai_vars.actions.get_current_options()]
        if(len(genout) == 1):
            
            genresult(genout[0])
        else:
            # Convert torch output format to transformers
            seqs = []
            for seq in genout:
                seqs.append({"generated_text": seq})
            if(koboldai_vars.lua_koboldbridge.restart_sequence is not None and koboldai_vars.lua_koboldbridge.restart_sequence > 0):
                genresult(genout[koboldai_vars.lua_koboldbridge.restart_sequence-1]["generated_text"])
            else:
                genselect(genout)
        
        # Format output before continuing
        #genout = applyoutputformatting(getnewcontent(genout))
        
        # Add formatted text to Actions array and refresh the game screen
        #koboldai_vars.actions.append(genout)
        #refresh_story()
        #emit('from_server', {'cmd': 'texteffect', 'data': koboldai_vars.actions.get_last_key() + 1 if len(koboldai_vars.actions) else 0})
        
        set_aibusy(0)
    else:
        errmsg = "Colab API Error: Failed to get a reply from the server. Please check the colab console."
        print("{0}{1}{2}".format(colors.RED, errmsg, colors.END))
        emit('from_server', {'cmd': 'errmsg', 'data': errmsg}, broadcast=True, room="UI_1")
        set_aibusy(0)

#==================================================================#
#  Send text to TPU mesh transformer backend
#==================================================================#
def tpumtjgenerate(txt, minimum, maximum, found_entries=None):
    koboldai_vars.generated_tkns = 0

    if(found_entries is None):
        found_entries = set()
    found_entries = tuple(found_entries.copy() for _ in range(koboldai_vars.numseqs))

    if not koboldai_vars.quiet:
        print("{0}Min:{1}, Max:{2}, Txt:{3}{4}".format(colors.YELLOW, minimum, maximum, utils.decodenewlines(tokenizer.decode(txt)), colors.END))

    koboldai_vars._actions = koboldai_vars.actions
    koboldai_vars._prompt = koboldai_vars.prompt
    if(koboldai_vars.dynamicscan):
        koboldai_vars._actions = koboldai_vars._actions.copy()

    # Submit input text to generator
    try:
        soft_tokens = tpumtjgetsofttokens()

        global past

        socketio.start_background_task(copy_current_request_context(check_for_backend_compilation))

        if(koboldai_vars.dynamicscan or (not koboldai_vars.nogenmod and koboldai_vars.has_genmod)):

            context = np.tile(np.uint32(txt), (koboldai_vars.numseqs, 1))
            past = np.empty((koboldai_vars.numseqs, 0), dtype=np.uint32)

            while(True):
                genout, n_generated, regeneration_required, halt = tpool.execute(
                    tpu_mtj_backend.infer_dynamic,
                    context,
                    gen_len = maximum-minimum+1,
                    numseqs=koboldai_vars.numseqs,
                    soft_embeddings=koboldai_vars.sp,
                    soft_tokens=soft_tokens,
                    excluded_world_info=found_entries,
                )

                past = np.pad(past, ((0, 0), (0, n_generated)))
                for r in range(koboldai_vars.numseqs):
                    for c in range(koboldai_vars.lua_koboldbridge.generated_cols):
                        assert koboldai_vars.lua_koboldbridge.generated[r+1][c+1] is not None
                        past[r, c] = koboldai_vars.lua_koboldbridge.generated[r+1][c+1]

                if(koboldai_vars.abort or halt or not regeneration_required):
                    break
                print("(regeneration triggered)")

                encoded = []
                for i in range(koboldai_vars.numseqs):
                    txt = utils.decodenewlines(tokenizer.decode(past[i]))
                    winfo, mem, anotetxt, _found_entries = calcsubmitbudgetheader(txt, force_use_txt=True, actions=koboldai_vars._actions)
                    found_entries[i].update(_found_entries)
                    txt, _, _ = calcsubmitbudget(len(koboldai_vars._actions), winfo, mem, anotetxt, koboldai_vars._actions, submission=txt)
                    encoded.append(np.array(txt, dtype=np.uint32))
                max_length = len(max(encoded, key=len))
                encoded = np.stack(tuple(np.pad(e, (max_length - len(e), 0), constant_values=tpu_mtj_backend.pad_token_id) for e in encoded))
                context = np.concatenate(
                    (
                        encoded,
                        past,
                    ),
                    axis=-1,
                )

        else:
            genout = tpool.execute(
                tpu_mtj_backend.infer_static,
                np.uint32(txt),
                gen_len = maximum-minimum+1,
                temp=koboldai_vars.temp,
                top_p=koboldai_vars.top_p,
                top_k=koboldai_vars.top_k,
                tfs=koboldai_vars.tfs,
                typical=koboldai_vars.typical,
                top_a=koboldai_vars.top_a,
                numseqs=koboldai_vars.numseqs,
                repetition_penalty=koboldai_vars.rep_pen,
                rpslope=koboldai_vars.rep_pen_slope,
                rprange=koboldai_vars.rep_pen_range,
                soft_embeddings=koboldai_vars.sp,
                soft_tokens=soft_tokens,
                sampler_order=koboldai_vars.sampler_order,
            )
            past = genout
            for i in range(koboldai_vars.numseqs):
                koboldai_vars.lua_koboldbridge.generated[i+1] = koboldai_vars.lua_state.table(*genout[i].tolist())
            koboldai_vars.lua_koboldbridge.generated_cols = koboldai_vars.generated_tkns = genout[0].shape[-1]

    except Exception as e:
        if(issubclass(type(e), lupa.LuaError)):
            koboldai_vars.lua_koboldbridge.obliterate_multiverse()
            koboldai_vars.lua_running = False
            emit('from_server', {'cmd': 'errmsg', 'data': 'Lua script error; please check console.'}, broadcast=True, room="UI_1")
            sendUSStatItems()
            print("{0}{1}{2}".format(colors.RED, "***LUA ERROR***: ", colors.END), end="", file=sys.stderr)
            print("{0}{1}{2}".format(colors.RED, str(e).replace("\033", ""), colors.END), file=sys.stderr)
            print("{0}{1}{2}".format(colors.YELLOW, "Lua engine stopped; please open 'Userscripts' and press Load to reinitialize scripts.", colors.END), file=sys.stderr)
        else:
            emit('from_server', {'cmd': 'errmsg', 'data': 'Error occurred during generator call; please check console.'}, broadcast=True, room="UI_1")
            print("{0}{1}{2}".format(colors.RED, traceback.format_exc().replace("\033", ""), colors.END), file=sys.stderr)
        set_aibusy(0)
        return

    for i in range(koboldai_vars.numseqs):
        koboldai_vars.lua_koboldbridge.outputs[i+1] = utils.decodenewlines(tokenizer.decode(past[i]))
    genout = past

    execute_outmod()
    if(koboldai_vars.lua_koboldbridge.regeneration_required):
        koboldai_vars.lua_koboldbridge.regeneration_required = False
        genout = []
        for i in range(koboldai_vars.numseqs):
            genout.append({"generated_text": koboldai_vars.lua_koboldbridge.outputs[i+1]})
            assert type(genout[-1]["generated_text"]) is str
    else:
        genout = [{"generated_text": utils.decodenewlines(tokenizer.decode(txt))} for txt in genout]

    koboldai_vars.actions.clear_unused_options()
    koboldai_vars.actions.append_options([x["generated_text"] for x in genout])
    genout = [{"generated_text": x['text']} for x in koboldai_vars.actions.get_current_options()]
    if(len(koboldai_vars.actions.get_current_options()) == 1):
        genresult(koboldai_vars.actions.get_current_options()[0]['text'])
    else:
        if(koboldai_vars.lua_koboldbridge.restart_sequence is not None and koboldai_vars.lua_koboldbridge.restart_sequence > 0):
            genresult(genout[koboldai_vars.lua_koboldbridge.restart_sequence-1]["generated_text"])
        else:
            genselect([{"generated_text": x['text']} for x in koboldai_vars.actions.get_current_options()])

    set_aibusy(0)


#==================================================================#
# Replaces returns and newlines with HTML breaks
#==================================================================#
def formatforhtml(txt):
    return txt.replace("\\r\\n", "<br/>").replace("\\r", "<br/>").replace("\\n", "<br/>").replace("\r\n", "<br/>").replace('\n', '<br/>').replace('\r', '<br/>').replace('&lt;/s&gt;', '<br/>')

#==================================================================#
# Strips submitted text from the text returned by the AI
#==================================================================#
def getnewcontent(txt):
    # If the submitted context was blank, then everything is new
    if(koboldai_vars.lastctx == ""):
        return txt
    
    # Tokenize the last context and the generated content
    ctxtokens = tokenizer.encode(utils.encodenewlines(koboldai_vars.lastctx), max_length=int(2e9), truncation=True)
    txttokens = tokenizer.encode(utils.encodenewlines(txt), max_length=int(2e9), truncation=True)
    dif       = (len(txttokens) - len(ctxtokens)) * -1
    
    # Remove the context from the returned text
    newtokens = txttokens[dif:]
    
    return utils.decodenewlines(tokenizer.decode(newtokens))

#==================================================================#
# Applies chosen formatting options to text submitted to AI
#==================================================================#
def applyinputformatting(txt):
    # Add sentence spacing
    if(koboldai_vars.formatoptns["frmtadsnsp"]):
        txt = utils.addsentencespacing(txt, koboldai_vars)
 
    return txt

#==================================================================#
# Applies chosen formatting options to text returned from AI
#==================================================================#
def applyoutputformatting(txt):
    # Use standard quotes and apostrophes
    txt = utils.fixquotes(txt)

    # Adventure mode clipping of all characters after '>'
    if(koboldai_vars.adventure):
        txt = koboldai_vars.acregex_ai.sub('', txt)
    
    # Trim incomplete sentences
    if(koboldai_vars.formatoptns["frmttriminc"] and not koboldai_vars.chatmode):
        txt = utils.trimincompletesentence(txt)
    # Replace blank lines
    if(koboldai_vars.formatoptns["frmtrmblln"] or koboldai_vars.chatmode):
        txt = utils.replaceblanklines(txt)
    # Remove special characters
    if(koboldai_vars.formatoptns["frmtrmspch"]):
        txt = utils.removespecialchars(txt, koboldai_vars)
	# Single Line Mode
    if(koboldai_vars.formatoptns["singleline"] or koboldai_vars.chatmode):
        txt = utils.singlelineprocessing(txt, koboldai_vars)
    
    return txt

#==================================================================#
# Sends the current story content to the Game Screen
#==================================================================#
def refresh_story():
    text_parts = ['<chunk n="0" id="n0" tabindex="-1">', koboldai_vars.comregex_ui.sub(lambda m: '\n'.join('<comment>' + l + '</comment>' for l in m.group().split('\n')), html.escape(koboldai_vars.prompt)), '</chunk>']
    for idx in koboldai_vars.actions:
        item = koboldai_vars.actions[idx]
        idx += 1
        item = html.escape(item)
        item = koboldai_vars.comregex_ui.sub(lambda m: '\n'.join('<comment>' + l + '</comment>' for l in m.group().split('\n')), item)  # Add special formatting to comments
        item = koboldai_vars.acregex_ui.sub('<action>\\1</action>', item)  # Add special formatting to adventure actions
        text_parts.extend(('<chunk n="', str(idx), '" id="n', str(idx), '" tabindex="-1">', item, '</chunk>'))
    emit('from_server', {'cmd': 'updatescreen', 'gamestarted': koboldai_vars.gamestarted, 'data': formatforhtml(''.join(text_parts))}, broadcast=True, room="UI_1")


#==================================================================#
# Signals the Game Screen to update one of the chunks
#==================================================================#
def update_story_chunk(idx: Union[int, str]):
    if idx == 'last':
        if len(koboldai_vars.actions) <= 1:
            # In this case, we are better off just refreshing the whole thing as the
            # prompt might not have been shown yet (with a "Generating story..."
            # message instead).
            refresh_story()
            setgamesaved(False)
            return

        idx = (koboldai_vars.actions.get_last_key() if len(koboldai_vars.actions) else 0) + 1

    if idx == 0:
        text = koboldai_vars.prompt
    else:
        # Actions are 0 based, but in chunks 0 is the prompt.
        # So the chunk index is one more than the corresponding action index.
        if(idx - 1 not in koboldai_vars.actions):
            return
        text = koboldai_vars.actions[idx - 1]

    item = html.escape(text)
    item = koboldai_vars.comregex_ui.sub(lambda m: '\n'.join('<comment>' + l + '</comment>' for l in m.group().split('\n')), item)  # Add special formatting to comments
    item = koboldai_vars.acregex_ui.sub('<action>\\1</action>', item)  # Add special formatting to adventure actions

    chunk_text = f'<chunk n="{idx}" id="n{idx}" tabindex="-1">{formatforhtml(item)}</chunk>'
    emit('from_server', {'cmd': 'updatechunk', 'data': {'index': idx, 'html': chunk_text}}, broadcast=True, room="UI_1")

    setgamesaved(False)

    #If we've set the auto save flag, we'll now save the file
    if koboldai_vars.autosave and (".json" in koboldai_vars.savedir):
        save()


#==================================================================#
# Signals the Game Screen to remove one of the chunks
#==================================================================#
def remove_story_chunk(idx: int):
    emit('from_server', {'cmd': 'removechunk', 'data': idx}, broadcast=True, room="UI_1")
    setgamesaved(False)


#==================================================================#
# Sends the current generator settings to the Game Menu
#==================================================================#
def refresh_settings():
    # Suppress toggle change events while loading state
    emit('from_server', {'cmd': 'allowtoggle', 'data': False}, broadcast=True, room="UI_1")
    
    if(koboldai_vars.model != "InferKit"):
        emit('from_server', {'cmd': 'updatetemp', 'data': koboldai_vars.temp}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'updatetopp', 'data': koboldai_vars.top_p}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'updatetopk', 'data': koboldai_vars.top_k}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'updatetfs', 'data': koboldai_vars.tfs}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'updatetypical', 'data': koboldai_vars.typical}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'updatetopa', 'data': koboldai_vars.top_a}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'updatereppen', 'data': koboldai_vars.rep_pen}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'updatereppenslope', 'data': koboldai_vars.rep_pen_slope}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'updatereppenrange', 'data': koboldai_vars.rep_pen_range}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'updateoutlen', 'data': koboldai_vars.genamt}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'updatetknmax', 'data': koboldai_vars.max_length}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'updatenumseq', 'data': koboldai_vars.numseqs}, broadcast=True, room="UI_1")
    else:
        emit('from_server', {'cmd': 'updatetemp', 'data': koboldai_vars.temp}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'updatetopp', 'data': koboldai_vars.top_p}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'updateikgen', 'data': koboldai_vars.ikgen}, broadcast=True, room="UI_1")
    
    emit('from_server', {'cmd': 'updateanotedepth', 'data': koboldai_vars.andepth}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'updatewidepth', 'data': koboldai_vars.widepth}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'updateuseprompt', 'data': koboldai_vars.useprompt}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'updateadventure', 'data': koboldai_vars.adventure}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'updatechatmode', 'data': koboldai_vars.chatmode}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'updatedynamicscan', 'data': koboldai_vars.dynamicscan}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'updateautosave', 'data': koboldai_vars.autosave}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'updatenopromptgen', 'data': koboldai_vars.nopromptgen}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'updaterngpersist', 'data': koboldai_vars.rngpersist}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'updatenogenmod', 'data': koboldai_vars.nogenmod}, broadcast=True, room="UI_1")
    
    emit('from_server', {'cmd': 'updatefrmttriminc', 'data': koboldai_vars.formatoptns["frmttriminc"]}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'updatefrmtrmblln', 'data': koboldai_vars.formatoptns["frmtrmblln"]}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'updatefrmtrmspch', 'data': koboldai_vars.formatoptns["frmtrmspch"]}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'updatefrmtadsnsp', 'data': koboldai_vars.formatoptns["frmtadsnsp"]}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'updatesingleline', 'data': koboldai_vars.formatoptns["singleline"]}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'updateoutputstreaming', 'data': koboldai_vars.output_streaming}, broadcast=True, room="UI_1")
    
    # Allow toggle events again
    emit('from_server', {'cmd': 'allowtoggle', 'data': True}, broadcast=True, room="UI_1")

#==================================================================#
#  Sets the logical and display states for the AI Busy condition
#==================================================================#
def set_aibusy(state):
    if(state):
        koboldai_vars.aibusy = True
        emit('from_server', {'cmd': 'setgamestate', 'data': 'wait'}, broadcast=True, room="UI_1")
    else:
        koboldai_vars.aibusy = False
        emit('from_server', {'cmd': 'setgamestate', 'data': 'ready'}, broadcast=True, room="UI_1")

#==================================================================#
# 
#==================================================================#
def editrequest(n):
    if(n == 0):
        txt = koboldai_vars.prompt
    else:
        txt = koboldai_vars.actions[n-1]
    
    koboldai_vars.editln = n
    emit('from_server', {'cmd': 'setinputtext', 'data': txt}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'enablesubmit', 'data': ''}, broadcast=True, room="UI_1")

#==================================================================#
# 
#==================================================================#
def editsubmit(data):
    koboldai_vars.recentedit = True
    if(koboldai_vars.editln == 0):
        koboldai_vars.prompt = data
    else:
        koboldai_vars.actions[koboldai_vars.editln-1] = data
    
    koboldai_vars.mode = "play"
    update_story_chunk(koboldai_vars.editln)
    emit('from_server', {'cmd': 'texteffect', 'data': koboldai_vars.editln}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'editmode', 'data': 'false'}, room="UI_1")
    send_debug()

#==================================================================#
#  
#==================================================================#
def deleterequest():
    koboldai_vars.recentedit = True
    # Don't delete prompt
    if(koboldai_vars.editln == 0):
        # Send error message
        pass
    else:
        koboldai_vars.actions.delete_action(koboldai_vars.editln-1)
        koboldai_vars.mode = "play"
        remove_story_chunk(koboldai_vars.editln)
        emit('from_server', {'cmd': 'editmode', 'data': 'false'}, room="UI_1")
    send_debug()

#==================================================================#
# 
#==================================================================#
def inlineedit(chunk, data):
    koboldai_vars.recentedit = True
    chunk = int(chunk)
    if(chunk == 0):
        if(len(data.strip()) == 0):
            return
        koboldai_vars.prompt = data
    else:
        if(chunk-1 in koboldai_vars.actions):
            koboldai_vars.actions[chunk-1] = data
        else:
            print(f"WARNING: Attempted to edit non-existent chunk {chunk}")

    setgamesaved(False)
    update_story_chunk(chunk)
    emit('from_server', {'cmd': 'texteffect', 'data': chunk}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'editmode', 'data': 'false'}, broadcast=True, room="UI_1")
    send_debug()

#==================================================================#
#  
#==================================================================#
def inlinedelete(chunk):
    koboldai_vars.recentedit = True
    chunk = int(chunk)
    # Don't delete prompt
    if(chunk == 0):
        # Send error message
        update_story_chunk(chunk)
        emit('from_server', {'cmd': 'errmsg', 'data': "Cannot delete the prompt."}, room="UI_1")
        emit('from_server', {'cmd': 'editmode', 'data': 'false'}, broadcast=True, room="UI_1")
    else:
        if(chunk-1 in koboldai_vars.actions):
            koboldai_vars.actions.delete_action(chunk-1)
        else:
            print(f"WARNING: Attempted to delete non-existent chunk {chunk}")
        setgamesaved(False)
        remove_story_chunk(chunk)
        emit('from_server', {'cmd': 'editmode', 'data': 'false'}, broadcast=True, room="UI_1")
    send_debug()

#==================================================================#
#   Toggles the game mode for memory editing and sends UI commands
#==================================================================#
def togglememorymode():
    if(koboldai_vars.mode == "play"):
        koboldai_vars.mode = "memory"
        emit('from_server', {'cmd': 'memmode', 'data': 'true'}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'setinputtext', 'data': koboldai_vars.memory}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'setanote', 'data': koboldai_vars.authornote}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'setanotetemplate', 'data': koboldai_vars.authornotetemplate}, broadcast=True, room="UI_1")
    elif(koboldai_vars.mode == "memory"):
        koboldai_vars.mode = "play"
        emit('from_server', {'cmd': 'memmode', 'data': 'false'}, broadcast=True, room="UI_1")

#==================================================================#
#   Toggles the game mode for WI editing and sends UI commands
#==================================================================#
def togglewimode():
    if(koboldai_vars.mode == "play"):
        koboldai_vars.mode = "wi"
        emit('from_server', {'cmd': 'wimode', 'data': 'true'}, broadcast=True, room="UI_1")
    elif(koboldai_vars.mode == "wi"):
        # Commit WI fields first
        requestwi()
        # Then set UI state back to Play
        koboldai_vars.mode = "play"
        emit('from_server', {'cmd': 'wimode', 'data': 'false'}, broadcast=True, room="UI_1")
    sendwi()

#==================================================================#
#   
#==================================================================#
def addwiitem(folder_uid=None):
    assert folder_uid is None or folder_uid in koboldai_vars.wifolders_d
    ob = {"key": "", "keysecondary": "", "content": "", "comment": "", "folder": folder_uid, "num": len(koboldai_vars.worldinfo), "init": False, "selective": False, "constant": False}
    koboldai_vars.worldinfo.append(ob)
    while(True):
        uid = int.from_bytes(os.urandom(4), "little", signed=True)
        if(uid not in koboldai_vars.worldinfo_u):
            break
    koboldai_vars.worldinfo_u[uid] = koboldai_vars.worldinfo[-1]
    koboldai_vars.worldinfo[-1]["uid"] = uid
    if(folder_uid is not None):
        koboldai_vars.wifolders_u[folder_uid].append(koboldai_vars.worldinfo[-1])
    emit('from_server', {'cmd': 'addwiitem', 'data': ob}, broadcast=True, room="UI_1")

#==================================================================#
#   Creates a new WI folder with an unused cryptographically secure random UID
#==================================================================#
def addwifolder():
    while(True):
        uid = int.from_bytes(os.urandom(4), "little", signed=True)
        if(uid not in koboldai_vars.wifolders_d):
            break
    ob = {"name": "", "collapsed": False}
    koboldai_vars.wifolders_d[uid] = ob
    koboldai_vars.wifolders_l.append(uid)
    koboldai_vars.wifolders_u[uid] = []
    emit('from_server', {'cmd': 'addwifolder', 'uid': uid, 'data': ob}, broadcast=True, room="UI_1")
    addwiitem(folder_uid=uid)

#==================================================================#
#   Move the WI entry with UID src so that it immediately precedes
#   the WI entry with UID dst
#==================================================================#
def movewiitem(dst, src):
    setgamesaved(False)
    if(koboldai_vars.worldinfo_u[src]["folder"] is not None):
        for i, e in enumerate(koboldai_vars.wifolders_u[koboldai_vars.worldinfo_u[src]["folder"]]):
            if(e is koboldai_vars.worldinfo_u[src]):
                koboldai_vars.wifolders_u[koboldai_vars.worldinfo_u[src]["folder"]].pop(i)
                break
    if(koboldai_vars.worldinfo_u[dst]["folder"] is not None):
        koboldai_vars.wifolders_u[koboldai_vars.worldinfo_u[dst]["folder"]].append(koboldai_vars.worldinfo_u[src])
    koboldai_vars.worldinfo_u[src]["folder"] = koboldai_vars.worldinfo_u[dst]["folder"]
    for i, e in enumerate(koboldai_vars.worldinfo):
        if(e is koboldai_vars.worldinfo_u[src]):
            _src = i
        elif(e is koboldai_vars.worldinfo_u[dst]):
            _dst = i
    koboldai_vars.worldinfo.insert(_dst - (_dst >= _src), koboldai_vars.worldinfo.pop(_src))
    sendwi()

#==================================================================#
#   Move the WI folder with UID src so that it immediately precedes
#   the WI folder with UID dst
#==================================================================#
def movewifolder(dst, src):
    setgamesaved(False)
    koboldai_vars.wifolders_l.remove(src)
    if(dst is None):
        # If dst is None, that means we should move src to be the last folder
        koboldai_vars.wifolders_l.append(src)
    else:
        koboldai_vars.wifolders_l.insert(koboldai_vars.wifolders_l.index(dst), src)
    sendwi()

#==================================================================#
#   
#==================================================================#
def sendwi():
    # Cache len of WI
    ln = len(koboldai_vars.worldinfo)

    # Clear contents of WI container
    emit('from_server', {'cmd': 'wistart', 'wifolders_d': koboldai_vars.wifolders_d, 'wifolders_l': koboldai_vars.wifolders_l, 'data': ''}, broadcast=True, room="UI_1")

    # Stable-sort WI entries in order of folder
    stablesortwi()

    koboldai_vars.worldinfo_i = [wi for wi in koboldai_vars.worldinfo if wi["init"]]

    # If there are no WI entries, send an empty WI object
    if(ln == 0):
        addwiitem()
    else:
        # Send contents of WI array
        last_folder = ...
        for wi in koboldai_vars.worldinfo:
            if(wi["folder"] != last_folder):
                emit('from_server', {'cmd': 'addwifolder', 'uid': wi["folder"], 'data': koboldai_vars.wifolders_d[wi["folder"]] if wi["folder"] is not None else None}, broadcast=True, room="UI_1")
                last_folder = wi["folder"]
            ob = wi
            emit('from_server', {'cmd': 'addwiitem', 'data': ob}, broadcast=True, room="UI_1")
    
    emit('from_server', {'cmd': 'wifinish', 'data': ''}, broadcast=True, room="UI_1")

#==================================================================#
#  Request current contents of all WI HTML elements
#==================================================================#
def requestwi():
    list = []
    for wi in koboldai_vars.worldinfo:
        list.append(wi["num"])
    emit('from_server', {'cmd': 'requestwiitem', 'data': list}, room="UI_1")

#==================================================================#
#  Stable-sort WI items so that items in the same folder are adjacent,
#  and items in different folders are sorted based on the order of the folders
#==================================================================#
def stablesortwi():
    mapping = {uid: index for index, uid in enumerate(koboldai_vars.wifolders_l)}
    koboldai_vars.worldinfo.sort(key=lambda x: mapping[x["folder"]] if x["folder"] is not None else float("inf"))
    last_folder = ...
    last_wi = None
    for i, wi in enumerate(koboldai_vars.worldinfo):
        wi["num"] = i
        wi["init"] = True
        if(wi["folder"] != last_folder):
            if(last_wi is not None and last_folder is not ...):
                last_wi["init"] = False
            last_folder = wi["folder"]
        last_wi = wi
    if(last_wi is not None):
        last_wi["init"] = False
    for folder in koboldai_vars.wifolders_u:
        koboldai_vars.wifolders_u[folder].sort(key=lambda x: x["num"])

#==================================================================#
#  Extract object from server and send it to WI objects
#==================================================================#
def commitwi(ar):
    for ob in ar:
        ob["uid"] = int(ob["uid"])
        koboldai_vars.worldinfo_u[ob["uid"]]["key"]          = ob["key"]
        koboldai_vars.worldinfo_u[ob["uid"]]["keysecondary"] = ob["keysecondary"]
        koboldai_vars.worldinfo_u[ob["uid"]]["content"]      = ob["content"]
        koboldai_vars.worldinfo_u[ob["uid"]]["comment"]      = ob.get("comment", "")
        koboldai_vars.worldinfo_u[ob["uid"]]["folder"]       = ob.get("folder", None)
        koboldai_vars.worldinfo_u[ob["uid"]]["selective"]    = ob["selective"]
        koboldai_vars.worldinfo_u[ob["uid"]]["constant"]     = ob.get("constant", False)
    stablesortwi()
    koboldai_vars.worldinfo_i = [wi for wi in koboldai_vars.worldinfo if wi["init"]]

#==================================================================#
#  
#==================================================================#
def deletewi(uid):
    if(uid in koboldai_vars.worldinfo_u):
        setgamesaved(False)
        # Store UID of deletion request
        koboldai_vars.deletewi = uid
        if(koboldai_vars.deletewi is not None):
            if(koboldai_vars.worldinfo_u[koboldai_vars.deletewi]["folder"] is not None):
                for i, e in enumerate(koboldai_vars.wifolders_u[koboldai_vars.worldinfo_u[koboldai_vars.deletewi]["folder"]]):
                    if(e is koboldai_vars.worldinfo_u[koboldai_vars.deletewi]):
                        koboldai_vars.wifolders_u[koboldai_vars.worldinfo_u[koboldai_vars.deletewi]["folder"]].pop(i)
            for i, e in enumerate(koboldai_vars.worldinfo):
                if(e is koboldai_vars.worldinfo_u[koboldai_vars.deletewi]):
                    del koboldai_vars.worldinfo[i]
                    break
            del koboldai_vars.worldinfo_u[koboldai_vars.deletewi]
            # Send the new WI array structure
            sendwi()
            # And reset deletewi
            koboldai_vars.deletewi = None

#==================================================================#
#  
#==================================================================#
def deletewifolder(uid):
    uid = int(uid)
    del koboldai_vars.wifolders_u[uid]
    del koboldai_vars.wifolders_d[uid]
    del koboldai_vars.wifolders_l[koboldai_vars.wifolders_l.index(uid)]
    setgamesaved(False)
    # Delete uninitialized entries in the folder we're going to delete
    koboldai_vars.worldinfo = [wi for wi in koboldai_vars.worldinfo if wi["folder"] != uid or wi["init"]]
    koboldai_vars.worldinfo_i = [wi for wi in koboldai_vars.worldinfo if wi["init"]]
    # Move WI entries that are inside of the folder we're going to delete
    # so that they're outside of all folders
    for wi in koboldai_vars.worldinfo:
        if(wi["folder"] == uid):
            wi["folder"] = None

    sendwi()

#==================================================================#
#  Look for WI keys in text to generator 
#==================================================================#
def checkworldinfo(txt, allowed_entries=None, allowed_folders=None, force_use_txt=False, scan_story=True, actions=None):
    original_txt = txt

    if(actions is None):
        actions = koboldai_vars.actions

    # Dont go any further if WI is empty
    if(len(koboldai_vars.worldinfo) == 0):
        return "", set()
    
    # Cache actions length
    ln = len(actions)
    
    # Don't bother calculating action history if widepth is 0
    if(koboldai_vars.widepth > 0 and scan_story):
        depth = koboldai_vars.widepth
        # If this is not a continue, add 1 to widepth since submitted
        # text is already in action history @ -1
        if(not force_use_txt and (txt != "" and koboldai_vars.prompt != txt)):
            txt    = ""
            depth += 1
        
        if(ln > 0):
            chunks = collections.deque()
            i = 0
            for key in reversed(actions):
                chunk = actions[key]
                chunks.appendleft(chunk)
                i += 1
                if(i == depth):
                    break
        
        if(ln >= depth):
            txt = "".join(chunks)
        elif(ln > 0):
            txt = koboldai_vars.comregex_ai.sub('', koboldai_vars.prompt) + "".join(chunks)
        elif(ln == 0):
            txt = koboldai_vars.comregex_ai.sub('', koboldai_vars.prompt)

    if(force_use_txt):
        txt += original_txt

    # Scan text for matches on WI keys
    wimem = ""
    found_entries = set()
    for wi in koboldai_vars.worldinfo:
        if(allowed_entries is not None and wi["uid"] not in allowed_entries):
            continue
        if(allowed_folders is not None and wi["folder"] not in allowed_folders):
            continue

        if(wi.get("constant", False)):
            wimem = wimem + wi["content"] + "\n"
            found_entries.add(id(wi))
            continue

        if(len(wi["key"].strip()) > 0 and (not wi.get("selective", False) or len(wi.get("keysecondary", "").strip()) > 0)):
            # Split comma-separated keys
            keys = wi["key"].split(",")
            keys_secondary = wi.get("keysecondary", "").split(",")

            for k in keys:
                ky = k
                # Remove leading/trailing spaces if the option is enabled
                if(koboldai_vars.wirmvwhtsp):
                    ky = k.strip()
                if ky in txt:
                    if wi.get("selective", False) and len(keys_secondary):
                        found = False
                        for ks in keys_secondary:
                            ksy = ks
                            if(koboldai_vars.wirmvwhtsp):
                                ksy = ks.strip()
                            if ksy in txt:
                                wimem = wimem + wi["content"] + "\n"
                                found_entries.add(id(wi))
                                found = True
                                break
                        if found:
                            break
                    else:
                        wimem = wimem + wi["content"] + "\n"
                        found_entries.add(id(wi))
                        break
    
    return wimem, found_entries
    
#==================================================================#
#  Commit changes to Memory storage
#==================================================================#
def memsubmit(data):
    emit('from_server', {'cmd': 'setinputtext', 'data': data}, broadcast=True, room="UI_1")
    # Maybe check for length at some point
    # For now just send it to storage
    if(data != koboldai_vars.memory):
        setgamesaved(False)
    koboldai_vars.memory = data
    koboldai_vars.mode = "play"
    emit('from_server', {'cmd': 'memmode', 'data': 'false'}, broadcast=True, room="UI_1")
    
    # Ask for contents of Author's Note field
    emit('from_server', {'cmd': 'getanote', 'data': ''}, room="UI_1")

#==================================================================#
#  Commit changes to Author's Note
#==================================================================#
def anotesubmit(data, template=""):
    assert type(data) is str and type(template) is str
    # Maybe check for length at some point
    # For now just send it to storage
    if(data != koboldai_vars.authornote):
        setgamesaved(False)
    koboldai_vars.authornote = data

    if(koboldai_vars.authornotetemplate != template):
        koboldai_vars.setauthornotetemplate = template
        print("anotesubmit")
        settingschanged()
    koboldai_vars.authornotetemplate = template

    emit('from_server', {'cmd': 'setanote', 'data': koboldai_vars.authornote}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'setanotetemplate', 'data': koboldai_vars.authornotetemplate}, broadcast=True, room="UI_1")

#==================================================================#
#  Assembles game data into a request to InferKit API
#==================================================================#
def ikrequest(txt):
    # Log request to console
    if not koboldai_vars.quiet:
        print("{0}Len:{1}, Txt:{2}{3}".format(colors.YELLOW, len(txt), txt, colors.END))
    
    # Build request JSON data
    reqdata = {
        'forceNoEnd': True,
        'length': koboldai_vars.ikgen,
        'prompt': {
            'isContinuation': False,
            'text': txt
        },
        'startFromBeginning': False,
        'streamResponse': False,
        'temperature': koboldai_vars.temp,
        'topP': koboldai_vars.top_p
    }
    
    # Create request
    req = requests.post(
        koboldai_vars.url, 
        json    = reqdata,
        headers = {
            'Authorization': 'Bearer '+koboldai_vars.apikey
            }
        )
    
    # Deal with the response
    if(req.status_code == 200):
        genout = req.json()["data"]["text"]

        koboldai_vars.lua_koboldbridge.outputs[1] = genout

        execute_outmod()
        if(koboldai_vars.lua_koboldbridge.regeneration_required):
            koboldai_vars.lua_koboldbridge.regeneration_required = False
            genout = koboldai_vars.lua_koboldbridge.outputs[1]
            assert genout is str

        if not koboldai_vars.quiet:
            print("{0}{1}{2}".format(colors.CYAN, genout, colors.END))
        koboldai_vars.actions.append(genout)
        update_story_chunk('last')
        emit('from_server', {'cmd': 'texteffect', 'data': koboldai_vars.actions.get_last_key() + 1 if len(koboldai_vars.actions) else 0}, broadcast=True, room="UI_1")
        send_debug()
        set_aibusy(0)
    else:
        # Send error message to web client
        er = req.json()
        if("error" in er):
            code = er["error"]["extensions"]["code"]
        elif("errors" in er):
            code = er["errors"][0]["extensions"]["code"]
            
        errmsg = "InferKit API Error: {0} - {1}".format(req.status_code, code)
        emit('from_server', {'cmd': 'errmsg', 'data': errmsg}, broadcast=True, room="UI_1")
        set_aibusy(0)

#==================================================================#
#  Assembles game data into a request to OpenAI API
#==================================================================#
def oairequest(txt, min, max):
    # Log request to console
    if not koboldai_vars.quiet:
        print("{0}Len:{1}, Txt:{2}{3}".format(colors.YELLOW, len(txt), txt, colors.END))
    
    # Store context in memory to use it for comparison with generated content
    koboldai_vars.lastctx = txt
    
    # Build request JSON data
    if 'GooseAI' in args.configname:
        reqdata = {
            'prompt': txt,
            'max_tokens': koboldai_vars.genamt,
            'temperature': koboldai_vars.temp,
            'top_a': koboldai_vars.top_a,
            'top_p': koboldai_vars.top_p,
            'top_k': koboldai_vars.top_k,
            'tfs': koboldai_vars.tfs,
            'typical_p': koboldai_vars.typical,
            'repetition_penalty': koboldai_vars.rep_pen,
            'repetition_penalty_slope': koboldai_vars.rep_pen_slope,
            'repetition_penalty_range': koboldai_vars.rep_pen_range,
            'n': koboldai_vars.numseqs,
            'stream': False
        }
    else:
        reqdata = {
            'prompt': txt,
            'max_tokens': koboldai_vars.genamt,
            'temperature': koboldai_vars.temp,
            'top_p': koboldai_vars.top_p,
            'n': koboldai_vars.numseqs,
            'stream': False
        }
    
    req = requests.post(
        koboldai_vars.oaiurl, 
        json    = reqdata,
        headers = {
            'Authorization': 'Bearer '+koboldai_vars.oaiapikey,
            'Content-Type': 'application/json'
            }
        )
    
    # Deal with the response
    if(req.status_code == 200):
        outputs = [out["text"] for out in req.json()["choices"]]

        for idx in range(len(outputs)):
            koboldai_vars.lua_koboldbridge.outputs[idx+1] = outputs[idx]

        execute_outmod()
        if (koboldai_vars.lua_koboldbridge.regeneration_required):
            koboldai_vars.lua_koboldbridge.regeneration_required = False
            genout = []
            for i in range(len(outputs)):
                genout.append(
                    {"generated_text": koboldai_vars.lua_koboldbridge.outputs[i + 1]})
                assert type(genout[-1]["generated_text"]) is str
        else:
            genout = [
                {"generated_text": utils.decodenewlines(txt)}
                for txt in outputs]

        koboldai_vars.actions.clear_unused_options()
        koboldai_vars.actions.append_options([x["generated_text"] for x in genout])
        genout = [{"generated_text": x['text']} for x in koboldai_vars.actions.get_current_options()]
        if (len(genout) == 1):
            genresult(genout[0]["generated_text"])
        else:
            if (koboldai_vars.lua_koboldbridge.restart_sequence is not None and
                    koboldai_vars.lua_koboldbridge.restart_sequence > 0):
                genresult(genout[koboldai_vars.lua_koboldbridge.restart_sequence - 1][
                              "generated_text"])
            else:
                genselect(genout)

        if not koboldai_vars.quiet:
            print("{0}{1}{2}".format(colors.CYAN, genout, colors.END))

        set_aibusy(0)
    else:
        # Send error message to web client            
        er = req.json()
        if("error" in er):
            type    = er["error"]["type"]
            message = er["error"]["message"]
            
        errmsg = "OpenAI API Error: {0} - {1}".format(type, message)
        emit('from_server', {'cmd': 'errmsg', 'data': errmsg}, broadcast=True, room="UI_1")
        set_aibusy(0)

#==================================================================#
#  Forces UI to Play mode
#==================================================================#
def exitModes():
    if(koboldai_vars.mode == "edit"):
        emit('from_server', {'cmd': 'editmode', 'data': 'false'}, broadcast=True, room="UI_1")
    elif(koboldai_vars.mode == "memory"):
        emit('from_server', {'cmd': 'memmode', 'data': 'false'}, broadcast=True, room="UI_1")
    elif(koboldai_vars.mode == "wi"):
        emit('from_server', {'cmd': 'wimode', 'data': 'false'}, broadcast=True, room="UI_1")
    koboldai_vars.mode = "play"

#==================================================================#
#  Launch in-browser save prompt
#==================================================================#
def saveas(data):
    
    name = data['name']
    savepins = data['pins']
    # Check if filename exists already
    name = utils.cleanfilename(name)
    if(not fileops.saveexists(name) or (koboldai_vars.saveow and koboldai_vars.svowname == name)):
        # All clear to save
        e = saveRequest(fileops.storypath(name), savepins=savepins)
        koboldai_vars.saveow = False
        koboldai_vars.svowname = ""
        if(e is None):
            emit('from_server', {'cmd': 'hidesaveas', 'data': ''}, room="UI_1")
        else:
            print("{0}{1}{2}".format(colors.RED, str(e), colors.END))
            emit('from_server', {'cmd': 'popuperror', 'data': str(e)}, room="UI_1")
    else:
        # File exists, prompt for overwrite
        koboldai_vars.saveow   = True
        koboldai_vars.svowname = name
        emit('from_server', {'cmd': 'askforoverwrite', 'data': ''}, room="UI_1")

#==================================================================#
#  Launch in-browser story-delete prompt
#==================================================================#
def deletesave(name):
    name = utils.cleanfilename(name)
    e = fileops.deletesave(name)
    if(e is None):
        if(koboldai_vars.smandelete):
            emit('from_server', {'cmd': 'hidepopupdelete', 'data': ''}, room="UI_1")
            getloadlist()
        else:
            emit('from_server', {'cmd': 'popuperror', 'data': "The server denied your request to delete this story"}, room="UI_1")
    else:
        print("{0}{1}{2}".format(colors.RED, str(e), colors.END))
        emit('from_server', {'cmd': 'popuperror', 'data': str(e)}, room="UI_1")

#==================================================================#
#  Launch in-browser story-rename prompt
#==================================================================#
def renamesave(name, newname):
    # Check if filename exists already
    name = utils.cleanfilename(name)
    newname = utils.cleanfilename(newname)
    if(not fileops.saveexists(newname) or name == newname or (koboldai_vars.saveow and koboldai_vars.svowname == newname)):
        e = fileops.renamesave(name, newname)
        koboldai_vars.saveow = False
        koboldai_vars.svowname = ""
        if(e is None):
            if(koboldai_vars.smanrename):
                emit('from_server', {'cmd': 'hidepopuprename', 'data': ''}, room="UI_1")
                getloadlist()
            else:
                emit('from_server', {'cmd': 'popuperror', 'data': "The server denied your request to rename this story"}, room="UI_1")
        else:
            print("{0}{1}{2}".format(colors.RED, str(e), colors.END))
            emit('from_server', {'cmd': 'popuperror', 'data': str(e)}, room="UI_1")
    else:
        # File exists, prompt for overwrite
        koboldai_vars.saveow   = True
        koboldai_vars.svowname = newname
        emit('from_server', {'cmd': 'askforoverwrite', 'data': ''}, room="UI_1")

#==================================================================#
#  Save the currently running story
#==================================================================#
def save():
    # Check if a file is currently open
    if(".json" in koboldai_vars.savedir):
        saveRequest(koboldai_vars.savedir)
    else:
        emit('from_server', {'cmd': 'saveas', 'data': ''}, room="UI_1")

#==================================================================#
#  Save the story via file browser
#==================================================================#
def savetofile():
    savpath = fileops.getsavepath(koboldai_vars.savedir, "Save Story As", [("Json", "*.json")])
    saveRequest(savpath)

#==================================================================#
#  Save the story to specified path
#==================================================================#
def saveRequest(savpath, savepins=True):    
    if(savpath):
        # Leave Edit/Memory mode before continuing
        exitModes()
        
        # Save path for future saves
        koboldai_vars.savedir = savpath
        txtpath = os.path.splitext(savpath)[0] + ".txt"
        # Build json to write
        js = {}
        js["gamestarted"] = koboldai_vars.gamestarted
        js["prompt"]      = koboldai_vars.prompt
        js["memory"]      = koboldai_vars.memory
        js["authorsnote"] = koboldai_vars.authornote
        js["anotetemplate"] = koboldai_vars.authornotetemplate
        js["actions"]     = tuple(koboldai_vars.actions.values())
        if savepins:
            js["actions_metadata"]     = koboldai_vars.actions.options(ui_version=1)
        js["worldinfo"]   = []
        js["wifolders_d"] = koboldai_vars.wifolders_d
        js["wifolders_l"] = koboldai_vars.wifolders_l
		
        # Extract only the important bits of WI
        for wi in koboldai_vars.worldinfo_i:
            if(True):
                js["worldinfo"].append({
                    "key": wi["key"],
                    "keysecondary": wi["keysecondary"],
                    "content": wi["content"],
                    "comment": wi["comment"],
                    "folder": wi["folder"],
                    "selective": wi["selective"],
                    "constant": wi["constant"]
                })
                
        txt = koboldai_vars.prompt + "".join(koboldai_vars.actions.values())

        # Write it
        try:
            file = open(savpath, "w")
        except Exception as e:
            return e
        try:
            file.write(json.dumps(js, indent=3))
        except Exception as e:
            file.close()
            return e
        file.close()
        
        try:
            file = open(txtpath, "w")
        except Exception as e:
            return e
        try:
            file.write(txt)
        except Exception as e:
            file.close()
            return e
        file.close()

        filename = path.basename(savpath)
        if(filename.endswith('.json')):
            filename = filename[:-5]
        koboldai_vars.laststory = filename
        emit('from_server', {'cmd': 'setstoryname', 'data': koboldai_vars.laststory}, broadcast=True, room="UI_1")
        setgamesaved(True)
        print("{0}Story saved to {1}!{2}".format(colors.GREEN, path.basename(savpath), colors.END))

#==================================================================#
#  Show list of saved stories
#==================================================================#
def getloadlist(data=None):
    emit('from_server', {'cmd': 'buildload', 'data': fileops.getstoryfiles()}, room="UI_1")

#==================================================================#
#  Show list of soft prompts
#==================================================================#
def getsplist():
    if(koboldai_vars.allowsp):
        emit('from_server', {'cmd': 'buildsp', 'data': fileops.getspfiles(koboldai_vars.modeldim)}, room="UI_1")

#==================================================================#
#  Get list of userscripts
#==================================================================#
def getuslist():
    files = {i: v for i, v in enumerate(fileops.getusfiles())}
    loaded = []
    unloaded = []
    userscripts = set(koboldai_vars.userscripts)
    for i in range(len(files)):
        if files[i]["filename"] not in userscripts:
            unloaded.append(files[i])
    files = {files[k]["filename"]: files[k] for k in files}
    userscripts = set(files.keys())
    for filename in koboldai_vars.userscripts:
        if filename in userscripts:
            loaded.append(files[filename])
    return unloaded, loaded

#==================================================================#
#  Load a saved story via file browser
#==================================================================#
def loadfromfile():
    loadpath = fileops.getloadpath(koboldai_vars.savedir, "Select Story File", [("Json", "*.json")])
    loadRequest(loadpath)

#==================================================================#
#  Load a stored story from a file
#==================================================================#
def loadRequest(loadpath, filename=None):
    if(loadpath):
        # Leave Edit/Memory mode before continuing
        exitModes()
        
        
        # Read file contents into JSON object
        if(isinstance(loadpath, str)):
            with open(loadpath, "r") as file:
                js = json.load(file)
            if(filename is None):
                filename = path.basename(loadpath)
        else:
            js = loadpath
            if(filename is None):
                filename = "untitled.json"
        js['v1_loadpath'] = loadpath
        js['v1_filename'] = filename
        loadJSON(js)

def loadJSON(json_text_or_dict):
    if isinstance(json_text_or_dict, str):
        json_data = json.loads(json_text_or_dict)
    else:
        json_data = json_text_or_dict
    if "file_version" in json_data:
        if json_data['file_version'] == 2:
            load_story_v2(json_data)
        else:
            load_story_v1(json_data)
    else:
        load_story_v1(json_data)

def load_story_v1(js):
    loadpath = js['v1_loadpath']
    filename = js['v1_filename']
    
    _filename = filename
    if(filename.endswith('.json')):
        _filename = filename[:-5]
    session['story'] = _filename
    #create the story
    #koboldai_vars.create_story(session['story'])
    koboldai_vars.create_story('default')
     
    koboldai_vars.laststory = _filename
    #set the story_name
    koboldai_vars.story_name = _filename
    

    # Copy file contents to vars
    koboldai_vars.gamestarted = js["gamestarted"]
    koboldai_vars.prompt      = js["prompt"]
    koboldai_vars.memory      = js["memory"]
    koboldai_vars.worldinfo_v2.reset()
    koboldai_vars.worldinfo   = []
    koboldai_vars.worldinfo_i = []
    koboldai_vars.worldinfo_u = {}
    koboldai_vars.wifolders_d = {int(k): v for k, v in js.get("wifolders_d", {}).items()}
    koboldai_vars.wifolders_l = js.get("wifolders_l", [])
    koboldai_vars.wifolders_u = {uid: [] for uid in koboldai_vars.wifolders_d}
    koboldai_vars.lastact     = ""
    koboldai_vars.submission  = ""
    koboldai_vars.lastctx     = ""
    koboldai_vars.genseqs = []

    actions = collections.deque(js["actions"])
    


    if(len(koboldai_vars.prompt.strip()) == 0):
        while(len(actions)):
            action = actions.popleft()
            if(len(action.strip()) != 0):
                koboldai_vars.prompt = action
                break
        else:
            koboldai_vars.gamestarted = False
    if(koboldai_vars.gamestarted):
        for s in actions:
            koboldai_vars.actions.append(s)

    if "actions_metadata" in js:
        if type(js["actions_metadata"]) == dict:
            for key in js["actions_metadata"]:
                if js["actions_metadata"][key]["Alternative Text"] != []:
                    data = js["actions_metadata"][key]["Alternative Text"]
                    data["text"] = data.pop("Text")
                    koboldai_vars.actions.set_options(self, data, key)
    
    # Try not to break older save files
    if("authorsnote" in js):
        koboldai_vars.authornote = js["authorsnote"]
    else:
        koboldai_vars.authornote = ""
    if("anotetemplate" in js):
        koboldai_vars.authornotetemplate = js["anotetemplate"]
    else:
        koboldai_vars.authornotetemplate = "[Author's note: <|>]"
    
    if("worldinfo" in js):
        num = 0
        for wi in js["worldinfo"]:
            koboldai_vars.worldinfo.append({
                "key": wi["key"],
                "keysecondary": wi.get("keysecondary", ""),
                "content": wi["content"],
                "comment": wi.get("comment", ""),
                "folder": wi.get("folder", None),
                "num": num,
                "init": True,
                "selective": wi.get("selective", False),
                "constant": wi.get("constant", False),
                "uid": None,
            })
            koboldai_vars.worldinfo_v2.append({
                "key": wi["key"],
                "keysecondary": wi.get("keysecondary", ""),
                "content": wi["content"],
                "comment": wi.get("comment", ""),
                "folder": wi.get("folder", None),
                "num": num,
                "init": True,
                "selective": wi.get("selective", False),
                "constant": wi.get("constant", False),
            })
            
            while(True):
                uid = int.from_bytes(os.urandom(4), "little", signed=True)
                if(uid not in koboldai_vars.worldinfo_u):
                    break
            koboldai_vars.worldinfo_u[uid] = koboldai_vars.worldinfo[-1]
            koboldai_vars.worldinfo[-1]["uid"] = uid
            if(koboldai_vars.worldinfo[-1]["folder"] is not None):
                koboldai_vars.wifolders_u[koboldai_vars.worldinfo[-1]["folder"]].append(koboldai_vars.worldinfo[-1])
            num += 1

    for uid in koboldai_vars.wifolders_l + [None]:
        koboldai_vars.worldinfo.append({"key": "", "keysecondary": "", "content": "", "comment": "", "folder": uid, "num": None, "init": False, "selective": False, "constant": False, "uid": None})
        while(True):
            uid = int.from_bytes(os.urandom(4), "little", signed=True)
            if(uid not in koboldai_vars.worldinfo_u):
                break
        try:
            koboldai_vars.worldinfo_u[uid] = koboldai_vars.worldinfo[-1]
        except:
            print(koboldai_vars.worldinfo)
        koboldai_vars.worldinfo[-1]["uid"] = uid
        if(koboldai_vars.worldinfo[-1]["folder"] is not None):
            koboldai_vars.wifolders_u[koboldai_vars.worldinfo[-1]["folder"]].append(koboldai_vars.worldinfo[-1])
    stablesortwi()
    koboldai_vars.worldinfo_i = [wi for wi in koboldai_vars.worldinfo if wi["init"]]

    # Save path for save button
    koboldai_vars.savedir = loadpath
    
    # Clear loadselect var
    koboldai_vars.loadselect = ""
    
    # Refresh game screen
    emit('from_server', {'cmd': 'setstoryname', 'data': koboldai_vars.laststory}, broadcast=True, room="UI_1")
    setgamesaved(True)
    sendwi()
    emit('from_server', {'cmd': 'setmemory', 'data': koboldai_vars.memory}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'setanote', 'data': koboldai_vars.authornote}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'setanotetemplate', 'data': koboldai_vars.authornotetemplate}, broadcast=True, room="UI_1")
    refresh_story()
    emit('from_server', {'cmd': 'setgamestate', 'data': 'ready'}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'hidegenseqs', 'data': ''}, broadcast=True, room="UI_1")
    print("{0}Story loaded from {1}!{2}".format(colors.GREEN, filename, colors.END))
    
    send_debug()

def load_story_v2(js):
    session['story'] = js['story_name']
    koboldai_vars.load_story(session['story'], js)
    

#==================================================================#
# Import an AIDungon game exported with Mimi's tool
#==================================================================#
def importRequest():
    importpath = fileops.getloadpath(koboldai_vars.savedir, "Select AID CAT File", [("Json", "*.json")])
    
    if(importpath):
        # Leave Edit/Memory mode before continuing
        exitModes()
        
        # Read file contents into JSON object
        file = open(importpath, "rb")
        koboldai_vars.importjs = json.load(file)
        
        # If a bundle file is being imported, select just the Adventures object
        if type(koboldai_vars.importjs) is dict and "stories" in koboldai_vars.importjs:
            koboldai_vars.importjs = koboldai_vars.importjs["stories"]
        
        # Clear Popup Contents
        emit('from_server', {'cmd': 'clearpopup', 'data': ''}, broadcast=True, room="UI_1")
        
        # Initialize vars
        num = 0
        koboldai_vars.importnum = -1
        
        # Get list of stories
        for story in koboldai_vars.importjs:
            ob = {}
            ob["num"]   = num
            if(story["title"] != "" and story["title"] != None):
                ob["title"] = story["title"]
            else:
                ob["title"] = "(No Title)"
            if(story["description"] != "" and story["description"] != None):
                ob["descr"] = story["description"]
            else:
                ob["descr"] = "(No Description)"
            if("actions" in story):
                ob["acts"]  = len(story["actions"])
            elif("actionWindow" in story):
                ob["acts"]  = len(story["actionWindow"])
            emit('from_server', {'cmd': 'addimportline', 'data': ob}, room="UI_1")
            num += 1
        
        # Show Popup
        emit('from_server', {'cmd': 'popupshow', 'data': True}, room="UI_1")

#==================================================================#
# Import an AIDungon game selected in popup
#==================================================================#
def importgame():
    if(koboldai_vars.importnum >= 0):
        # Cache reference to selected game
        ref = koboldai_vars.importjs[koboldai_vars.importnum]
        
        # Copy game contents to vars
        koboldai_vars.gamestarted = True
        
        # Support for different versions of export script
        if("actions" in ref):
            if(len(ref["actions"]) > 0):
                koboldai_vars.prompt = ref["actions"][0]["text"]
            else:
                koboldai_vars.prompt = ""
        elif("actionWindow" in ref):
            if(len(ref["actionWindow"]) > 0):
                koboldai_vars.prompt = ref["actionWindow"][0]["text"]
            else:
                koboldai_vars.prompt = ""
        else:
            koboldai_vars.prompt = ""
        koboldai_vars.memory      = ref["memory"]
        koboldai_vars.authornote  = ref["authorsNote"] if type(ref["authorsNote"]) is str else ""
        koboldai_vars.authornotetemplate = "[Author's note: <|>]"
        koboldai_vars.actions.reset()
        koboldai_vars.actions_metadata = {}
        koboldai_vars.worldinfo   = []
        koboldai_vars.worldinfo_i = []
        koboldai_vars.worldinfo_u = {}
        koboldai_vars.wifolders_d = {}
        koboldai_vars.wifolders_l = []
        koboldai_vars.wifolders_u = {uid: [] for uid in koboldai_vars.wifolders_d}
        koboldai_vars.lastact     = ""
        koboldai_vars.submission  = ""
        koboldai_vars.lastctx     = ""
        
        # Get all actions except for prompt
        if("actions" in ref):
            if(len(ref["actions"]) > 1):
                for act in ref["actions"][1:]:
                    koboldai_vars.actions.append(act["text"])
        elif("actionWindow" in ref):
            if(len(ref["actionWindow"]) > 1):
                for act in ref["actionWindow"][1:]:
                    koboldai_vars.actions.append(act["text"])
        
        # Get just the important parts of world info
        if(ref["worldInfo"] != None):
            if(len(ref["worldInfo"]) > 1):
                num = 0
                for wi in ref["worldInfo"]:
                    koboldai_vars.worldinfo.append({
                        "key": wi["keys"],
                        "keysecondary": wi.get("keysecondary", ""),
                        "content": wi["entry"],
                        "comment": wi.get("comment", ""),
                        "folder": wi.get("folder", None),
                        "num": num,
                        "init": True,
                        "selective": wi.get("selective", False),
                        "constant": wi.get("constant", False),
                        "uid": None,
                    })
                    while(True):
                        uid = int.from_bytes(os.urandom(4), "little", signed=True)
                        if(uid not in koboldai_vars.worldinfo_u):
                            break
                    koboldai_vars.worldinfo_u[uid] = koboldai_vars.worldinfo[-1]
                    koboldai_vars.worldinfo[-1]["uid"] = uid
                    if(koboldai_vars.worldinfo[-1]["folder"]) is not None:
                        koboldai_vars.wifolders_u[koboldai_vars.worldinfo[-1]["folder"]].append(koboldai_vars.worldinfo[-1])
                    num += 1

        for uid in koboldai_vars.wifolders_l + [None]:
            koboldai_vars.worldinfo.append({"key": "", "keysecondary": "", "content": "", "comment": "", "folder": uid, "num": None, "init": False, "selective": False, "constant": False, "uid": None})
            while(True):
                uid = int.from_bytes(os.urandom(4), "little", signed=True)
                if(uid not in koboldai_vars.worldinfo_u):
                    break
            koboldai_vars.worldinfo_u[uid] = koboldai_vars.worldinfo[-1]
            koboldai_vars.worldinfo[-1]["uid"] = uid
            if(koboldai_vars.worldinfo[-1]["folder"] is not None):
                koboldai_vars.wifolders_u[koboldai_vars.worldinfo[-1]["folder"]].append(koboldai_vars.worldinfo[-1])
        stablesortwi()
        koboldai_vars.worldinfo_i = [wi for wi in koboldai_vars.worldinfo if wi["init"]]
        
        # Clear import data
        koboldai_vars.importjs = {}
        
        # Reset current save
        koboldai_vars.savedir = getcwd()+"\\stories"
        
        # Refresh game screen
        koboldai_vars.laststory = None
        emit('from_server', {'cmd': 'setstoryname', 'data': koboldai_vars.laststory}, broadcast=True, room="UI_1")
        setgamesaved(False)
        sendwi()
        emit('from_server', {'cmd': 'setmemory', 'data': koboldai_vars.memory}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'setanote', 'data': koboldai_vars.authornote}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'setanotetemplate', 'data': koboldai_vars.authornotetemplate}, broadcast=True, room="UI_1")
        refresh_story()
        emit('from_server', {'cmd': 'setgamestate', 'data': 'ready'}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'hidegenseqs', 'data': ''}, broadcast=True, room="UI_1")

#==================================================================#
# Import an aidg.club prompt and start a new game with it.
#==================================================================#
def importAidgRequest(id):    
    exitModes()
    
    urlformat = "https://prompts.aidg.club/api/"
    req = requests.get(urlformat+id)

    if(req.status_code == 200):
        js = req.json()
        
        # Import game state
        koboldai_vars.gamestarted = True
        koboldai_vars.prompt      = js["promptContent"]
        koboldai_vars.memory      = js["memory"]
        koboldai_vars.authornote  = js["authorsNote"]
        koboldai_vars.authornotetemplate = "[Author's note: <|>]"
        koboldai_vars.actions.reset()
        koboldai_vars.actions_metadata = {}
        koboldai_vars.worldinfo   = []
        koboldai_vars.worldinfo_i = []
        koboldai_vars.worldinfo_u = {}
        koboldai_vars.wifolders_d = {}
        koboldai_vars.wifolders_l = []
        koboldai_vars.wifolders_u = {uid: [] for uid in koboldai_vars.wifolders_d}
        koboldai_vars.lastact     = ""
        koboldai_vars.submission  = ""
        koboldai_vars.lastctx     = ""
        
        num = 0
        for wi in js["worldInfos"]:
            koboldai_vars.worldinfo.append({
                "key": wi["keys"],
                "keysecondary": wi.get("keysecondary", ""),
                "content": wi["entry"],
                "comment": wi.get("comment", ""),
                "folder": wi.get("folder", None),
                "num": num,
                "init": True,
                "selective": wi.get("selective", False),
                "constant": wi.get("constant", False),
                "uid": None,
            })
            while(True):
                uid = int.from_bytes(os.urandom(4), "little", signed=True)
                if(uid not in koboldai_vars.worldinfo_u):
                    break
            koboldai_vars.worldinfo_u[uid] = koboldai_vars.worldinfo[-1]
            koboldai_vars.worldinfo[-1]["uid"] = uid
            if(koboldai_vars.worldinfo[-1]["folder"]) is not None:
                koboldai_vars.wifolders_u[koboldai_vars.worldinfo[-1]["folder"]].append(koboldai_vars.worldinfo[-1])
            num += 1

        for uid in koboldai_vars.wifolders_l + [None]:
            koboldai_vars.worldinfo.append({"key": "", "keysecondary": "", "content": "", "comment": "", "folder": uid, "num": None, "init": False, "selective": False, "constant": False, "uid": None})
            while(True):
                uid = int.from_bytes(os.urandom(4), "little", signed=True)
                if(uid not in koboldai_vars.worldinfo_u):
                    break
            koboldai_vars.worldinfo_u[uid] = koboldai_vars.worldinfo[-1]
            koboldai_vars.worldinfo[-1]["uid"] = uid
            if(koboldai_vars.worldinfo[-1]["folder"] is not None):
                koboldai_vars.wifolders_u[koboldai_vars.worldinfo[-1]["folder"]].append(koboldai_vars.worldinfo[-1])
        stablesortwi()
        koboldai_vars.worldinfo_i = [wi for wi in koboldai_vars.worldinfo if wi["init"]]

        # Reset current save
        koboldai_vars.savedir = getcwd()+"\\stories"
        
        # Refresh game screen
        koboldai_vars.laststory = None
        emit('from_server', {'cmd': 'setstoryname', 'data': koboldai_vars.laststory}, broadcast=True, room="UI_1")
        setgamesaved(False)
        sendwi()
        emit('from_server', {'cmd': 'setmemory', 'data': koboldai_vars.memory}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'setanote', 'data': koboldai_vars.authornote}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'setanotetemplate', 'data': koboldai_vars.authornotetemplate}, broadcast=True, room="UI_1")
        refresh_story()
        emit('from_server', {'cmd': 'setgamestate', 'data': 'ready'}, broadcast=True, room="UI_1")

#==================================================================#
#  Import World Info JSON file
#==================================================================#
def wiimportrequest():
    importpath = fileops.getloadpath(koboldai_vars.savedir, "Select World Info File", [("Json", "*.json")])
    if(importpath):
        file = open(importpath, "rb")
        js = json.load(file)
        if(len(js) > 0):
            # If the most recent WI entry is blank, remove it.
            if(not koboldai_vars.worldinfo[-1]["init"]):
                del koboldai_vars.worldinfo[-1]
            # Now grab the new stuff
            num = len(koboldai_vars.worldinfo)
            for wi in js:
                koboldai_vars.worldinfo.append({
                    "key": wi["keys"],
                    "keysecondary": wi.get("keysecondary", ""),
                    "content": wi["entry"],
                    "comment": wi.get("comment", ""),
                    "folder": wi.get("folder", None),
                    "num": num,
                    "init": True,
                    "selective": wi.get("selective", False),
                    "constant": wi.get("constant", False),
                    "uid": None,
                })
                while(True):
                    uid = int.from_bytes(os.urandom(4), "little", signed=True)
                    if(uid not in koboldai_vars.worldinfo_u):
                        break
                koboldai_vars.worldinfo_u[uid] = koboldai_vars.worldinfo[-1]
                koboldai_vars.worldinfo[-1]["uid"] = uid
                if(koboldai_vars.worldinfo[-1]["folder"]) is not None:
                    koboldai_vars.wifolders_u[koboldai_vars.worldinfo[-1]["folder"]].append(koboldai_vars.worldinfo[-1])
                num += 1
            for uid in [None]:
                koboldai_vars.worldinfo.append({"key": "", "keysecondary": "", "content": "", "comment": "", "folder": uid, "num": None, "init": False, "selective": False, "constant": False, "uid": None})
                while(True):
                    uid = int.from_bytes(os.urandom(4), "little", signed=True)
                    if(uid not in koboldai_vars.worldinfo_u):
                        break
                koboldai_vars.worldinfo_u[uid] = koboldai_vars.worldinfo[-1]
                koboldai_vars.worldinfo[-1]["uid"] = uid
                if(koboldai_vars.worldinfo[-1]["folder"] is not None):
                    koboldai_vars.wifolders_u[koboldai_vars.worldinfo[-1]["folder"]].append(koboldai_vars.worldinfo[-1])
        
        if not koboldai_vars.quiet:
            print("{0}".format(koboldai_vars.worldinfo[0]))
                
        # Refresh game screen
        setgamesaved(False)
        sendwi()

#==================================================================#
#  Starts a new story
#==================================================================#
def newGameRequest(): 
    # Leave Edit/Memory mode before continuing
    exitModes()
    
    # Clear vars values
    koboldai_vars.gamestarted = False
    koboldai_vars.prompt      = ""
    koboldai_vars.memory      = ""
    koboldai_vars.actions.reset()
    koboldai_vars.actions_metadata = {}
    
    koboldai_vars.authornote  = ""
    koboldai_vars.authornotetemplate = koboldai_vars.setauthornotetemplate
    koboldai_vars.worldinfo   = []
    koboldai_vars.worldinfo_i = []
    koboldai_vars.worldinfo_u = {}
    koboldai_vars.wifolders_d = {}
    koboldai_vars.wifolders_l = []
    koboldai_vars.lastact     = ""
    koboldai_vars.submission  = ""
    koboldai_vars.lastctx     = ""
    
    # Reset current save
    koboldai_vars.savedir = getcwd()+"\\stories"
    
    # Refresh game screen
    koboldai_vars.laststory = None
    emit('from_server', {'cmd': 'setstoryname', 'data': koboldai_vars.laststory}, broadcast=True, room="UI_1")
    setgamesaved(True)
    sendwi()
    emit('from_server', {'cmd': 'setmemory', 'data': koboldai_vars.memory}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'setanote', 'data': koboldai_vars.authornote}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'setanotetemplate', 'data': koboldai_vars.authornotetemplate}, broadcast=True, room="UI_1")
    setStartState()

def randomGameRequest(topic, memory=""): 
    if(koboldai_vars.noai):
        newGameRequest()
        koboldai_vars.memory = memory
        emit('from_server', {'cmd': 'setmemory', 'data': koboldai_vars.memory}, broadcast=True, room="UI_1")
        return
    koboldai_vars.recentrng = topic
    koboldai_vars.recentrngm = memory
    newGameRequest()
    setgamesaved(False)
    _memory = memory
    if(len(memory) > 0):
        _memory = memory.rstrip() + "\n\n"
    koboldai_vars.memory      = _memory + "You generate the following " + topic + " story concept :"
    koboldai_vars.lua_koboldbridge.feedback = None
    actionsubmit("", force_submit=True, force_prompt_gen=True)
    koboldai_vars.memory      = memory
    emit('from_server', {'cmd': 'setmemory', 'data': koboldai_vars.memory}, broadcast=True, room="UI_1")

def final_startup():
    # Prevent tokenizer from taking extra time the first time it's used
    def __preempt_tokenizer():
        if("tokenizer" not in globals()):
            return
        utils.decodenewlines(tokenizer.decode([25678, 559]))
        tokenizer.encode(utils.encodenewlines("eunoia"))
    threading.Thread(target=__preempt_tokenizer).start()

    # Load soft prompt specified by the settings file, if applicable
    if(path.exists("settings/" + getmodelname().replace('/', '_') + ".settings")):
        file = open("settings/" + getmodelname().replace('/', '_') + ".settings", "r")
        js   = json.load(file)
        if(koboldai_vars.allowsp and "softprompt" in js and type(js["softprompt"]) is str and all(q not in js["softprompt"] for q in ("..", ":")) and (len(js["softprompt"]) == 0 or all(js["softprompt"][0] not in q for q in ("/", "\\")))):
            spRequest(js["softprompt"])
        else:
            koboldai_vars.spfilename = ""
        file.close()

    # Precompile TPU backend if required
    if(koboldai_vars.use_colab_tpu or koboldai_vars.model in ("TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX")):
        soft_tokens = tpumtjgetsofttokens()
        if(koboldai_vars.dynamicscan or (not koboldai_vars.nogenmod and koboldai_vars.has_genmod)):
            threading.Thread(
                target=tpu_mtj_backend.infer_dynamic,
                args=(np.tile(np.uint32((23403, 727, 20185)), (koboldai_vars.numseqs, 1)),),
                kwargs={
                    "soft_embeddings": koboldai_vars.sp,
                    "soft_tokens": soft_tokens,
                    "gen_len": 1,
                    "use_callback": False,
                    "numseqs": koboldai_vars.numseqs,
                    "excluded_world_info": list(set() for _ in range(koboldai_vars.numseqs)),
                },
            ).start()
        else:
            threading.Thread(
                target=tpu_mtj_backend.infer_static,
                args=(np.uint32((23403, 727, 20185)),),
                kwargs={
                    "soft_embeddings": koboldai_vars.sp,
                    "soft_tokens": soft_tokens,
                    "gen_len": 1,
                    "numseqs": koboldai_vars.numseqs,
                },
            ).start()

def send_debug():
    if koboldai_vars.debug:
        debug_info = ""
        try:
            debug_info = "{}Newline Mode: {}\n".format(debug_info, koboldai_vars.newlinemode)
        except:
            pass
        try:
            debug_info = "{}Action Length: {}\n".format(debug_info, koboldai_vars.actions.get_last_key())
        except:
            pass
        try:
            debug_info = "{}Actions Metadata Length: {}\n".format(debug_info, max(koboldai_vars.actions_metadata) if len(koboldai_vars.actions_metadata) > 0 else 0)
        except:
            pass
        try:
            debug_info = "{}Actions: {}\n".format(debug_info, [k for k in koboldai_vars.actions])
        except:
            pass
        try:
            debug_info = "{}Actions Metadata: {}\n".format(debug_info, [k for k in koboldai_vars.actions_metadata])
        except:
            pass
        try:
            debug_info = "{}Last Action: {}\n".format(debug_info, koboldai_vars.actions[koboldai_vars.actions.get_last_key()])
        except:
            pass
        try:
            debug_info = "{}Last Metadata: {}\n".format(debug_info, koboldai_vars.actions_metadata[max(koboldai_vars.actions_metadata)])
        except:
            pass

        emit('from_server', {'cmd': 'debug_info', 'data': debug_info}, broadcast=True, room="UI_1")


#==================================================================#
# UI V2 CODE
#==================================================================#
@app.route('/new_ui')
def new_ui_index():
    if 'story' in session:
        if session['story'] not in koboldai_vars.story_list():
            session['story'] = 'default'
    return render_template('index_new.html', settings=gensettings.gensettingstf if koboldai_vars.model != "InferKit" else gensettings.gensettingsik )

def ui2_connect():
    #Send all variables to client
    koboldai_vars.send_to_ui()
    
    pass

#==================================================================#
# File Popup options
#==================================================================#
@socketio.on('upload_file')
def upload_file(data):
    print("upload_file {}".format(data['filename']))
    if 'current_folder' in session:
        path = os.path.abspath(os.path.join(session['current_folder'], data['filename']).replace("\\", "/")).replace("\\", "/")
        print("Want to save to {}".format(path))
        if 'popup_jailed_dir' not in session:
            print("Someone is trying to upload a file to your server. Blocked.")
        elif session['popup_jailed_dir'] is None:
            if os.path.exists(path):
                emit("error_popup", "The file already exists. Please delete it or rename the file before uploading", room="UI_2");
            else:
                with open(path, "wb") as f:
                    f.write(data['data'])
                get_files_folders(session['current_folder'])
        elif session['popup_jailed_dir'] in session['current_folder']:
            if os.path.exists(path):
                emit("error_popup", "The file already exists. Please delete it or rename the file before uploading", room="UI_2");
            else:
                with open(path, "wb") as f:
                    f.write(data['data'])
                get_files_folders(session['current_folder'])

@socketio.on('popup_change_folder')
def popup_change_folder(data):
    print("Doing popup change folder: {}".format(data))
    if 'popup_jailed_dir' not in session:
        print("Someone is trying to get at files in your server. Blocked.")
        return
    if session['popup_jailed_dir'] is None:
        get_files_folders(data)
    elif session['popup_jailed_dir'] in data:
        get_files_folders(data)
    else:
        print("User is trying to get at files in your server outside the jail. Blocked. Jailed Dir: {}  Requested Dir: {}".format(session['popup_jailed_dir'], data))

@socketio.on('popup_rename')
def popup_rename(data):
    if 'popup_renameable' not in session:
        print("Someone is trying to rename a file in your server. Blocked.")
        return
    if not session['popup_renameable']:
        print("Someone is trying to rename a file in your server. Blocked.")
        return
    
    if session['popup_jailed_dir'] is None:
        os.rename(data['file'], data['new_name'])
        get_files_folders(os.path.dirname(data['file']))
    elif session['popup_jailed_dir'] in data:
        os.rename(data['file'], data['new_name'])
        get_files_folders(os.path.dirname(data['file']))
    else:
        print("User is trying to rename files in your server outside the jail. Blocked. Jailed Dir: {}  Requested Dir: {}".format(session['popup_jailed_dir'], data['file']))

@socketio.on('popup_delete')
def popup_delete(data):
    if 'popup_deletable' not in session:
        print("Someone is trying to delete a file in your server. Blocked.")
        return
    if not session['popup_deletable']:
        print("Someone is trying to delete a file in your server. Blocked.")
        return
    
    if session['popup_jailed_dir'] is None:
        import shutil
        if os.path.isdir(data):
            shutil.rmtree(data)
        else:
            os.remove(data)
        path = os.path.abspath(data).replace("\\", "/")
        if path[-1] == "/":
            path = path[:-1]
        path = "/".join(path.split("/")[:-1])
        get_files_folders(path)
    elif session['popup_jailed_dir'] in data:
        import shutil
        if os.path.isdir(data):
            shutil.rmtree(data)
        else:
            os.remove(data)
        path = os.path.abspath(data).replace("\\", "/")
        if path[-1] == "/":
            path = path[:-1]
        path = "/".join(path.split("/")[:-1])
        get_files_folders(path)
    else:
        print("User is trying to delete files in your server outside the jail. Blocked. Jailed Dir: {}  Requested Dir: {}".format(session['popup_jailed_dir'], data))

@socketio.on('popup_edit')
def popup_edit(data):
    if 'popup_editable' not in session:
        print("Someone is trying to edit a file in your server. Blocked.")
        return
    if not session['popup_editable']:
        print("Someone is trying to edit a file in your server. Blocked.")
        return
    
    if session['popup_jailed_dir'] is None:
        emit("popup_edit_file", {"file": data, "text": open(data, 'r', encoding='utf-8').read()});
    elif session['popup_jailed_dir'] in data:
        emit("popup_edit_file", {"file": data, "text": open(data, 'r', encoding='utf-8').read()});
    else:
        print("User is trying to delete files in your server outside the jail. Blocked. Jailed Dir: {}  Requested Dir: {}".format(session['popup_jailed_dir'], data))

@socketio.on('popup_change_file')
def popup_change_file(data):
    if 'popup_editable' not in session:
        print("Someone is trying to edit a file in your server. Blocked.")
        return
    if not session['popup_editable']:
        print("Someone is trying to edit a file in your server. Blocked.")
        return
    
    if session['popup_jailed_dir'] is None:
        with open(data['file'], 'w') as f:
            f.write(data['data'])
    elif session['popup_jailed_dir'] in data['file']:
        with open(data['file'], 'w') as f:
            f.write(data['data'])
    else:
        print("User is trying to delete files in your server outside the jail. Blocked. Jailed Dir: {}  Requested Dir: {}".format(session['popup_jailed_dir'], data))

def file_popup(popup_title, starting_folder, return_event, upload=True, jailed=True, folder_only=True, renameable=False, deleteable=False, 
                                                           editable=False, show_breadcrumbs=True, item_check=None, show_hidden=False,
                                                           valid_only=False, hide_extention=False):
    #starting_folder = The folder we're going to get folders and/or items from
    #return_event = the socketio event that will be emitted when the load button is clicked
    #jailed = if set to true will look for the session variable jailed_folder and prevent navigation outside of that folder
    #folder_only = will only show folders, no files
    #deletable = will show the delete icons/methods.
    #editable = will show the edit icons/methods
    #show_breadcrumbs = will show the breadcrumbs at the top of the screen
    #item_check will call this function to check if the item is valid as a selection if not none. Will pass absolute directory as only argument to function
    #show_hidden = ... really, you have to ask?
    #valid_only = only show valid files
    #hide_extention = hide extensions
    if jailed:
        session['popup_jailed_dir'] = os.path.abspath(starting_folder).replace("\\", "/")
    else:
        session['popup_jailed_dir'] = None
    session['popup_deletable'] = deleteable
    session['popup_renameable'] = renameable
    session['popup_editable'] = editable
    session['popup_show_hidden'] = show_hidden
    session['popup_item_check'] = item_check
    session['popup_folder_only'] = folder_only
    session['popup_show_breadcrumbs'] = show_breadcrumbs
    session['upload'] = upload
    session['valid_only'] = valid_only
    session['hide_extention'] = hide_extention
    
    socketio.emit("load_popup", {"popup_title": popup_title, "call_back": return_event, "renameable": renameable, "deleteable": deleteable, "editable": editable, 'upload': upload}, broadcast=False, room="UI_2")
    socketio.emit("load_popup", {"popup_title": popup_title, "call_back": return_event, "renameable": renameable, "deleteable": deleteable, "editable": editable, 'upload': upload}, broadcast=True, room="UI_1")
    
    get_files_folders(starting_folder)
    
def get_files_folders(starting_folder):
    import stat
    session['current_folder'] = starting_folder
    item_check = session['popup_item_check']
    show_breadcrumbs = session['popup_show_breadcrumbs']
    show_hidden = session['popup_show_hidden']
    folder_only = session['popup_folder_only']
    valid_only = session['valid_only']
    hide_extention = session['hide_extention']
    
    if starting_folder == 'This PC':
        breadcrumbs = [['This PC', 'This PC']]
        items = [["{}:/".format(chr(i)), "{}:\\".format(chr(i))] for i in range(65, 91) if os.path.exists("{}:".format(chr(i)))]
    else:
        path = os.path.abspath(starting_folder).replace("\\", "/")
        if path[-1] == "/":
            path = path[:-1]
        breadcrumbs = []
        for i in range(len(path.split("/"))):
            breadcrumbs.append(["/".join(path.split("/")[:i+1]),
                                 path.split("/")[i]])
        if len(breadcrumbs) == 1:
            breadcrumbs = [["{}:/".format(chr(i)), "{}:\\".format(chr(i))] for i in range(65, 91) if os.path.exists("{}:".format(chr(i)))]
        else:
            if len([["{}:/".format(chr(i)), "{}:\\".format(chr(i))] for i in range(65, 91) if os.path.exists("{}:".format(chr(i)))]) > 0:
                breadcrumbs.insert(0, ['This PC', 'This PC'])
        
        #if we're jailed, remove the stuff before the jail from the breadcrumbs
        if session['popup_jailed_dir'] is not None:
            
            breadcrumbs = breadcrumbs[len(session['popup_jailed_dir'].split("/")):]
        
        folders = []
        files = []
        base_path = os.path.abspath(starting_folder).replace("\\", "/")
        for item in os.listdir(base_path):
            item_full_path = os.path.join(base_path, item).replace("\\", "/")
            if hasattr(os.stat(item_full_path), "st_file_attributes"):
                hidden = bool(os.stat(item_full_path).st_file_attributes & stat.FILE_ATTRIBUTE_HIDDEN)
            else:
                hidden = item[0] == "."
            if item_check is None:
                valid_selection = True
            else:
                valid_selection = item_check(item_full_path)
                
            if (show_hidden and hidden) or not hidden:
                if os.path.isdir(os.path.join(base_path, item)):
                    folders.append([True, item_full_path, item,  valid_selection])
                else:
                    if hide_extention:
                        item = ".".join(item.split(".")[:-1])
                    if valid_only:
                        if valid_selection:
                            files.append([False, item_full_path, item,  valid_selection])
                    else:
                        files.append([False, item_full_path, item,  valid_selection])
                        
        items = folders
        if not folder_only:
            items += files
            
    socketio.emit("popup_items", items, broadcast=False, include_self=True, room="UI_2")
    socketio.emit("popup_items", items, broadcast=True, include_self=True, room="UI_1")
    if show_breadcrumbs:
        socketio.emit("popup_breadcrumbs", breadcrumbs, broadcast=False, room="UI_2")
        socketio.emit("popup_breadcrumbs", breadcrumbs, broadcast=True, room="UI_1")

#==================================================================#
# Event triggered when browser SocketIO detects a variable change
#==================================================================#
@socketio.on('var_change')
def UI_2_var_change(data):
    classname = data['ID'].split("_")[0]
    name = data['ID'][len(classname)+1:]
    classname += "_settings"
    
    #Need to fix the data type of value to match the module
    if type(getattr(koboldai_vars, name)) == int:
        value = int(data['value'])
    elif type(getattr(koboldai_vars, name)) == float:
        value = float(data['value'])
    elif type(getattr(koboldai_vars, name)) == bool:
        value = bool(data['value'])
    elif type(getattr(koboldai_vars, name)) == str:
        value = str(data['value'])
    else:
        print("Unknown Type {} = {}".format(name, type(getattr(koboldai_vars, name))))
    
    #print("Setting {} to {} as type {}".format(name, value, type(value)))
    setattr(koboldai_vars, name, value)
    
    #Now let's save except for story changes
    if classname != "story_settings":
        with open("settings/{}.v2_settings".format(classname), "w") as settings_file:
            settings_file.write(getattr(koboldai_vars, "_{}".format(classname)).to_json())
    
    return {'id': data['ID'], 'status': "Saved"}
    
#==================================================================#
# Saving Story
#==================================================================#
@socketio.on('save_story')
def UI_2_save_story(data):
    json_data = koboldai_vars.to_json('story_settings')
    save_name = koboldai_vars.story_name if koboldai_vars.story_name is not None else "untitled"
    with open("stories/{}_v2.json".format(save_name), "w") as settings_file:
        settings_file.write(json_data)
    koboldai_vars.gamesaved = True
    
    
#==================================================================#
# Event triggered when Selected Text is edited
#==================================================================#
@socketio.on('Set Selected Text')
def UI_2_Set_Selected_Text(data):
    print("Updating Selected Text: {}".format(data))
    koboldai_vars.actions[int(data['id'])] = data['text']

#==================================================================#
# Event triggered when Option is Selected
#==================================================================#
@socketio.on('Use Option Text')
def UI_2_Set_Selected_Text(data):
    print("Using Option Text: {}".format(data))
    koboldai_vars.actions.use_option(int(data['option']), action_step=int(data['chunk']))


#==================================================================#
# Event triggered when user clicks the submit button
#==================================================================#
@socketio.on('submit')
def UI_2_submit(data):
    koboldai_vars.lua_koboldbridge.feedback = None
    koboldai_vars.recentrng = koboldai_vars.recentrngm = None
    actionsubmit(data['data'], actionmode=koboldai_vars.actionmode)
    
#==================================================================#
# Event triggered when user clicks the pin button
#==================================================================#
@socketio.on('Pinning')
def UI_2_Pinning(data):
    koboldai_vars.actions.toggle_pin(int(data['chunk']), int(data['option']))
    
#==================================================================#
# Event triggered when user clicks the back button
#==================================================================#
@socketio.on('back')
def UI_2_back(data):
    print("back")
    ignore = koboldai_vars.actions.pop()
    
#==================================================================#
# Event triggered when user clicks the redo button
#==================================================================#
@socketio.on('redo')
def UI_2_redo(data):
    if len(koboldai_vars.actions.get_current_options()) == 1:
        koboldai_vars.actions.use_option(0)

#==================================================================#
# Event triggered when user clicks the redo button
#==================================================================#
@socketio.on('retry')
def UI_2_retry(data):
    koboldai_vars.actions.clear_unused_options()
    koboldai_vars.lua_koboldbridge.feedback = None
    koboldai_vars.recentrng = koboldai_vars.recentrngm = None
    actionsubmit("", actionmode=koboldai_vars.actionmode)
    
#==================================================================#
# Event triggered when user clicks the load model button
#==================================================================#
@socketio.on('load_model_button')
def UI_2_load_model_button(data):
    sendModelSelection()
    
#==================================================================#
# Event triggered when user clicks the a model
#==================================================================#
@socketio.on('select_model')
def UI_2_load_model_button(data):
    print(data)
    
    #We've selected a menu
    if data['model'] in model_menu:
        sendModelSelection(menu=data['model'])
    #We've selected a custom line
    elif data['menu'] in ("NeoCustom", "GPT2Custom"):
        get_model_info(data['menu'], directory=data['display_name'])
    #We've selected a custom menu
    elif data['model'] in ("NeoCustom", "GPT2Custom"):
        sendModelSelection(menu=data['model'], folder="./models")
    else:
        #We now have some model we want to potentially load.
        #First we need to send the client the model parameters (layers, etc)
        get_model_info(data['model'])

#==================================================================#
# Event triggered when user loads a model
#==================================================================#
@socketio.on('load_model')
def UI_2_load_model(data):
    print(data)
    if not os.path.exists("settings/"):
        os.mkdir("settings")
    changed = True
    if not utils.HAS_ACCELERATE:
        data['disk_layers'] = "0"
    if os.path.exists("settings/" + data['model'].replace('/', '_') + ".breakmodel"):
        with open("settings/" + data['model'].replace('/', '_') + ".breakmodel", "r") as file:
            file_data = file.read().split('\n')[:2]
            if len(file_data) < 2:
                file_data.append("0")
            gpu_layers, disk_layers = file_data
            if gpu_layers == data['gpu_layers'] and disk_layers == data['disk_layers']:
                changed = False
    if changed:
        f = open("settings/" + data['model'].replace('/', '_') + ".breakmodel", "w")
        f.write(data['gpu_layers'] + '\n' + data['disk_layers'])
        f.close()
    koboldai_vars.colaburl = data['url'] + "/request"
    koboldai_vars.model = data['model']
    koboldai_vars.custmodpth = data['path']
    print("loading Model")
    load_model(use_gpu=data['use_gpu'], gpu_layers=data['gpu_layers'], disk_layers=data['disk_layers'], online_model=data['online_model'])

#==================================================================#
# Event triggered when load story is clicked
#==================================================================#
@socketio.on('load_story_list')
def UI_2_load_story_list(data):
    file_popup("Select Story to Load", "./stories", "load_story", upload=True, jailed=True, folder_only=False, renameable=True, 
                                                                  deleteable=True, show_breadcrumbs=True, item_check=valid_story,
                                                                  valid_only=True, hide_extention=True)

def valid_story(file):
        if file.endswith(".json"):
            with open(file, "r") as f:
                try:
                    js = json.load(f)
                except:
                    pass
                    return False
                
                return 'actions' in js

#==================================================================#
# Event triggered on load story
#==================================================================#
@socketio.on('load_story')
def UI_2_load_story(file):
    print("loading {}".format(file))
    loadRequest(file)


#==================================================================#
# Event triggered to rely a message
#==================================================================#
@socketio.on('relay')
def UI_2_relay(data):
    socketio.emit(data[0], data[1], **data[2])


#==================================================================#
# Test
#==================================================================#
@app.route("/actions")
def show_actions():
    return koboldai_vars.actions.actions
    
@app.route("/vars")
def show_vars():
    json_data = {}
    json_data['story_settings'] = json.loads(koboldai_vars.to_json("story_settings"))
    json_data['model_settings'] = json.loads(koboldai_vars.to_json("model_settings"))
    json_data['user_settings'] = json.loads(koboldai_vars.to_json("user_settings"))
    json_data['system_settings'] = json.loads(koboldai_vars.to_json("system_settings"))
    return json_data

    

#==================================================================#
#  Final startup commands to launch Flask app
#==================================================================#
print("", end="", flush=True)
if __name__ == "__main__":
    print("{0}\nStarting webserver...{1}".format(colors.GREEN, colors.END), flush=True)

    general_startup()
    patch_transformers()
    #show_select_model_list()
    if koboldai_vars.model == "" or koboldai_vars.model is None:
        koboldai_vars.model = "ReadOnly"
    load_model(initial_load=True)

    # Start Flask/SocketIO (Blocking, so this must be last method!)
    port = args.port if "port" in args and args.port is not None else 5000
    koboldai_settings.port = port
    
    if(koboldai_vars.host):
        if(args.localtunnel):
            import subprocess, shutil
            localtunnel = subprocess.Popen([shutil.which('lt'), '-p', str(port), 'http'], stdout=subprocess.PIPE)
            attempts = 0
            while attempts < 10:
                try:
                    cloudflare = str(localtunnel.stdout.readline())
                    cloudflare = (re.search("(?P<url>https?:\/\/[^\s]+loca.lt)", cloudflare).group("url"))
                    break
                except:
                    attempts += 1
                    time.sleep(3)
                    continue
            if attempts == 10:
                print("LocalTunnel could not be created, falling back to cloudflare...")
                from flask_cloudflared import _run_cloudflared
                cloudflare = _run_cloudflared(port)
        elif(args.ngrok):
            from flask_ngrok import _run_ngrok
            cloudflare = _run_ngrok()
        elif(args.remote):
           from flask_cloudflared import _run_cloudflared
           cloudflare = _run_cloudflared(port)
        if(args.localtunnel or args.ngrok or args.remote):
            with open('cloudflare.log', 'w') as cloudflarelog:
                cloudflarelog.write("KoboldAI has finished loading and is available at the following link : " + cloudflare)
                print(format(colors.GREEN) + "KoboldAI has finished loading and is available at the following link : " + cloudflare + format(colors.END))
        else:
            print("{0}Webserver has started, you can now connect to this machine at port {1}{2}"
                  .format(colors.GREEN, port, colors.END))
        koboldai_vars.serverstarted = True
        socketio.run(app, host='0.0.0.0', port=port)
    else:
        if args.unblock:
            import webbrowser
            webbrowser.open_new('http://localhost:{0}'.format(port))
            print("{0}Server started!\nYou may now connect with a browser at http://127.0.0.1:{1}/{2}"
                  .format(colors.GREEN, port, colors.END))
            koboldai_vars.serverstarted = True
            socketio.run(app, port=port, host='0.0.0.0')
        else:
            try:
                from flaskwebgui import FlaskUI
                koboldai_vars.serverstarted = True
                koboldai_vars.flaskwebgui = True
                FlaskUI(app, socketio=socketio, start_server="flask-socketio", maximized=True, close_server_on_exit=True).run()
            except:
                import webbrowser
                webbrowser.open_new('http://localhost:{0}'.format(port))
                print("{0}Server started!\nYou may now connect with a browser at http://127.0.0.1:{1}/{2}"
                        .format(colors.GREEN, port, colors.END))
                koboldai_vars.serverstarted = True
                socketio.run(app, port=port)

else:
    general_startup()
    patch_transformers()
    #show_select_model_list()
    if koboldai_vars.model == "" or koboldai_vars.model is None:
        koboldai_vars.model = "ReadOnly"
    load_model(initial_load=True)
    koboldai_settings.port = args.port if "port" in args and args.port is not None else 5000
    print("{0}\nServer started in WSGI mode!{1}".format(colors.GREEN, colors.END), flush=True)
    
