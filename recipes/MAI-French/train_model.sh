#!/bin/bash -l

eval $(/home/lpierron/.linuxbrew/bin/brew shellenv)

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/lpierron/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/lpierron/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/lpierron/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/lpierron/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

# usage: train_tacotron.py [-h] --continue_path CONTINUE_PATH [--restore_path RESTORE_PATH]
#                          --config_path CONFIG_PATH [--debug DEBUG] [--rank RANK]
#                          [--group_id GROUP_ID]

# optional arguments:
#   -h, --help            show this help message and exit
#   --continue_path CONTINUE_PATH
#                         Training output folder to continue training. Use to continue a training. If
#                         it is used, "config_path" is ignored.
#   --restore_path RESTORE_PATH
#                         Model file to be restored. Use to finetune a model.
#   --config_path CONFIG_PATH
#                         Path to config file for training.
#   --debug DEBUG         Do not verify commit integrity to run training.
#   --rank RANK           DISTRIBUTED: process rank for distributed training.
#   --group_id GROUP_ID   DISTRIBUTED: process group id.

# Copy corpus in /tmp/tts to accelerate training
OPT_COPY='false'
# Distribute training on multiple GPUS
OPT_GPU='false'
# Test the creation of config.json
OPT_DRYRUN='false'

# Process all options supplied on the command line
while getopts ':cgd' 'OPTKEY'; do
    case ${OPTKEY} in
        'c')
            # Update the value of the option c flag we defined above
            OPT_COPY='true'
            ;;
        'g')
            # Update the value of the option g flag we defined above
            OPT_GPU='true'
            ;;
        'd')
            # Update the value of the option g flag we defined above
            OPT_DRYRUN='true'
            ;;
        '?')
            echo "INVALID OPTION -- ${OPTARG}" >&2
            exit 1
            ;;
        ':')
            echo "MISSING ARGUMENT for option -- ${OPTARG}" >&2
            exit 1
            ;;
        *)
            echo "UNIMPLEMENTED OPTION -- ${OPTKEY}" >&2
            exit 1
            ;;
    esac
done

# [optional] Remove all options processed by getopts.
shift $(( OPTIND - 1 ))
[[ "${1}" == "--" ]] && shift

if [ "$#" -eq 1 ] && [ -d "$1" ]; then
  continue="--continue_path $1"
elif [ "$#" -ne 0 ]; then
  echo "Usage: $0 [CONTINUE_PATH]" >&2
  exit 1
fi

# PATHS
home_path="/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/lpierron"
corpus_path=${home_path}/mailabs/fr_FR

output_path="${home_path}/Models/LJSpeech/"
phoneme_cache_path="/tmp/tts/phoneme_cache/"
dataset_path="${corpus_path}/monsieur_lecoq.tar.xz"

# Make a wworking directory
mkdir -p /tmp/tts
# copy MAILABS dataset
cp -u ${dataset_path} /tmp/tts
# decompress dataset
tar -xJf /tmp/tts/monsieur_lecoq.tar.xz -C /tmp/tts  --skip-old-files
# Copy phoneme cache
rsync -a "${home_path}/Models/phoneme_cache_fr_ezwa/" "${phoneme_cache_path}"

# Testing avalibality of GPUs
nvidia-smi -L
gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -s -d,)

# training ....
# change the GPU id if needed
conda activate tts

if ${OPT_GPU}; then
  CUDA_VISIBLE_DEVICES="$gpus"\
  python3 /home/lpierron/Mozilla_TTS/COQUI-TTS/TTS/TTS/bin/distribute.py \
        --script /home/lpierron/Mozilla_TTS/COQUI-TTS/TTS/TTS/bin/train_tacotron.py  \
                --config_path model_config.json${continue}
else
  CUDA_VISIBLE_DEVICES="0" python3 /home/lpierron/Mozilla_TTS/COQUI-TTS/TTS/TTS/bin/train_tacotron.py  \
          --config_path model_config.json ${continue}
fi

# train vocoder ...
# CUDA_VISIBLE_DEVICES="0" python TTS/vocoder/train.py --config_path vocoder_config.json
