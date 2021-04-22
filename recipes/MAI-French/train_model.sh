#!/bin/bash

if [ -f ~/.bashrc ]; then . ~/.bashrc; fi

# PATHS
home_path="/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/lpierron"
corpus_path=${home_path}/mailabs

output_path="${home_path}/Models/LJSpeech/"
phoneme_cache_path="/tmp/tts/phoneme_cache/"
dataset_path="${corpus_path}/monsieur_lecoq.tar.xz"

# Make a wworking directory
mkdir -p /tmp/tts
# copy MAILABS dataset
cp -u ${dataset_path} /tmp/tts
# decompress dataset
tar -xJfk /tmp/tts/monsieur_lecoq.tar.xz -C /tmp/tts
# Copy phoneme cache
rsync -a "${home_path}/Models/phoneme_cache_fr_ezwa/" "${phoneme_cache_path}"

# Testing avalibality of GPUs
nvidia-smi -L
gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -s -d,)

# training ....
# change the GPU id if needed
conda activate tf
CUDA_VISIBLE_DEVICES="$gpus" python ../../TTS/train_tacotron.py --config_path model_config.json
# train vocoder ...
# CUDA_VISIBLE_DEVICES="0" python TTS/vocoder/train.py --config_path vocoder_config.json
