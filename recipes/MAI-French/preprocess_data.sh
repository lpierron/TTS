 #!/bin/bash

 # PATHS
 home_path="/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/lpierron"
 corpus_path=${home_path}/mailabs

 output_path="${home_path}/Models/LJSpeech/"
 phoneme_cache_path="/tmp/tts/phoneme_cache_fr_ezwa"
 dataset_path="${corpus_path}/monsieur_lecoq.tar.xz"

 # Make a wworking directory
mkdir -p /tmp/tts
# copy MAILABS dataset
cp ${dataset_path} /tmp/tts
# decompress
tar -xjf /tmp/tts/monsieur_lecoq.tar.xz -C /tmp/tts
# compute dataset mean and variance for normalization
python ../../TTS/bin/compute_statistics.py --config_path model_config.json --out_path scale_stats.npy

# training ....
# change the GPU id if needed
# CUDA_VISIBLE_DEVICES="0" python ../../TTS/train_tacotron.py --config_path model_config.json
# train vocoder ...
# CUDA_VISIBLE_DEVICES="0" python TTS/vocoder/train.py --config_path vocoder_config.json
