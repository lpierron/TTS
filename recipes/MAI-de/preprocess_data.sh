 #!/bin/bash -l

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

 # PATHS
 home_path="/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/lpierron"
 corpus_path=${home_path}/mailabs/de_DE

 output_path="${home_path}/Models/LJSpeech/"
 phoneme_cache_path="/tmp/tts/phoneme_cache_de_eva_k"
 dataset_path="${corpus_path}/eva_k.tar.xz"

 # Make a wworking directory
mkdir -p /tmp/tts
# copy MAILABS dataset
cp -u ${dataset_path} /tmp/tts
# decompress
tar -xJf /tmp/tts/eva_k.tar.xz -C /tmp/tts  --skip-old-files
# compute dataset mean and variance for normalization
conda activate tts
python ../../TTS/bin/compute_statistics.py --config_path model_config.json --out_path scale_stats.npy
