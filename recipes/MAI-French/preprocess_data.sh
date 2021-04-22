 #!/bin/bash -l

if [ -f ~/.bashrc ]; then . ~/.bashrc; fi

 # PATHS
 home_path="/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/lpierron"
 corpus_path=${home_path}/mailabs/fr_FR

 output_path="${home_path}/Models/LJSpeech/"
 phoneme_cache_path="/tmp/tts/phoneme_cache_fr_ezwa"
 dataset_path="${corpus_path}/monsieur_lecoq.tar.xz"

 # Make a wworking directory
mkdir -p /tmp/tts
# copy MAILABS dataset
cp -u ${dataset_path} /tmp/tts
# decompress
tar -xJf /tmp/tts/monsieur_lecoq.tar.xz -C /tmp/tts  --skip-old-files
# compute dataset mean and variance for normalization
conda activate tf
python ../../TTS/bin/compute_statistics.py --config_path model_config.json --out_path scale_stats.npy
