#!/bin/bash

# if [ "$1" = "train" ]; then
# 	CUDA_VISIBLE_DEVICES=0 python run.py train --train-src=./en_es_data/train.es --train-tgt=./en_es_data/train.en --dev-src=./en_es_data/dev.es --dev-tgt=./en_es_data/dev.en --vocab=vocab.json --cuda
# elif [ "$1" = "test" ]; then
#     CUDA_VISIBLE_DEVICES=0 python run.py decode model.bin ./en_es_data/test.es ./en_es_data/test.en outputs/test_outputs.txt --cuda
# elif [ "$1" = "train_local" ]; then
# 	python run.py train --train-src=./en_es_data/train.es --train-tgt=./en_es_data/train.en --dev-src=./en_es_data/dev.es --dev-tgt=./en_es_data/dev.en --vocab=vocab.json
# elif [ "$1" = "test_local" ]; then
#     python run.py decode model.bin ./en_es_data/test.es ./en_es_data/test.en outputs/test_outputs.txt
# elif [ "$1" = "vocab" ]; then
# 	python vocab.py --train-src=./en_es_data/train.es --train-tgt=./en_es_data/train.en vocab.json
# else
# 	echo "Invalid Option Selected"
# fi

if [ "$1" = "train" ]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py train --train-src=./mathqa/q.txt --train-tgt=./mathqa/a.txt --dev-src=./mathqa/dq.txt --dev-tgt=./mathqa/da.txt --vocab=mathqa-vocab.json --cuda
elif [ "$1" = "test" ]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py decode mathqa-trained/model8layer.bin ./mathqa/tq.txt ./mathqa/ta.txt outputs/mathqa-test_outputs8layer.txt --cuda
elif [ "$1" = "train_local" ]; then
	python run.py train --train-src=./mathqa/q.txt --train-tgt=./mathqa/a.txt --dev-src=./mathqa/dq.txt --dev-tgt=./mathqa/da.txt --vocab=mathqa-vocab.json
elif [ "$1" = "test_local" ]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py decode mathqa-trained/model8layer.bin ./mathqa/new_q.txt ./mathqa/new_a.txt outputs/mathqa-test_outputs_try.txt --cuda
elif [ "$1" = "vocab" ]; then
	python vocab.py --train-src=./mathqa/q.txt --train-tgt=./mathqa/a.txt mathqa-vocab.json
else
	echo "Invalid Option Selected"
fi
