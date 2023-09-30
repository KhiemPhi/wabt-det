# Testing Code
python -u main.py --gpu 0 --batch 40  --learning_rate 1e-4 --epochs 10  -l "softmax" -w "weights/mina_youtube_best.ckpt" -tp "context_json" -t -c
python -u main.py --gpu 0 --batch 40  --learning_rate 1e-4 --epochs 10  -l "softmax" -w "weights/mina_twitter_best.ckpt" -tp "context_json" -tw -t -c

