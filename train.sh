# Training Code 
python -u main.py --gpu 0 --batch 40  --learning_rate 1e-4 --epochs 10  -l "cross-entropy"  -tp "context_json" -c
python -u main.py --gpu 0 --batch 40  --learning_rate 1e-4 --epochs 10  -l "cross-entropy"  -tp "context_json" -c -tw