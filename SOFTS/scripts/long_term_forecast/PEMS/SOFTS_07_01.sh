model_name=SOFTS

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS07.npz \
  --model_id PEMS07_96_12 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 12 \
  --e_layers 2 \
  --enc_in 883 \
  --dec_in 883 \
  --c_out 883 \
  --des 'Exp' \
  --d_model 512 \
  --d_core 512 \
  --d_ff 512 \
  --learning_rate 0.0003 \
  --lradj cosine \
  --train_epochs 30 \
  --patience 3 \
  --use_norm 0 \
  --itr 1
