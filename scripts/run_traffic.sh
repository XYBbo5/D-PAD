for seq_len in 336
do
    for pred_len in 96 192 336 720
    do
        echo traffic_${seq_len}-${pred_len}
        python run_long.py \
            --data traffic \
            --data_path traffic.csv \
            --features M \
            --seq_len ${seq_len} \
            --pred_len ${pred_len} \
            --enc_hidden 256 \
            --levels 2 \
            --lr 1e-4 \
            --dropout 0 \
            --batch_size 8 \
            --RIN 1 \
            --model_name DPAD_SE_traffic_I${seq_len}_o${pred_len}_lr1e-4_bs8_dp0_h256_l2 
            --model DPAD_SE
    done
done
