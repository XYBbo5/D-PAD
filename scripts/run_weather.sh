for seq_len in 336
do
    for dp in 0 0.25 0.5
    do
        for pred_len in 96 192 336 720
        do
            echo weather_${seq_len}-${pred_len}
            python run_long.py \
                --data weather \
                --data_path weather.csv \
                --features M \
                --seq_len ${seq_len} \
                --pred_len ${pred_len} \
                --enc_hidden 336 \
                --levels 2 \
                --lr 1e-4 \
                --dropout ${dp} \
                --batch_size 64 \
                --RIN 1 \
                --model_name DPAD_SE_weather_I${seq_len}_o${pred_len}_lr1e-4_bs64_dp${dp}_h336_l2 
                --model DPAD_SE
        done
    done
done