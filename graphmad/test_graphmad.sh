nKnots=10
mixup_batch_size=15
epochs=600
num_trials=7
t=1.08
weight_type="ideal"

for dataset in "AIDS" "MUTAG"
do
    for mixup_func in "linear" "sigmoid" "logit" "clusterpath"
    do
        echo ${dataset} $mixup_func
        CUDA_VISIBLE_DEVICES=0  python -u graphmad_test.py \
            --experiment_name "exptest_${dataset}_${mixup_func}" \
            --dataset "${dataset}" --data_path "data/" \
            --log_screen True --seed 1234 --num_trials ${num_trials} \
            --epoch ${epochs} \
            --nomixup "True" --gmixup "True" --graphmad "True" \
            --mixup_func "${mixup_func}" \
            --weight_type "${weight_type}" \
            --mixup_batch_size ${mixup_batch_size} --nKnots ${nKnots} --t ${t} \
            --sorted True --aligned True --sas_only True
    done
done
