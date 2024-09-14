for i in $(seq 10 3 100);
do
    echo $i
    python points_run_model_run_local.py $i
done
