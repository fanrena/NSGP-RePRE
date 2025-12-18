CFGS=(
    # Your configs here. Note that you have to setup checkpoint loading in your configs.
)


for CFG in "${CFGS[@]}"
do 
    echo $CFG
    CUDA_VISIBLE_DEVICES=${your_devices} bash tools/dist_train.sh $CFG 2
done
