#!/bin/bash
set -xe

function main {
    source oob-common/common.sh
    # set common info
    init_params $@
    fetch_device_info
    set_environment

    # requirements
    pip uninstall -y pytorch3dunet && python setup.py develop
    pip install h5py scikit-image==0.20.0 tensorboard pyyaml
    cp oob-common/context_func.py pytorch3dunet/unet3d/.

    # if multiple use 'xxx,xxx,xxx'
    model_name_list=($(echo "${model_name}" |sed 's/,/ /g'))
    batch_size_list=($(echo "${batch_size}" |sed 's/,/ /g'))

    # generate benchmark
    for model_name in ${model_name_list[@]}
    do
        #
	if [ ${model_name} == "unet2d" ];then
	    train_config=resources/2DUnet_dsb2018/train_config.yml
	    ln -sf ${DATASET_DIR} 2dunet_datasets
	elif [ ${model_name} == "unet3d" ];then
	    train_config=resources/3DUnet_confocal_boundary/train_config.yml
	    ln -sf ${DATASET_DIR} 3dunet_datasets
	fi
        for batch_size in ${batch_size_list[@]}
        do
            # clean workspace
            logs_path_clean

            # generate launch script for multiple instance
            generate_core
            # launch
            echo -e "\n\n\n\n Running..."
            source ${excute_cmd_file}
            echo -e "Finished.\n\n\n\n"
            # collect launch result
            collect_perf_logs
        done
    done
}

# run
function generate_core {
    # generate multiple instance script
    for(( i=0; i<instance; i++ ))
    do
        real_cores_per_instance=$(echo ${device_array[i]} |awk -F, '{print NF}')
        log_file="${log_dir}/rcpi${real_cores_per_instance}-ins${i}.log"

        # instances
        if [ "${device}" == "cpu" ];then
            OOB_EXEC_HEADER=" numactl -m $(echo ${device_array[i]} |awk -F ';' '{print $2}') "
            OOB_EXEC_HEADER+=" -C $(echo ${device_array[i]} |awk -F ';' '{print $1}') "
        elif [ "${device}" == "cuda" ];then
            OOB_EXEC_HEADER=" CUDA_VISIBLE_DEVICES=${device_array[i]} "
        elif [ "${device}" == "xpu" ];then
            OOB_EXEC_HEADER=" ZE_AFFINITY_MASK=${i} "
	    fi
        printf " ${OOB_EXEC_HEADER} \
	    python pytorch3dunet/train.py --config ${train_config} \
	        --batch_size ${batch_size} --device ${device} \
		    --num_iter $num_iter --num_warmup $num_warmup \
		    --channels_last $channels_last --precision $precision \
                ${addtion_options} \
        > ${log_file} 2>&1 &  \n" |tee -a ${excute_cmd_file}
        if [ "${numa_nodes_use}" == "0" ];then
            break
        fi
    done
    echo -e "\n wait" >> ${excute_cmd_file}
}

# download common files
rm -rf oob-common && git clone https://github.com/intel-sandbox/oob-common.git -b gpu_oob

# Start
main "$@"
