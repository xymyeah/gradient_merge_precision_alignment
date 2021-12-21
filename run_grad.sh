cp /paddle/workspace/Paddle/python/paddle/distributed/passes/gradient_merge.py /paddle/workspace/Paddle/build/python/paddle/distributed/passes/gradient_merge.py
cp /paddle/workspace/Paddle/python/paddle/distributed/auto_parallel/parallelizer.py /paddle/workspace/Paddle/build/python/paddle/distributed/auto_parallel/parallelizer.py
cp /paddle/workspace/Paddle/python/paddle/distributed/auto_parallel/auto_search.py /paddle/workspace/Paddle/build/python/paddle/distributed/auto_parallel/auto_search.py
cp /paddle/workspace/Paddle/python/paddle/distributed/auto_parallel/completion.py /paddle/workspace/Paddle/build/python/paddle/distributed/auto_parallel/completion.py
cp /paddle/workspace/Paddle/python/paddle/distributed/auto_parallel/dist_context.py /paddle/workspace/Paddle/build/python/paddle/distributed/auto_parallel/dist_context.py
cp /paddle/workspace/Paddle/python/paddle/distributed/auto_parallel/dist_context.py /paddle/workspace/Paddle/build/python/paddle/distributed/auto_parallel/dist_context.py
cp /paddle/workspace/Paddle/python/paddle/distributed/auto_parallel/utils.py /paddle/workspace/Paddle/build/python/paddle/distributed/auto_parallel/utils.py
cp /paddle/workspace/Paddle/python/paddle/distributed/fleet/base/fleet_base.py /paddle/workspace/Paddle/build/python/paddle/distributed/fleet/base/fleet_base.py

export PYTHONPATH=/paddle/workspace/Paddle/build/python:$PYTHONPATH

export GLOG_v=1
#export FLAGS_benchmark=1
export FLAGS_call_stack_level=2
export CUDA_VISIBLE_DEVICES=0
output_dir="./log_grad"
rm -rf ${output_dir}
mkdir ${output_dir}
python3 -m paddle.distributed.fleet.launch \
    --log_dir ${output_dir} \
    --gpus=${CUDA_VISIBLE_DEVICES} \
    test_auto_parallel_mlp_semi_grad.py \
    --output_dir ${output_dir} \
    --auto_search true \
    --max_steps 48 \
    --save_steps 10 \
    --device "gpu" \
    --checkpoint_path ${output_dir} \
    --global_batch_size 32  > ${output_dir}/lanch.log 2>&1