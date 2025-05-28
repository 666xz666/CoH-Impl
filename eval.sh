export CUDA_VISIBLE_DEVICES=0,1,2,3
python eval.py --model_name mixtral\
               --dataset ICEWS14_forecasting
            #    --batch_size 32