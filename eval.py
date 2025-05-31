from tqdm import tqdm
import openai
import argparse

import torch

from utils import Data, Evaluator, make_batch
from coh import CoH

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CoH model with specified parameters.")
    parser.add_argument("--model_name", type=str, default="mixtral", help="Model name to use.")
    parser.add_argument("--model_path", type=str, default="/root/autodl-fs/models/TheBloke/Mixtral-8x7B-Instruct-v0___1-GPTQ/", help="Path to the model.")
    parser.add_argument("--dataset", type=str, default="ICEWS14_forecasting", help="Dataset name to use.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference.")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use.")
    parser.add_argument("--l_r", type=lambda x: (str(x).lower() == 'true'), default=True, help="Learning rate flag (True/False).")
    parser.add_argument("--i_s", type=lambda x: (str(x).lower() == 'true'), default=True, help="Input size flag (True/False).")
    args = parser.parse_args()
    
    model_name = args.model_name
    model_path = args.model_path
    dataset = args.dataset
    batch_size = args.batch_size
    num_gpus = args.num_gpus
    l_r = args.l_r
    i_s = args.i_s

    # model_name = "mixtral"
    # model_path = "/root/autodl-fs/models/TheBloke/Mixtral-8x7B-Instruct-v0___1-GPTQ/"
    # num_gpus = 1
    # dataset = "ICEWS14_forecasting"
    # batch_size = 32
    # l_r = True,
    # i_s = True,
    
    ###################### vLLM
    from vllm import LLM
    llm = LLM(
        model=model_path,
        quantization="gptq",
        dtype=torch.float16,
        tensor_parallel_size=num_gpus,
        enforce_eager=True
    )
    
    params_coh_vllm = {
        "max_tokens": 8000,
        "top_p": 1.0,
        "temperature": 0.0,
        "stop": ["</s>", "[/INST], '<|im_end|>', '<|endoftext|>"]
    }
    
    params_fliter_vllm = {
        "max_tokens": 16,
        "top_p": 1.0,
        "temperature": 0.0,
        "stop": ['</s>', '[/INST]', '<|im_end|>', '<|endoftext|>']
    }
    
    ##################### OpenAI
    # llm = openai.OpenAI(
    #     api_key="no_key",
    #     base_url="http://localhost:8000/v1"
    # )
    #
    # params_coh_openai = {
    #     "model": model_path,
    #    
    #     "max_tokens": 8000,
    #     "top_p": 1.0,
    #     "temperature": 0.3,
    #     "stop": ["</s>", "[/INST]", '<|im_end|>', '<|endoftext|>']
    # }
    #
    # params_fliter_openai = {
    #     "model": model_path,
    #    
    #     "max_tokens": 16,
    #     "top_p": 1.0,
    #     "temperature": 0.3,
    #     "stop": ['</s>', '[/INST]', '<|im_end|>', '<|endoftext|>']
    # }
    
    contents = Data(
        model_name=model_name,
        llm=llm,
        param=params_fliter_vllm,
        
        dataset=dataset, 
        add_reverse_relation=True
        )
    
    coh = CoH(
        num_entities=contents.num_entities,
        num_relations=contents.num_relations,
        entity2id=contents.entity2id,
        id2entity=contents.id2entity,
        id2relation=contents.id2relation,
        s_his_dict=contents.get_adj_his_format_dict(),
        llm=llm,
        # tokenizer=tokenizer,
        params=params_coh_vllm,
        expand_n=5,
        l_r=l_r,
        i_s=i_s
    )

    test_data = contents.test_data

    def extract_data(data):
        s, p, o, t = data
        return [s, p, t], o

    q, a = zip(*[extract_data(item) for item in test_data])
    
    q_batches = make_batch(q, batch_size)
    a_batches = make_batch(a, batch_size)
    
    title = f"coh test {dataset} "
    
    if not l_r:
        title += "| w/o LR "
    elif not i_s:
        title += "| w/o IS "
    
    eval1 = Evaluator(title + "| step1")
    eval2 = Evaluator(title + "| step2")

    total = len(q)
    cur = 0
    for q_batch, a_batch in zip(q_batches, a_batches):
        scores = coh(q_batch) 
        
        eval1.update(score_batch=scores[0], true_list=a_batch)
        eval2.update(score_batch=scores[1], true_list=a_batch)
        
        cur += len(q_batch)
        print(f"\nprocessed {cur}/{total}")
        
        eval1.print()
        eval2.print()
    
    print("*************** final result ******************")
    eval1.print()
    eval2.print()