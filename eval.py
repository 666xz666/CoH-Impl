from tqdm import tqdm
import argparse
from vllm import LLM
import torch

from utils import Data, Evaluator, make_batch
from coh import CoH

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Run CoH model with specified parameters.")
    # parser.add_argument("--model_name", type=str, default="mixtral", help="Model name to use.")
    # parser.add_argument("--dataset", type=str, default="ICEWS14_forecasting", help="Dataset name to use.")
    # parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference.")
    # args = parser.parse_args()
    
    # model_name = args.model_name
    # dataset = args.dataset
    # batch_size = args.batch_size

    model_name = "mixtral"
    model_path = "/root/autodl-fs/models/TheBloke/Mixtral-8x7B-Instruct-v0___1-GPTQ/"
    num_gpus = 1
    dataset = "ICEWS14_forecasting"
    
    llm = LLM(
        model=model_path,
        quantization="gptq",
        dtype=torch.float16,
        tensor_parallel_size=num_gpus,
        enforce_eager=True
    )
    
    params_fliter = {
        "max_tokens": 16,
        "top_p": 1.0,
        "temperature": 0.0,
        "stop": ['</s>', '[/INST]', '<|im_end|>', '<|endoftext|>']
    }

    contents = Data(
        model_name=model_name,
        llm=llm,
        param=params_fliter,
        
        dataset=dataset, 
        add_reverse_relation=True
        )
    
    params_coh = {
        "max_tokens": 1024,
        "top_p": 1.0,
        "temperature": 0.0,
        "stop": ["</s>", "[/INST]"]
    }

    coh = CoH(
        num_entities=contents.num_entities,
        num_relations=contents.num_relations,
        entity2id=contents.entity2id,
        id2entity=contents.id2entity,
        id2relation=contents.id2relation,
        s_his_dict=contents.get_adj_his_format_dict(),
        llm=llm,
        # tokenizer=tokenizer,
        params=params_coh,
        expand_n=5
    )

    test_data = contents.fliter_test_data
    test_data = test_data[:10]

    def extract_data(data):
        s, p, o, t, _ = data
        return [s, p, t], o

    q, a = zip(*[extract_data(item) for item in test_data])
    
    # q_batches = make_batch(q, batch_size)
    # a_batches = make_batch(a, batch_size)
    
    eval = Evaluator("coh test {}".format(dataset))

    # for q_batch, a_batch in tqdm(zip(q_batches, a_batches)):
    #     scores = coh(q_batch)  
    #     eval.update(score_batch=scores, true_list=a_batch)
    #     eval.print()  

    scores = coh(q)
    eval.update(score_batch=scores, true_list=a)
    
    print("*************** final result ******************")
    eval.print()
