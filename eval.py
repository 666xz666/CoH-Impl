from tqdm import tqdm
import argparse

from utils import vllm_builder, Data, Evaluator, make_batch
from coh import CoH

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CoH model with specified parameters.")
    parser.add_argument("--model_name", type=str, default="mixtral", help="Model name to use.")
    parser.add_argument("--dataset", type=str, default="ICEWS14_forecasting", help="Dataset name to use.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference.")
    args = parser.parse_args()
    
    model_name = args.model_name
    dataset = args.dataset
    batch_size = args.batch_size

    llm, task_params = vllm_builder(model_name)

    contents = Data(
        model_name=model_name,
        llm=llm,
        param=task_params['fliter'],
        
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
        params=task_params,
        expand_n=5
    )

    test_data = contents.test_data
    # test_data = test_data[:3]

    def extract_data(data):
        s, p, o, t = data
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
    
    eval.print()
