from torch import nn
import torch
import logging
import numpy as np
import random

from .prompts import PICK_N_HIS, PICK_N_CHAINS, HIS_TO_ANSWER
from .utils import int_to_ordinal, parse_predict_answer, parse_ids_to_list, llm_generate

# the core class of Chain-of-Histories
class CoH(nn.Module):
    """
    TODO: complete
    """
    def __init__(
        self,
        
        num_entities,   # number of entities
        num_relations,  # number of relations
        
        id2entity,      
        entity2id,      # entity dict
        id2relation,    # relation dict
        s_his_dict,     # a dict from every entity to a list of its first-order histories 
        
        l_r=True,      # Logical Reasoning
        i_s=True,      # Index Sorting
        
        expand_n=5,     # chain expand num
        
        top_n=30,       # pick top n most relevant histories/chains in every step
        steps=2,        # total steps of CoH
        alpha=0.3,      # hyper parameter in the output score formula of CoH
        
        llm=None,       # LLM used, vllm.LLM or transformers.AutoModelForCausalLM
        tokenizer=None, # transformers.AutoTokenizer
        params={}       # sampling params of LLM (dict)
        ):
        """
        init CoH Module
        """
        super(CoH, self).__init__()
        
        self.steps = steps
        self.top_n = top_n
        self.expand_n = expand_n
        self.alpha = alpha
        
        self.llm = llm
        self.tokenizer = tokenizer
        self.params = params
        
        self.num_entities = num_entities
        self.num_relations = num_relations
        
        self.id2entity = id2entity
        self.entity2id = entity2id
        self.id2relation = id2relation
        
        self.l_r = l_r
        self.i_s = i_s
        
        self.s_his_dict = s_his_dict
        
        logging.info("testing LLM...")
        self.test_llm()
        logging.info("LLM is working!")
        
        self.init()
        
    def init(self):
        self.chain_list_batch = []       # the current order his chain list
        self.q_list = None               # the query list
        
    def format_his_list(self, his_list):
        """
        [[s, r, o, t], ...] -> ["Germany Sign agreement with Denmark on the 153rd day", ...]
        """
        res = []
        for his in his_list:
            s, r, o, t = his
            res.append(self.id2entity[s] + "\t" + self.id2relation[r] + "\t" + self.id2entity[o] + "\t" + int_to_ordinal(t))
        return res
    
    def generate(self, base_prompts: list[str]):
        return llm_generate(self.llm, self.tokenizer, self.params, base_prompts)

    def test_llm(self) -> bool:
        """
        Test if the LLM is working by generating a simple prompt
        
        Returns:
        True if the LLM is working, False otherwise
        """
        test_prompt = HIS_TO_ANSWER.format(
    histories="""
Government (Nigeria)\tEngage in diplomatic cooperation with\tIndependent Corrupt Practices Commission\ton the 339th day; Independent Corrupt Practices Commission\tArrest or detain or charge with legal action to\tCitizen (Nigeria)\ton the 308th day;
Government (Nigeria)\tCriticize or denounce\tBoko Haram\ton the 337th day; 
Boko Haram\tUse conventional military force to\tCitizen (Nigeria)\ton the 336th day;
Government (Nigeria)\tThreaten\tEducation (Nigeria)\ton the 337th day; 
Education (Nigeria)\tMake statement to\tMuslim (Nigeria)\ton the 332nd day;
Government (Nigeria)\tMake optimistic comment on\tCitizen (Nigeria)\ton the 336th day;
Citizen (Nigeria)\tMake an appeal or request to\tMember of the Judiciary (Nigeria)\ton the 331st day;
· · · · · ·
    """,
    query="""
Government (Nigeria)\tMake an appeal or request to\twhom\ton the 340th day?
    """
)
        try:
            output = self.generate([test_prompt])
            return isinstance(output, list) and len(output) > 0
        except Exception as e:
            logging.error(f"Error testing LLM: {e}")
            return False
    
    def get_n_his(self, his_list_batch):
        """
        Get top n histories from first-order histories for a batch of queries
        """
        self.chain_list_batch = []
        
        if self.l_r:
            prompts = []
            for q, his_list in zip(self.q_list, his_list_batch): 
                s, p, t = q
                hstr_list = self.format_his_list(his_list)
                his_context = ""
                for idx in range(len(hstr_list)):
                    his_context += str(idx) + ":[ " + hstr_list[idx] + "];\n"
                
                query = self.id2entity[s] + "\t" + self.id2relation[p] + "\twhom\t" + int_to_ordinal(t)
                prompt = PICK_N_HIS.format(
                    histories=his_context,
                    query=query,
                    top_n=self.top_n
                )
                prompts.append(prompt)
            
            outputs = self.generate(prompts)
            
            for output, his_list in zip(outputs, his_list_batch):
                selected_his_list = parse_ids_to_list(output, his_list)
                chain_list = [[item] for item in selected_his_list] 
                if len(chain_list) > self.top_n:
                    chain_list = chain_list[:self.top_n]
                self.chain_list_batch.append(chain_list)
            
        else:
            for output, his_list in zip(outputs, his_list_batch):
                chain_list = [[item] for item in his_list[-self.top_n:]] # use latest n histories
                self.chain_list_batch.append(chain_list)
                
                  
    def get_n_chains(self):
        """
        Get top n chains from last order chains for a batch of queries
        """
        results = []
        
        if self.l_r:
            prompts = []
            for q, chain_list in zip(self.q_list, self.chain_list_batch):  
                s, p, t = q
                chains_context = ""
                for idx in range(len(chain_list)):
                    chains_context += str(idx) + ": [" + ", ".join(self.format_his_list(chain_list[idx])) + "]\n"
                
                query = self.id2entity[s] + "\t" + self.id2relation[p] + "\twhom\t" + int_to_ordinal(t)
                prompt = PICK_N_CHAINS.format(
                    chains=chains_context,
                    query=query,
                    top_n=self.top_n
                )
                prompts.append(prompt)
            
            outputs = self.generate(prompts)
            
            for output, chain_list in zip(outputs, self.chain_list_batch):
                picked_list = parse_ids_to_list(output, chain_list)
                if len(picked_list) > self.top_n:
                    picked_list = picked_list[:self.top_n]
                results.append(picked_list)
        else:
            for output, chain_list in zip(outputs, self.chain_list_batch):
                picked_list = chain_list
                if len(picked_list) > self.top_n:
                    picked_list = picked_list[-self.top_n:] # use latest n histories
                results.append(picked_list)
        
        self.chain_list_batch = results
        
    def get_single_temporal_neighbor(self, src_entity_id, query_time, expand_n):
        """
        Retrieves the expand_n most recent histories for a single source entity
        that occurred before the given query time.

        Parameters:
        src_entity_id: The ID of the source entity.
        query_time: The query timestamp.

        Returns:
        A list of histories, where each history is [s, r, o, t].
        """
        all_histories_for_s = self.s_his_dict.get(src_entity_id, [])
        
        # Filter histories that occurred before the query_time
        # and sort them by timestamp in descending order (most recent first)
        # Note: 'his[3]' is assumed to be the timestamp in the history tuple [s, r, o, t].
        # The original code used his[2] which is the object, correcting to his[3].
        filtered_and_sorted_histories = sorted(
            [his for his in all_histories_for_s if his[3] < query_time],
            key=lambda x: x[3],
            reverse=True
        )
        
        # Select the top expand_n most recent histories
        selected_neighbors = []
        if len(filtered_and_sorted_histories) > expand_n:
            selected_neighbors = filtered_and_sorted_histories[:expand_n]
        else:
            selected_neighbors = filtered_and_sorted_histories
        
        selected_neighbors.reverse()    # Reverse to have them in chronological order
        
        # Histories in s_his_dict are already [s, r, o, t], so no need to remap item[0], item[1] etc.
        return selected_neighbors
           
    def expand_chains(self):
        """
        Expands chains to the next order for a batch of queries.
        For each chain, it finds relevant next-hop histories from the last entity
        in the chain, ensuring they occurred *before* the query time.
        The strict chronological order *after* the last event in the current chain is not enforced.
        """
        new_chain_list_batch = []

        for q, chains_for_q in zip(self.q_list, self.chain_list_batch):
            query_s, query_p, query_t = q
            expanded_chains_for_q = []

            for chain in chains_for_q:
                if not chain:
                    continue

                last_his = chain[-1]
                last_entity_in_chain = last_his[2]  # The object (o) of the last history

                # Use the single-entity get_single_temporal_neighbor function
                # to get histories related to the last entity in the chain,
                # occurring before the query time.
                selected_next_histories = self.get_single_temporal_neighbor(
                    last_entity_in_chain,  # Source entity for expansion
                    query_t,               # Target time for filtering
                    self.expand_n
                )

                # Create new expanded chains
                for next_his in selected_next_histories:
                    expanded_chain = chain + [next_his]
                    expanded_chains_for_q.append(expanded_chain)
            
            new_chain_list_batch.append(expanded_chains_for_q)
            
        self.chain_list_batch = new_chain_list_batch
    
    def get_predict_score(self):
        """
        get predict from the last order chains for a batch of queries and
        graph from the last order chains
        """
        prompts = []
        for q, chain_list in zip(self.q_list, self.chain_list_batch):
            s, p, t = q

            # Flatten the chain list and sort by time
            flattened_chains = [item for sublist in chain_list for item in sublist]

            # Convert sorted chains to text format
            chain_text = "\n".join([self.id2entity[s] + "\t" + self.id2relation[r] + "\t" + self.id2entity[o] + "\t" + int_to_ordinal(t) for s, r, o, t in flattened_chains])

            # Create the prompt for prediction
            query = self.id2entity[s] + "\t" + self.id2relation[p] + "\twhom\t" + int_to_ordinal(t)
            prompt = HIS_TO_ANSWER.format(
                histories=chain_text,
                query=query
            )
            prompts.append(prompt)
            
        def parse_output(output):
            candidates = parse_predict_answer(output, self.entity2id) # not include all the entities, only the entitities LLM predict,
                                                      # a list of entities'name, need to parse using self.entity2id
                      
            if not self.i_s:
                random.shuffle(candidates)            # shuffle the indexes                     
            
            res = np.zeros(self.num_entities, dtype=int)
            for i in range(len(candidates)):
                res[self.entity2id[candidates[i]]] = i + 1
            return res
            
        outputs = self.generate(prompts)
        
        # TODO: peek the predict text outputs
        # for text in outputs:
        #     print("--------------------------------------------------------------------------")
        #     print(text)
        
        predicts = []
        for output in outputs:
            predicts.append(parse_output(output))
        
        # predicts: (batch_size, num_entities)  
        predicts = torch.tensor(np.array(predicts), dtype=torch.float32)  # Convert to tensor  
        
        # Calculate scores using PyTorch tensor operations
        alpha = torch.tensor(self.alpha, dtype=torch.float32)
        exp_part = torch.exp(alpha * predicts)
        scores = 1 / (1 + exp_part)
            
        scores[predicts == 0] = 0 # 0 means the entity not be mentioned by LLM!    
            
        return scores

    def forward(self, q_list):
        """
        input: (batch_size, 3)
        output: (steps, (batch_size, num_entities))
        """
        self.init() # clean
        
        step_scores_list = []
        
        self.q_list = q_list
        
        # Initialize his_list_batch by getting the initial (first-order) histories
        # for each query's subject, occurring before the query time.
        his_list_batch = []
        for q in q_list:
            subject_id = q[0]
            query_time = q[2]
            # Retrieve the most recent 'expand_n' histories for the subject
            # that happened *before* the query_time.
            initial_histories = self.get_single_temporal_neighbor(subject_id, query_time, 100)
            his_list_batch.append(initial_histories)
        
        self.get_n_his(his_list_batch)
        step_scores_list.append(self.get_predict_score())
        
        for i in range(self.steps - 1):     
            self.expand_chains()        
            self.get_n_chains()
            step_scores_list.append(self.get_predict_score())
            
        return step_scores_list    
                
                
               
