import os
import sys
from collections import defaultdict
import numpy as np
import torch
from datetime import datetime, timedelta

from coh.utils import llm_generate

PackageDir = os.path.dirname(__file__)
sys.path.insert(1, PackageDir)
DataDir = os.path.join(os.path.dirname(__file__), 'data')

FLITER_PROMPT = """
Do you know the fact that {query} ?
please give "Yes" or "No" directly.
"""

# Define the base date
BASE_DATE_DICT = {
    'ICEWS14_forecasting': '2014-01-01',
    'ICEWS18_forecasting': '2018-01-01',
    'ICEWS0515_forecasting': '2005-01-01'
    }

def try_gpu():
    return "cuda" if torch.cuda.is_available() else "cpu"

def make_batch(data, batch_size):
    batches = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batches.append(batch)
    return batches

class Data:
    def __init__(
        self, 
        
        model_name="mixtral",
        llm=None,
        param={},
        tokenizer=None,
        
        dataset=None, 
        add_reverse_relation=False,        
        ):
        """
        :param dataset:
        :param add_reverse_relation: if True, add reversed relation
        
        TODO: complete
        """
        self.model_name = model_name
        self.llm = llm
        self.tokenizer = tokenizer
        self.param = param
        
        # load data
        self.dataset = dataset
        self.id2entity, self.entity2id = self._id2entity(dataset=dataset)
        self.id2relation = self._id2relation(dataset=dataset)
        num_relations = len(self.id2relation)  # number of original relations, i.e., no reversed relation
        reversed_id2relation = {}
        if add_reverse_relation:
            for ind, rel in self.id2relation.items():
                reversed_id2relation[ind + num_relations] = 'Reversed ' + rel
            self.id2relation.update(reversed_id2relation)

            self.num_relations = 2 * num_relations
        else:
            self.num_relations = num_relations
        self.num_entities = len(self.id2entity)
        self.id2relation[self.num_relations] = 'selfloop'

        self.train_data = self._load_data(os.path.join(DataDir, dataset), "train")
        self.valid_data = self._load_data(os.path.join(DataDir, dataset), "valid")
        self.test_data = self._load_data(os.path.join(DataDir, dataset), "test")
        
        if os.path.exists(os.path.join(DataDir, dataset, "test_" + self.model_name + ".txt")):
            print("cache for fliter test data found.")
            self.fliter_test_data = self._load_data(os.path.join(DataDir, dataset), "test_" + self.model_name)
        else:
            print("no cache for fliter_test_data, do filter...")
            self.fliter_test_data = self._load_fliter_data(os.path.join(DataDir, dataset), self.test_data, "test")
            print("done.")

        # add reverse event into the data set
        if add_reverse_relation:
            self.train_data = np.concatenate([self.train_data[:, :-1],
                                              np.vstack([[event[2], event[1] + num_relations, event[0], event[3]]
                                                   for event in self.train_data])], axis=0)
        seen_entities = set(self.train_data[:, 0]).union(set(self.train_data[:, 2]))
        seen_relations = set(self.train_data[:, 1])

        val_mask = [evt[0] in seen_entities and evt[2] in seen_entities and evt[1] in seen_relations
                    for evt in self.valid_data]
        self.valid_data_seen_entity = self.valid_data[val_mask]

        if add_reverse_relation:
            self.valid_data = np.concatenate([self.valid_data[:, :-1],
                                              np.vstack([[event[2], event[1] + num_relations, event[0], event[3]]
                                                   for event in self.valid_data])], axis=0)
            self.valid_data_seen_entity = np.concatenate([self.valid_data_seen_entity[:, :-1],
                                                    np.vstack([[event[2], event[1] + num_relations, event[0], event[3]]
                                                               for event in self.valid_data_seen_entity])], axis=0)

        test_mask = [evt[0] in seen_entities and evt[2] in seen_entities and evt[1] in seen_relations
                     for evt in self.test_data]
        test_mask_conjugate = ~np.array(test_mask)

        print('seen dataset proportion: ' + str(np.asarray(test_mask).sum()/len(test_mask)))
        print('unseen dataset proportion: ' + str(test_mask_conjugate.sum()/test_mask_conjugate.size))

        self.test_data_seen_entity = self.test_data[test_mask]
        self.test_data_unseen_entity = self.test_data[test_mask_conjugate]

        if add_reverse_relation:
            self.test_data = np.concatenate([self.test_data[:, :-1],
                                             np.vstack(
                                                 [[event[2], event[1] + num_relations, event[0], event[3]]
                                                  for event in self.test_data])], axis=0)
            self.test_data_seen_entity = np.concatenate([self.test_data_seen_entity[:, :-1],
                                                         np.vstack(
                                                             [[event[2], event[1] + num_relations, event[0], event[3]]
                                                              for event in self.test_data_seen_entity])], axis=0)
            self.test_data_unseen_entity = np.concatenate([self.test_data_unseen_entity[:, :-1],
                                                         np.vstack(
                                                             [[event[2], event[1] + num_relations, event[0], event[3]]
                                                              for event in self.test_data_unseen_entity])], axis=0)

        self.data = np.concatenate([self.train_data, self.valid_data, self.test_data], axis=0)
        self.timestamps = self._get_timestamps(self.data)
        
    def _relative_to_absolute_date(self, relative_day):
        """
        Convert relative dates to absolute dates.

        param relative_day: Relative number of days (the number of days relative to the base date)
        param base_date: Reference date (in the format of 'YYYY-MM-DD')
        return: Absolute date (format: 'YYYY-MM-DD')
        """
        # Convert the base date string to a datetime object
        base_date = datetime.strptime(BASE_DATE_DICT[self.dataset], '%Y-%m-%d')
        # Calculate the absolute date
        absolute_date = base_date + timedelta(days=int(relative_day/24))
        # Convert absolute dates to string format
        return absolute_date.strftime('%Y-%m-%d')
        
    def _load_fliter_data(self, data_dir, data, data_type="test"):
        def get_prompt(his):
            s, p, o, t, _ = his
            fact = self.id2entity[s] + "\t" + self.id2relation[p] + "\t" + self.id2entity[o] + "\ton " + self._relative_to_absolute_date(t)
            prompt = FLITER_PROMPT.format(query=fact)
            return prompt
        
        prompts = [get_prompt(his) for his in data]
        outputs = llm_generate(self.llm, self.tokenizer, self.param, prompts)
        res = [item for item, output in zip(data, outputs) if output.strip()[0].lower() != 'y']
        
        with open(os.path.join(data_dir, "{}.txt".format(data_type + '_' + self.model_name)), 'w', encoding='utf-8') as f:
            for item in res:
                # Convert each element of the numpy array row to string and join with tab
                # This ensures the output format is "int\tint\tint\tint\n"
                f.write("\t".join(map(str, item.tolist())) + "\n")
                
        return np.array(res)

    def _load_data(self, data_dir, data_type="train"):
        with open(os.path.join(data_dir, "{}.txt".format(data_type)), 'r', encoding='utf-8') as f:
            data = f.readlines()
            data = np.array([line.split("\t") for line in data])  # only cut by "\t", not by white space.
            data = np.vstack([[int(_.strip()) for _ in line] for line in data])  # remove white space
        return data

    @staticmethod
    def _get_timestamps(data):
        timestamps = np.array(sorted(list(set([d[3] for d in data]))))
        return timestamps

    def neg_sampling_object(self, Q, dataset='train', start_time=0):
        '''
        :param Q: number of negative sampling for each real quadruple
        :param start_time: neg sampling for events since start_time (inclusive), used for warm start training
        :param dataset: indicate which data set to choose negative sampling from
        :return:
        List[List[int]]: [len(train_data), Q], list of Q negative sampling for each event in train_data
        '''
        neg_object = []
        spt_o = defaultdict(list)  # dict: (s, p, r)--> [o]
        if dataset == 'train':
            contents_dataset = self.train_data
            assert start_time < max(self.train_data[:, 3])
        elif dataset == 'valid':
            contents_dataset = self.valid_data_seen_entity
            assert start_time < max(self.valid_data_seen_entity[:, 3])
        elif dataset == 'test':
            contents_dataset = self.test_data_seen_entity
            assert start_time < max(self.test_data_seen_entity[:, 3])
        else:
            raise ValueError("invalid input for dataset, choose 'train', 'valid' or 'test'")

        data_after_start_time = [event for event in contents_dataset if event[3] >= start_time]
        for event in data_after_start_time:
            spt_o[(event[0], event[1], event[3])].append(event[2])
        for event in data_after_start_time:
            neg_object_one_node = []
            while True:
                candidate = np.random.choice(self.num_entities)
                if candidate not in spt_o[(event[0], event[1], event[3])]:
                    neg_object_one_node.append(
                        candidate)  # 0-th is a dummy node used to stuff the neighborhood when there is not enough nodes in the neighborhood
                if len(neg_object_one_node) == Q:
                    neg_object.append(neg_object_one_node)
                    break

        return np.stack(neg_object, axis=0)

    def _id2entity(self, dataset):
        with open(os.path.join(DataDir, dataset, "entity2id.txt"), 'r', encoding='utf-8') as f:
            _mapping = f.readlines()
            _mapping = [entity.strip().split("\t") for entity in _mapping]
            mapping = {int(ent2idx[1].strip()): ent2idx[0].strip() for ent2idx in _mapping}
            r_mapping = {ent2idx[0].strip().lower(): int(ent2idx[1].strip()) for ent2idx in _mapping} # lower name to id
        return mapping, r_mapping

    def _id2relation(self, dataset):
        with open(os.path.join(DataDir, dataset, "relation2id.txt"), 'r', encoding='utf-8') as f:
            mapping = f.readlines()
            mapping = [relation.strip().split("\t") for relation in mapping]
            id2relation = {}
            for rel2idx in mapping:
                id2relation[int(rel2idx[1].strip())] = rel2idx[0].strip()
        return id2relation

    def get_adj_list(self):
        '''
        adj_list for the whole dataset, including training data, validation data and test data
        :return:
        adj_list: List[List[(o(int), p(str), t(int))]], adj_list[i] is the list of (o,p,t) of events
        where entity i is the subject. Each row is sorted by timestamp of events, object and relation index
        '''
        adj_list_dict = defaultdict(list)
        for event in self.data:
            adj_list_dict[int(event[0])].append((int(event[2]), int(event[1]), int(event[3])))

        subject_index_sorted = sorted(adj_list_dict.keys())
        adj_list = [sorted(adj_list_dict[_], key=lambda x: (x[2], x[0], x[1])) for _ in subject_index_sorted]

        return adj_list

    def get_adj_dict(self):
        '''
        same as get_adj_list, but return dictionary, key is the index of subject
        :return:
        '''
        adj_dict = defaultdict(list)
        for event in self.data:
            adj_dict[int(event[0])].append((int(event[2]), int(event[1]), int(event[3])))

        for value in adj_dict.values():
            value.sort(key=lambda x: (x[2], x[0], x[1]))

        return adj_dict
    
    def get_adj_his_format_dict(self):
        '''
        Get a dictionary similar to get_adj_dict but with events in [s, p, o, t] format.
        This includes events from the whole dataset (train, valid, test).
        Each list of events is sorted by timestamp.
        :return:
        adj_his_format_dict: dict, key is the subject entity index, value is the list of [s, p, o, t] events sorted by timestamp
        '''
        adj_his_format_dict = defaultdict(list)
        
        # Add events from the combined dataset (self.data)
        for event in self.data:
            # Ensure event is a list of integers, similar to get_his_dict's output
            adj_his_format_dict[int(event[0])].append(event.tolist())
        
        # Sort events for each subject entity by timestamp
        for key in adj_his_format_dict:
            adj_his_format_dict[key].sort(key=lambda x: x[3])  # Sort by the 4th element (timestamp)
            
        return adj_his_format_dict
    
    def get_his_dict(self):
        '''
        Get a dictionary containing the history of events for each subject entity.
        The dictionary includes events from the training and validation datasets.
        Each list of events is sorted by timestamp.
        :return:
        his_dict: dict, key is the subject entity index, value is the list of [s, p, o, t] events sorted by timestamp
        '''
        his_dict = defaultdict(list)
        
        # Add and sort training data events
        for event in self.train_data:
            his_dict[int(event[0])].append(event.tolist())
        # Sort events for each subject entity by timestamp
        for key in his_dict:
            his_dict[key].sort(key=lambda x: x[3])  # Sort by the 4th element (timestamp)
        
        # Add and sort validation data events
        for event in self.valid_data:
            his_dict[int(event[0])].append(event.tolist())
        # Sort events for each subject entity by timestamp again
        for key in his_dict:
            his_dict[key].sort(key=lambda x: x[3])  # Sort by the 4th element (timestamp)
        
        return his_dict

    def get_spt2o(self, dataset: str):
        '''
        mapping between (s, p, t) -> list(o), i.e. values of dict are objects share the same subject, predicate and time.
        calculated for the convenience of evaluation "fil" on object prediction
        :param dataset: 'train', 'valid', 'test'
        :return:
        dict (s, p, t) -> o
        '''
        if dataset == 'train':
            events = self.train_data
        elif dataset == 'valid':
            events = self.valid_data
        elif dataset == 'test':
            events = self.test_data
        else:
            raise ValueError("invalid input {} for dataset, please input 'train', 'valid' or 'test'".format(dataset))
        spt2o = defaultdict(list)
        for event in events:
            spt2o[(event[0], event[1], event[3])].append(event[2])
        return spt2o

    def get_sp2o(self):
        '''
        get dict d which mapping between (s, p) -> list(o). More specifically, for each event in the **whole data set**,
        including training, validation and test data set, its object will be in d[(s,p)]
        it's calculated for the convenience of a looser evaluation "fil" on object prediction
        :param dataset: 'train', 'valid', 'test'
        :return:
        dict (s, p) -> o
        '''
        sp2o = defaultdict(list)
        for event in self.data:
            sp2o[(event[0], event[1])].append(event[2])
        return sp2o

class Evaluator:
    def __init__(self, title):
        """
        Initialize the evaluator.

        :param device: Device to run the evaluation on (e.g., 'cuda' or 'cpu').
        """
        self.title = title
        self.device = try_gpu()
        self.mrr = 0.0
        self.hits1 = 0.0
        self.hits3 = 0.0
        self.hits10 = 0.0
        self.total_queries = 0

    def update(self, score_batch, true_list):
        """
        Update the evaluation metrics with a batch of predictions.

        :param score_batch: Tensor of shape (batch_size, num_entities) containing prediction scores.
        :param true_list: List of indices of the true entities (length = batch_size).
        """
        score_batch = score_batch.to(self.device)
        true_list = torch.tensor(true_list).to(self.device)

        # Get the ranks of the true entities
        _, sorted_indices = torch.sort(score_batch, descending=True)
        ranks = torch.nonzero(sorted_indices == true_list.unsqueeze(1), as_tuple=False)[:, 1] + 1

        # Update evaluation metrics
        self.mrr += torch.sum(1.0 / ranks).item()
        self.hits1 += torch.sum(ranks == 1).item()
        self.hits3 += torch.sum(ranks <= 3).item()
        self.hits10 += torch.sum(ranks <= 10).item()
        self.total_queries += len(true_list)
    
    def print(self):
        """
        Compute and print the final evaluation metrics.
        """
        mrr = (self.mrr / self.total_queries) * 100  # Convert to percentage
        hits1 = (self.hits1 / self.total_queries) * 100  # Convert to percentage
        hits3 = (self.hits3 / self.total_queries) * 100  # Convert to percentage
        hits10 = (self.hits10 / self.total_queries) * 100  # Convert to percentage

        print(f"\n--- {self.title} ---")
        print(f"MRR (%): {mrr:.2f}")
        print(f"Hits@1 (%): {hits1:.2f}")
        print(f"Hits@3 (%): {hits3:.2f}")
        print(f"Hits@10 (%): {hits10:.2f}")
        print(f"-------------------------")

    def reset(self):
        """
        Reset the evaluator's internal state.
        """
        self.mrr = 0.0
        self.hits1 = 0.0
        self.hits3 = 0.0
        self.hits10 = 0.0
        self.total_queries = 0