# CoH-Impl

```
@article{xiang2024enhancing,
  title={Enhancing Temporal Knowledge Graph Forecasting with Large Language Models via Chain-of-History Reasoning},
  author={Xia, Yuwei and Wang, Ding and Liu, Qiang and Wang, Liang and Wu, Shu and Zhang, Xiaoyu},
  journal={arXiv preprint arXiv:2402.14382},
  year={2024},
  institution={Institute of Information Engineering, Chinese Academy of Sciences; School of Cyber Security, University of Chinese Academy of Sciences; School of Artificial Intelligence, University of Chinese Academy of Sciences; Institute of Automation, Chinese Academy of Sciences},
  url={https://arxiv.org/abs/2402.14382}
}

```

```python
python eval.py \
--model_name "mixtral" \
--model_path "/root/autodl-fs/models/TheBloke/Mixtral-8x7B-Instruct-v0___1-GPTQ/" \
--dataset "ICEWS14_forecasting" \
--batch_size 32 \
--num_gpus 1 \
--l_r True \
--i_s True