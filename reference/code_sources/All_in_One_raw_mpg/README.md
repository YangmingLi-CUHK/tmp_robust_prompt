
### News

[13 JUl 2023] **I am delighted to share the news that we polished the code in the following link:**

https://github.com/sheldonresearch/ProG

We will maintain the above link for a long time.


Best Wishes!


***************************



### History Messages


This is the source code for our under reviewed paper 'All in One: Multi-task Prompting for Graph Neural Networks' in SIGKDD2023

Anyone reviewing this project is currently not allowed to download or use the code for any non-academic purpose. We plan to polish the code and release 
a formal python package once the paper is accepted. 


MPG: this is the core modules of our paper, including meta-learning components, kernel models of graph prompt, and customized graph nerual networks.

NoMeta: this is a demo folder presenting how to leverage the prompt graph for normal tasks.

- pre_trained_gnn: this is a folder containing pre-trained graph neural networks
- prompt.py: this is the prompt tuning framework for specific task with tunable head
- prompt_w_o_h.py: this is the prompt wihouth any tunable head

Meta: this is a demo folder presenting how to conduct prompt graph with meta-learning setting

