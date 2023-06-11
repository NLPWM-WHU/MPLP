# Mimicking the Thinking Process for Emotion Recognition in Conversation with Prompts and Paraphrasing
Code for our paper, "Mimicking the Thinking Process for Emotion Recognition in Conversation
with Prompts and Paraphrasing", IJCAI 2023, AI and Social Good track.


# Dependencies
- torch==1.13.1+cu117
- transformers==4.17.0
- sklearn
- fitlog

# Required data
Download required data for training of the second stage (including the base models we used, the raw texts of dialogues, the features of utterances, and the sample sets): 
- MELD:  [MELD_data](https://drive.google.com/file/d/136yjdkvxM_Ps0eGfNHBf-uuar-Ru5EoS/view?usp=sharing)
- IEMOCAP: [IEMOCAP_data](https://drive.google.com/file/d/1vMx9hxm696kCO6YvaTKEuobCPr7qVdEK/view?usp=sharing)
- DailyDialog: [DailyDialog_data](https://drive.google.com/file/d/1eBWK0KGIysP8QcZkIg3Y7dXC52wn7QRh/view?usp=sharing)
- SentiWordNet: [SentiWordNet_3.0.0](https://github.com/aesuli/SentiWordNet/blob/master/data/SentiWordNet_3.0.0.txt)

# Training
Train the second stage model with the following commands:
- MEDL: `bash train_prompt2.sh MELD 42 stage2 128 0.3 ./data/MELD/meld_stage1 3 bert_score`
- IEMOCAP: `bash train_prompt2.sh IEMOCAP 43 stage2 160 0.1 ./data/IEMOCAP/iemo_stage1 3 bert_score`
- DailyDialog: `bash train_prompt2.sh DailyDialog 2 stage2 128 0.2 ./data/DailyDialog/dd_stage1 5 bm25`

The optim values of these hyper-parameters(alpha and k) can vary under different seeds or environments (similar to [CoG-BART](https://github.com/whatissimondoing/CoG-BART)), and adjustments may be needed in different experimental settings.


# Quick Start
We also provide checkpoints for reproduction:
- MELD: [MELD_checkpoint](https://pan.baidu.com/s/1vtiAq93-ZTdWRI5EN4jMfA?pwd=n5nn) 提取码：n5nn  `bash eval_prompt2.sh MELD 128 0.3 ./data/MELD/meld_stage2 3 bert_score`
- IEMOCAP: [IEMOCAP_checkpoint](https://pan.baidu.com/s/1ybzgXdMG76SaEQsv8kSysQ?pwd=zfta) 提取码：zfta `bash eval_prompt2.sh IEMOCAP 160 0.1 ./data/IEMOCAP/iemo_stage2 3 bert_score`
- DailyDialog: [DailyDialog_checkpoint](https://pan.baidu.com/s/1Lbdv6chNn_KKqgiJfTpKPA?pwd=omu5) 提取码：omu5 `bash eval_prompt2.sh DailyDialog 128 0.2 ./data/DailyDialog/dd_stage2 5 bm25`


# Acknowledgment
Some code of this project are referenced from [CoG-BART](https://github.com/whatissimondoing/CoG-BART) and [DAG-ERC](https://github.com/shenwzh3/DAG-ERC). We thank their open source materials for contribution to this task.
