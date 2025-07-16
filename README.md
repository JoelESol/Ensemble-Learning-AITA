# Ensemble-Learning-AITA


Dataset from here: https://github.com/iterative/aita_dataset

LLM model from here: https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF

Please test out our model on Colab!!!

https://colab.research.google.com/drive/1u9q68-GfQPNtlyJOaRAD1Ctn8j0DyOYS?usp=sharing

Folder Structure of Project:
```
├── analysis
      ├── (Report files)
├── dataset
      ├── agent_results.jsonl
      ├── aita_clean.csv
├── networks
      ├── openhermes-2.5-mistral-7b.Q4_K_M.gguf
├── transformer_outputs
      ├── *.png
      ├── verdict_predictor.pth
batched_llm_processing.py
ensemble_trainer.py
llm_results_visualization.py
```
