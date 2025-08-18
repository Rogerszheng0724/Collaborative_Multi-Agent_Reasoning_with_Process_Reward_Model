# Collaborative Multi-Agent Reasoning with Process Reward Model (CMAR)


## 📖 Overview
Large Language Models (LLMs) achieve high accuracy on reasoning tasks, but their inference remains opaque and error-prone.  
This project proposes **Collaborative Multi-Agent Reasoning (CMAR)**, which integrates three complementary reasoning chains:

- **Graph of Thoughts (GoT)**  
- **Layer of Thoughts (LoT)**  
- **Reversal of Thought (RoT)**  

All are orchestrated under a **Multi-Agent System (MAS)** and refined via an **Implicit Process Reward Model (PRM)**.  
The goal is to enhance **accuracy, interpretability, and robustness** in complex reasoning tasks.

---

## 🚀 Features
- **Multi-Agent Orchestration**: GoT, LoT, RoT collaborate via structured debate.  
- **Process Reward Model (PRM)**: Iterative refinement without extra annotations.  
- **Transparent ThoughtFlow**: Auditable intermediate reasoning steps.  
- **Improved Accuracy**: Achieves higher correctness, clarity, completeness, and relevance compared to baselines.  

---

## 🧩 Architecture
![architecture](https://github.com/user-attachments/assets/a59efc1f-9864-4eee-976d-5f62017c0ca2)


- **GoT**: Constructs a non-linear graph of reasoning nodes with PRM-based evaluation.  
- **LoT**: Decomposes tasks hierarchically with constraint-based prompting.  
- **RoT**: Uses reverse reasoning guided by preference and semantic validation.  
- **MAS Orchestrator**: Coordinates debate and synthesis among agents.  
- **PRM Loop**: Iteratively evaluates and refines reasoning outputs.  

---

## 📊 Datasets
- **Dolly** (brainstorming, creative writing)  
- **GSM8K** (mathematical reasoning)  
- **MMLU** (logical fallacies, jurisprudence, security studies)  

Total: **200 samples** compiled for experiments.

---

## 📝 Evaluation
### ThoughtFlow Evaluation
- **Cross-Entropy (CE) Loss**  
- **ReCEval**: Correctness, Informativeness, Reference-free evaluation of reasoning chains  

### Final Answer Evaluation
- **Correctness**  
- **Clarity**  
- **Completeness**  
- **Relevance**

![evaluation](https://github.com/user-attachments/assets/ac92773c-ec4d-4f5c-824a-b417f6ae35cd)

---

## 📈 Results
- **CMAR outperforms single CoT methods (GoT, LoT, RoT)** in correctness (+40% improvement).  
- **Clarity**: LoT contributes strongest clarity; CMAR balances coherence across tasks.  
- **Completeness**: CMAR achieves high coverage of reasoning steps.  
- **Relevance**: CMAR maintains strong topical alignment.
 
![evaluation](https://github.com/user-attachments/assets/d4143835-2e44-4134-b7f3-bda658115598)

---

## 🔍 Case Study
**Task:** *What words rhyme with orange?*  
- **Baseline (LoT):** Provided limited near rhymes.  
- **CMAR:** Generated diverse and phonetically aligned rhymes (e.g., *sporange, storage, forage, porridge*) through iterative debate and critique.  

---

## 📚 References
Key related works:
- Graph of Thoughts (Besta et al., AAAI 2024)  
- Layer of Thoughts (Fungwacharakorn et al., 2024)  
- Reversal of Thought (Yuan et al., 2024)  
- ReCEval: Evaluating Reasoning Chains (Prasad et al., EMNLP 2023)  
- PRIME: Process Reinforcement through Implicit Rewards (2025)  

---

## 📌 Future Work
- Scaling data coverage and applications.  
- Exploring new CoT configurations.  
- Extending reasoning generalizability in multi-agent settings.  

---

## 👩‍💻 Authors
- Shih-Wen Ke  
- Po-Hsiu Cheng  
- Po-Chun Chang  
- Bing-Ruei Zou  

Department of Information Management, National Central University, Taiwan  
