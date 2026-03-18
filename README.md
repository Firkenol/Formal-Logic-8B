# Formal-Logic-8B: Bridging Informal Vernacular and Logical Structures

### Overview
Formal-Logic-8B is a domain-specific adaptation of the Meta-Llama-3.1-8B-Instruct model. This project uses Continuous Pre-Training (CPT) to improve the model's ability to formalize informal logic found in chaotic, real-world datasets.

### Origins
The architecture evolved from the underlying data parsing requirements of the LocoGo routing optimizer and the EduDrama NLP pipeline. The primary bottleneck in autonomous logistics and unstructured text analysis is the translation gap between informal human vernacular and strict programmatic logic. This model bridges that gap.

### Technical Stack
* **Base Model:** Meta-Llama-3.1-8B-Instruct
* **Training Infrastructure:** Azure NC96ads_A100_v4 (4x NVIDIA A100 80GB)
* **Optimization:** DeepSpeed ZeRO-3
* **Secret Management:** Azure Key Vault for Hugging Face API integration