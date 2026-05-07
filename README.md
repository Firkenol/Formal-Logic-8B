# Formal-Logic-8B: Bridging Informal Vernacular and Logical Structures

# Formal-Logic-8B

## Overview
Formal-Logic-8B is a domain-specific adaptation of the `Meta-Llama-3.1-8B-Instruct` architecture. Utilizing Continuous Pre-Training (CPT) and instruction fine-tuning, this project enhances the model's capacity to extract and formalize strict programmatic logic from chaotic, real-world, informal datasets.

## Origins & Motivation
This architecture was developed to resolve data parsing bottlenecks encountered in the LocoGo routing optimizer and the EduDrama NLP pipeline. The primary challenge in autonomous logistics and unstructured text analysis is the translation gap between informal human vernacular and deterministic programmatic execution. Formal-Logic-8B bridges this translation gap.

## Technical Stack
- **Base Architecture:** Meta-Llama-3.1-8B-Instruct
- **Training Infrastructure:** Azure NC96ads_A100_v4 (4x NVIDIA A100 80GB)
- **Optimization Engine:** DeepSpeed ZeRO-3
- **Secret Management:** Azure Key Vault for Hugging Face integration
