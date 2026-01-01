# Legal Reasoning: Geometry-Aware Graph Injection with BoxE

**Legal Reasoning Project | NCCU**

This project focuses on geometrizing legal logic by transforming regulatory texts into high-dimensional hyper-rectangles. By employing a hybrid RGAT-BoxE architecture, the system solves range ambiguity and achieves expert-level reasoning in the Occupational Safety and Health (OSH) domain.
* Claim: This project is now still working with many future works. If there are any modification or updates,  we will written in this README file.
* *Note: This project entails large use with OPENAI API.*

---

## 1. Load Laws and Regulations

### Python File: `loader.py`
*   **Idea:** Convert raw data which consists of "symbol-formulated" "tables" and "formulas" into pure text and wrap all law documents in pdf into a single json file.
*   **Set Environment Variables: [Ubuntu]**
    ```bash
    export OPENAI_API_KEY=your_key_here
    export INPUT_FOLDER=./path/to/pdfs
    export OUTPUT_FOLDER=./output
    ```
    *   *For example:*
        *   `INPUT_FOLDER=./load_law;`
        *   `OUTPUT_FOLDER=./loaded_law`
*   **Install Dependencies:**
    *   `pip install -r requirements.txt`
    *   **Requirements:**
        *   `PyMuPDF>=1.23.0`
        *   `pytesseract>=0.3.10`
        *   `Pillow>=9.0.0`
        *   `opencv-python>=4.5.0`
        *   `openai>=1.0.0`
        *   `numpy>=1.21.0`
        *   `tiktoken>=0.5.0`
    *   *Time to load dependencies: 5 to 10 minutes*
*   **Inputting Files:** in `load_law` folder (or `load_law_tiny` for testing)
*   **Outputting Files:** `all_documents` (contains all laws) and respective extracted laws
*   **Execution Time:**
    *   About 17 minutes for `load_law_all`
    *   About 5 minutes for `load_law_tiny`

### Python File: `formatting.py`
*   **Idea:** Make laws more possible to be matched for subsequent matching law name use.
*   **Set Environment Variables: [Ubuntu]**
    ```bash
    python formatting.py -i "D:\path\to\input\all_documents.json" -o "D:\path\to\output\_legal_content.json" -f json
    ```
*   **Install Dependencies:**
    *   `pip install pandas`
*   **Inputting Files:** `all_documents.json`
*   **Outputting Files:** `legal_content`
*   **Execution Time:** fast

---

## 2. Load Occupational Safety and Health Documents

### Python File: `osh_doc_structure.py`
*   **Idea:** [Extract structure from OSH documents]
*   **Install Dependencies:**
    *   `pip install --upgrade pip`
    *   `pip install openai PyMuPDF python-dotenv`
    *   *Time to load dependencies: 1 minute*
*   **Set Environment Variables: [Ubuntu]**
    ```bash
    cat > .env <<'EOF'
    OPENAI_API_KEY="sk-REPLACE_WITH_YOUR_KEY"
    OSH_MODE="folder"
    OSH_FOLDER_PATH="./osh_case_folder_1"
    OSH_OUTPUT_FOLDER="./extraction_output"
    OSH_SINGLE_PDF_PATH="./sample.pdf"
    EOF
    
    mkdir -p osh_case_folder_1 extraction_output
    
    python3 -m dotenv run -- python osh_doc_structure.py --mode folder
    
    python3 -m dotenv run -- python osh_doc_structure.py --mode single --single-pdf-path "./input_pdfs/myfile.pdf"
    ```
    *   *For example:*
        *   `INPUT_FOLDER=./load_law;`
        *   `OUTPUT_FOLDER=./loaded_law`
*   **Inputting Files:** Documents of collection of incidents in occupational safety and health
    *   Source 1: https://www.osha.gov.tw/48110/48461/48517/48553/ (103~107)
    *   Source 2: https://www.doli.taichung.gov.tw/1516279/ (104~114)
*   **Outputting Files:** `osh_doc_merged.json` (which contains collections of incidents and respective extracted incidents)
*   **Execution Time:**
    *   `osh_case_folder_1`: about 20 minutes
    *   `osh_case_folder_2`: about 14 minutes
    *   `osh_case_folder_3`: about 12 minutes

### Python File: `merge_jsons.py`
*   **Idea:** Merge those respective json files in the "extraction_output" folder.
*   **Inputting Files:** The folder that collects all structured osh json documents (for example, `extraction_output`).
*   **Outputting Files:** `osh_doc_merged.json`

---

## 3. Build Knowledge Graph

### Python File: `graph_builder.ipynb`
*   **Idea:** Build a knowledge graph for subsequent training work based on the structured osh document.
*   **Set Environment: [Linux]**
    *   Manually upload `osh_doc_merged.json` at the same directory as the ipynb file.
    *   `export OSH_INPUT="/home/you/data/incidents.json"`
    *   `export OSH_OUT_DIR="/home/you/osh_output"`
    *   `export OSH_STRICT="0"`  *(# "0" => strict False, otherwise strict True)*
*   **Set Environment: [Linux] (Colab)**
    *   Manually upload `osh_doc_merged.json` at the same directory as the ipynb file, for example, at `/content`.
*   **Install Dependencies:**
    *   `pip install -r requirements.txt`
    *   **Requirements:**
        *   `networkx>=2.8`
        *   `pandas>=1.3`
        *   `openpyxl>=3.0`
    *   *Time to load dependencies: 2:01 minutes*
*   **Inputting File:** `osh_doc_merged.json`
*   **Main Outputting File:** `knowledge_graph.json`
*   **Execution Time:** fast.

---

## 4. Refine Knowledge Graph

### Python File: `kg_eval.ipynb`
*   **Idea:** Refine a knowledge graph by merging semantically equivalent or highly similar nodes and creating similarity edges between related nodes, thereby optimizing the graph structure and addressing sparsity issues.
*   **Set Environment: [Linux]**
    *   Manually upload `knowledge_graph.json` at the same directory as the ipynb file.
    *   Need to install dependencies.
*   **Set Environment: [Linux] (Colab)**
    *   Manually upload `knowledge_graph.json` at the same directory as the ipynb file, for example, at `/content`.
    *   Set hugging face key.
*   **Inputting File:** `knowledge_graph.json`
*   **Outputting File:** `knowledge_graph_connected.json`
    *   *Important Note:* The `knowledge_graph_connected.json` we upload in the folder `4_graph_refinement` is **not** the original file produced by this `kg_eval.ipynb`. Instead, this `knowledge_graph_connected.json` entails manual work, so we will process this knowledge graph for subsequent uses.
*   **Execution Time:** fast.

---

## 5. Map Law Content to Knowledge Graph

### Python File: `graph_mapping.py`
*   **Idea:** This section maps law content to knowledge graph based on the "regulation node" with law name labelling.
*   **Inputting File:** `knowledge_graph_connected.json`
*   **Outputting File:** `knowledge_graph_final.json`
*   **Execution Time:** fast.

---

## 6. Graph Neural Network and Knowledge

### Python File: `GNN_and_KGE.ipynb`
*   **Idea:** This project effectively geometrizes legal logic by transforming regulatory texts into high-dimensional hyper-rectangles, employing a hybrid RGAT-BoxE architecture to solve range ambiguity and achieve expert-level legal reasoning.
*   **Inputting File:** `knowledge_graph_connected.json`
*   **Outputting File:** `knowledge_graph_final.json`
*   **Execution Time:** fast.

### Overview of the Code Implementation:

#### 1. Core Goal & Pain Points Addressed
The primary objective is to build an AI reasoning engine capable of **"deduction"** like a legal expert.
*   **Range Ambiguity:** Regulations often contain numerical intervals.
    *   *Solution:* **BoxE (Box Embeddings)** models relations as "boxes" in high-dimensional space.
*   **Semantic Cold Start:** Legal entities have rich textual definitions.
    *   *Solution:* Integration of **LLMs (text2vec)** for semantic feature initialization.
*   **Cross-reference Complexity:** Intricate web of citations.
    *   *Solution:* **RGAT (Relational Graph Attention Network)** automatically learns critical reference relationships.

#### 2. Model Architecture (Encoder-Decoder Paradigm)
*   **Phase 1-3: Data Preprocessing**
    *   *Graph Normalization:* JSON to PyTorch Geometric (PyG).
    *   *Semantic Initialization:* `shibing624/text2vec-base-chinese` model (768-dim vectors).
    *   *Topology Augmentation:* Adding "inverse edges" and "self-loops".
*   **Phase 4: Encoder - RGAT**
    *   *Role:* "Comprehend" the legal structure.
    *   *Mechanism:* Aggregates info based on "relation types" (e.g., "leads_to", "violates") via Attention.
*   **Phase 5: Decoder - BoxE**
    *   *Role:* Perform "geometric reasoning."
    *   *Innovation:* Relations as hyper-rectangles (Head Box and Tail Box). Judgment involves calculating if an entity vector falls inside these "boxes."

#### 3. Training Strategy
*   **Self-Adversarial Negative Sampling:** Focuses on distinguishing "hard negatives."
*   **Geometric Constraints:** Enforcing box widths to be $>0$ during training.

#### 4. Validation & Results
*   Benchmarks against **RotatE** model.
*   **BoxE Performance:** Hit@1 approx. **96%** on validation set.

---

## 7. Create Training Dataset for Subsequent Supervised Fine-Tuning (SFT) Work

### Python File: `Train_Data_Preparation.ipynb`
*   **Idea:** A comprehensive pipeline for "Occupational Safety and Health (OSH) Knowledge Graph (KG) Enhancement and LLM Training Data Generation." Combines Neural Networks (BERT) with Symbolic Logic (Knowledge Graph) to produce high-quality, "explainable reasoning" training data.
*   **Inputting Files:** `knowledge_graph_final.json`; `osh_doc_merged.json`; `boxe_validation_set_clean.json`.
*   **Outputting Files:** `osh_legal_ground_truth_cleaned`; `sft_training_data_final`.
*   **Execution Time:** fast (without content creation); extremely time-consuming if content creation is entailed.

### Overview of the Code Implementation:

**Phase 1: Neuro-Symbolic Preprocessing of the Knowledge Graph**
*   **Semantic Embedding (Action B):** BERT model converts nodes to vectors.
*   **Semantic Pruning (Action C):** Calculates Cosine Similarity. Retains high similarity edges as `VIOLATES_SPECIFICALLY`; lower as `IS_RELEVANT_TO`.
*   **Hierarchy Injection (Action D):** Builds "Regulation -> Law" structure.
*   **Hard Negative Mining:** Identifies textually similar but logically incorrect regulations.

**Phase 2: Ground Truth Construction**
*   **Regulation Extraction:** Regex to extract regulation names, articles, paragraphs.
*   **Data Cleaning:** Filters out invalid data.

**Phase 3: Entity Alignment**
*   **Text Matching:** Uses TF-IDF and Cosine Similarity to pair accident descriptions with Incident Nodes.

**Phase 4: Reasoning-Integrated SFT Data Generation**
*   **Path Traversal:** Searches paths: `Incident <- Cause -> Violation -> Regulation`.
*   **Chain-of-Thought Generation:** Generates a reasoning chain (e.g., "This case shows... (Violation Node); based on graph analysis... (Regulation Node)"). Uses Retrieval (RAG-like) if no graph path exists.

---

## 8. Split SFT Data

### Python File: `split_data_openai.ipynb`
*   **Idea:** Split the raw occupational hazard training dataset (`sft_training_data_final.jsonl`) into high-quality Training, Validation, and Testing sets.
    *   *Note:* An alternative with content created by Qwen 2.5-7B is in the "alternative" folder.
*   **Inputting Files:** `knowledge_graph_final.json`; `osh_doc_merged.json`; `boxe_validation_set_clean.json`.
*   **Outputting Files:** `osh_legal_ground_truth_cleaned`; `sft_training_data_final`.
*   **Execution Time:** fast (without content creation); extremely time-consuming if content creation is entailed.

### Overview of the Code Implementation:

**1. Core Challenge: Data Imbalance**
*   Occupational accident data exhibits a "long-tail effect."

**2. Solution: AI-Assisted Stratified Split**
*   **Automated Labeling (AI Labeling):** Uses OpenAI's `gpt-4o-mini` to determine accident category based on official table.
*   **Caching Mechanism:** Uses `incident_categories_cache.json`.
*   **Stratified Sampling:** Uses sklearn's `train_test_split` with stratify. Fallback to random if samples are too few.

**3. Workflow Summary**
1.  **Load:** Import raw JSONL.
2.  **Classify:** LLM determines "Accident Type".
3.  **Split:** Stratified splitting to generate `train.jsonl`, `val.jsonl`, `test.jsonl`.
4.  **Output:** Save to `split_dataset_smart/`.

---

## 9. File Collection
*   This folder collects all important files that are produced before the final training and evaluation work.

---

## 10. Train and Evaluation

### Python File: `Entire_Pipeline_GraphAdapter_to_Evaluation.ipynb`
*   **Idea:** The primary goal of this notebook is to split the raw occupational hazard training dataset (`sft_training_data_final.jsonl`) into high-quality Training, Validation, and Testing sets.
    *   *Note:* An alternative with content created by Qwen 2.5-7B is in the "alternative" folder.
*   **Inputting File:** All documents that are in "9_file_collection" folder.
*   **Outputting File:** results of:
    1.  Vanilla Qwen 2.5-7B
    2.  Qwen2.5-7B + RAG (legal content, not OSH incident documents)
    3.  Qwen2.5-7B + RGAT + BoxE (our method)
    4.  GPT-4o by openai api.
*   **Execution Time:** time-consuming.

### Overview of the Code Implementation:

**1. Phase 1: Infrastructure & Configuration**
*   Unified `ConfigManager` (Colab vs. Local), strict seed locking.

**2. Phase 2: Geometric KG Parsing**
*   Parse BoxE embedding tensors into geometric structures (Center/Width).
*   `NeighborhoodCacher` to pre-compute n-hop neighbors.

**3. Phase 3: Data Engineering**
*   `DynamicAlignmentMapper` to convert legal text into precise Entity IDs.

**4. Phase 4: RAG Baseline**
*   Sparse retrieval (BM25) + Dense retrieval (BGE-M3) + `LegalReranker`.

**5. Phase 5: Neuro-Symbolic Architecture**
*   `NeuroSymbolicQwen` model.
*   `GraphProjector` and `GraphAttentionPooling` to map geometric features into LLM's semantic space.

**6. Phase 6: BoxE Logic Engine**
*   Explicit reasoning using geometric spatial operations (Intersection/Volume).
*   Check if "Incident Event + Violation Relation" falls within "Law Box".

**7. Phase 7: Two-Stage Training**
*   "Modality Alignment" (freeze LLM, train Projector).
*   "Instruction Tuning" (unfreeze Projector + LoRA).
*   `IDBridge` via N-gram alignment.

**8. Phase 8: Unified Inference Pipeline**
*   `InferenceStrategy` factory pattern to standardize prompts across modes.

**9. Phase 9: Multi-Dimensional Evaluation**
*   `LegalExtractor` for structured citations.
*   "Strict Matching" (ID Match), "Soft Matching" (Name Match), BERTScore/ROUGE.

### How to implement "RGAT+BoxE" framework into the LLM?
Integrates RGAT (structural awareness) and BoxE (geometric properties) into Qwen.

**1. Data Parsing & Preparation (Phase 2)**
*   **Geometric Parsing (BoxEParser):** Splits `final_embedding.pt` into Centers and Widths.
*   **Neighbor Pre-computation (NeighborhoodCacher):** Pre-calculates 1-hop/2-hop neighbors (BFS).

**2. Neural Architecture Design (Phase 5)**
*   **Implementing RGAT (GraphAttentionPooling):**
    *   *Query:* BoxE embedding of incident.
    *   *Key/Value:* BoxE embeddings of neighbors.
    *   *Output:* Graph Context Vector.
*   **Dimension Projection (GraphProjector):** Adapter (Linear -> LayerNorm -> ReLU -> Dropout -> Linear).
*   **Embedding Layer Injection:** Replaces special `Placeholder Token` (e.g., `<|extra_0|>`) with `Graph Context Vector`.

**3. BoxE Logical Inference (Phase 6 & 8)**
*   **Geometric Subsumption (BoxELogicEngine):** Calculates intersection/containment mathematically.
*   **Prompt Augmentation:** Suggestions converted to text and appended to System Prompt.

**Summary:** Parse -> Aggregate -> Project & Inject -> Calculate.

---

## Future Work A: System Decomposition and Comparative Analysis
Restructure into five distinct components for benchmarking.

**1. Modular Experiment Design**
We will restructure the project into five distinct components:
*   **(1) Baseline A: Vanilla Qwen 2.5-7B:** Zero-shot/Few-shot.
*   **(2) Baseline B: Qwen 2.5-7B + RAG:** Restricted to legal content. Hybrid search.
*   **(3) Our Method: Qwen 2.5-7B + RGAT + BoxE:** Full Neuro-Symbolic architecture.
*   **(4) Skyline: GPT-4o:** Performance ceiling.
*   **(5) Comprehensive Comparison & Analysis:** Centralized analysis module.

**2. Unified Data Pipeline**
*   **Single Source of Truth:** Exact same `train/val/test` splits.
*   **Data Versioning:** Fingerprinting (MD5).

**3. Standardized Evaluation Protocol**
*   **Output CSV:** `incident_id`, `generated_response`, `ground_truth`, `latency`.
*   **Metrics:** ROUGE-L, BERT-F1, Legal Citation Accuracy.

## Future Work B: Expand Global Ontology and Create Multiple Graphs with Respective Training Method
For current graph, there are several problems:
1.  Current "ontology" vs "standard ontology".
2.  Training complication (focused only on Incident -> Cause -> Violation -> Regulation).
3.  "Cause" nodes are not atomized.

## Future Work C: Enhance the Robustness of Evaluation Section based on References
*   Reference: **LawBench** (Memorization, Understanding, Application).
*   open-compass/LawBench: Benchmarking Legal Knowledge of Large Language Models

## Future Work D: Allocate Complex Reasoning to Multiple Agents
*   Reference: **Chatlaw** (Multi-Agent Collaborative Legal Assistant).
*   [2306.16092] Chatlaw: A Multi-Agent Collaborative Legal Assistant with Knowledge Graph Enhanced Mixture-of-Experts Large Language Model

---

# Preliminary Results
* Qwen 2.5-7B + RGAT + BoxE: BERT-F1 44.05%; ROUGE-L 21.16%.
* GPT-4o: BERT-F1 44.23%; ROUGE-L 17.90%.

# Version and Updates
* 2026/1/1: Upload preliminary work and results.

# Contributors
* Hsu Kuan, Huang
* Wen You, Chen
* Po Yu, Chen

# Acknowledgment
* Instructor: Chia Kai, Liu (CK)
* Professor: Chung Pei, Pien
* Data Source: Occupational Safety and Health Adminstration, Ministry of Labor
* Data Source: Labor Affairs Bureau of Taichung City Government
