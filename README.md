## Introduction
This repository holds the code for a genomic large language model designed to produce sequence embeddings approximating the edit distance. It is trained via contrastive
learning based on a pretrained DNA large laugage model. The details are included in the paper: Edit Distance Embedding with Genomic Large Language Model.

## Model

The pretrained models are available on [Hugging Face](https://huggingface.co) under the following repositories:
- [`PSUXL/LLMED-MAE`](https://huggingface.co/PSUXL/LLMED-MAE)
- [`PSUXL/LLMED-triplet`](https://huggingface.co/PSUXL/LLMED-triplet)
- [`PSUXL/LLMED-combined`](https://huggingface.co/PSUXL/LLMED-combined)


### Usage

These models are trained based on the DNABERT2 model strucuture. Here is an example code snippet to generate embeddings using the PSUXL/LLMED-MAE model:
```
import torch
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig

# Load DNABERT2 tokenizer and configuration
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")

# Load model
model = AutoModel.from_pretrained("PSUXL/LLMED-MAE", trust_remote_code=True, config=config)

dna = "AGAGCGACGACGTGTAGCAGCTGTACGACTGAGC"

# Get sequence embedding with mean pooling
inputs = tokenizer(dna, return_tensors = 'pt')["input_ids"]
hidden_states = model(inputs)[0] # [1, sequence_length, 768]
embedding_mean = torch.mean(hidden_states[0], dim=0)
```

## Experiments
The repository includes code for two experiments:


### Edit Distance Approximation
This experiment evaluates the correlation between the distances between sequence embeddings and the actual edit distances and the approximation error between the prediction and actual edit distances.
The codes are at edit_distance/. To compute the correlation and approximation error:

```
cd ./edit_distance
python3 main.py sampledata PSUXL/LLMED-MAE
```

### Similar Sequence Search
This experiment demonstrates the model's ability to identify most similar sequences for a given input sequence. The code for this experiment can be found in the similar_sequence_search/ directory. We adopted the pipeline and code from [`Convolutional Embedding for Edit Distance`](https://github.com/xinyandai/string-embed) and integrated our model into the workflow.

```
cd ./similar_sequence_search
python3 main.py --dataset sampledata --nt 100 --nq 100 --save-split --recall --embed bert --model-dir PSUXL/LLMED-MAE
```

### Phylogeny Reconstruction
This experiment reconstructs phylogenetic trees by calculating the pairwise distance between sequences based on their embeddings. These distances are then used to infer evolutionary relationships.

The workflow consists of two main steps:

1. Tree Construction: Generating a phylogenetic tree from the sequences in sample.fa and saving the output to llm.treefile.

2. Evaluation: Comparing the generated tree against a ground-truth tree to calculate the normalized Robinson-Foulds (nRF) score, which measures topological similarity.

```
cd ./phylogeny/

# 1. Generate the phylogenetic tree using model embeddings
python3 phy.py sample.fa PSUXL/LLMED-MAE llm.treefile 

# 2. Calculate the nRF score against the ground truth
python3 RF.py llm.treefile gt.treefile
```