import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# bert embedding
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
from transformers.models.bert.configuration_bert import BertConfig
os.environ["TOKENIZERS_PARALLELISM"] = "false"



def bert_embedding(vecs, model_path):

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, config=config).to(device)

    vecs = tokenizer(vecs, padding="longest", return_tensors="pt")

    embedding = []

    batch_size = 128 # Adjust based on your GPU memory
    input_ids = vecs["input_ids"]
    attention_masks = vecs["attention_mask"]
    num_samples = input_ids.size(0)

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_ids = input_ids[i : i + batch_size].to(device)
            batch_mask = attention_masks[i : i + batch_size].to(device)

            # Forward pass
            outputs = model(batch_ids, attention_mask=batch_mask)
            hidden = outputs[0]

            sum_embeddings = torch.sum(hidden * batch_mask.unsqueeze(-1), dim=1)
            mask_sum = torch.clamp(batch_mask.sum(dim=1, keepdim=True), min=1e-9)
            
            mean_pooled = sum_embeddings / mask_sum
            embedding.extend(mean_pooled.cpu().numpy())

    return np.array(embedding)