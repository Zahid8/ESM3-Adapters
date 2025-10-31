import copy
import csv

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from icecream import ic

from esm.models.esm3 import ESM3
from esm.data import FastaBatchedDataset
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

from peft import LoraConfig, get_peft_model

# ────────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────────
LOG_PATH = "/scratch/user/zahidhussain909/esm3-darpins/Scripts/logs/log_lora.csv"
TRAIN_FASTA_PATH = "/scratch/user/zahidhussain909/esm3-darpins/fasta/train.fasta"
TEST_FASTA_PATH = "/scratch/user/zahidhussain909/esm3-darpins/fasta/test.fasta"
MODEL_SAVE_PATH = "/scratch/user/zahidhussain909/esm3-darpins/finetuned_models/esm3_lora.pt"
#MASK_POSITIONS = [33, 35, 36, 38, 46, 47, 66, 68, 69, 71, 79, 80, 99, 101, 102, 104, 112, 113]
MASK_POSITIONS = [85, 73, 134, 58, 91, 61, 41, 53, 124, 127, 94, 89, 122, 62, 40, 102, 78, 86, 82, 106, 143, 109, 98, 101, 116]
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 50
LORA_RANK = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.1


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_base_model(device):
    model = (
        ESM3.from_pretrained("esm3_sm_open_v1", device=device)
        .to(device)
        .to(torch.float32)
    )
    model.train()
    return model


def wrap_with_lora(base_model):
    config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        use_dora=True,                      # use DORA (Weight-Decomposed Low-Rank Adaptation)
        target_modules=[
            "attn.layernorm_qkv.1",
            "attn.out_proj",
            "geom_attn.proj",
            "geom_attn.out_proj",
            "ffn.1",
            "ffn.3",
            # "output_heads.sequence_head.0",
            # "output_heads.sequence_head.3",
            # "output_heads.structure_head.0",
            # "output_heads.structure_head.3",
            # "output_heads.ss8_head.0",
            # "output_heads.ss8_head.3",
            # "output_heads.sasa_head.0",
            # "output_heads.sasa_head.3",
            # "output_heads.function_head.0",
            # "output_heads.function_head.3",
            # "output_heads.residue_head.0",
            # "output_heads.residue_head.3",
        ],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        modules_to_save=["lm_head"],        # “lm_head” is the ESM3 masked-LM head
    )
    lora_model = get_peft_model(base_model, config)
    lora_model.print_trainable_parameters()
    lora_model.train()
    return lora_model


def get_data_loaders(device):
    tokenizer = EsmSequenceTokenizer()
    
    def collate_fn(batch):
        labels, seqs = zip(*batch)
        enc = tokenizer(list(seqs), return_tensors="pt", padding=True)
        tokens = enc["input_ids"].to(device)
        return labels, seqs, tokens
    
    train_dataset = FastaBatchedDataset.from_file(TRAIN_FASTA_PATH)
    test_dataset = FastaBatchedDataset.from_file(TEST_FASTA_PATH)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return tokenizer, train_loader, test_loader


def apply_fixed_masking(tokens, positions_to_mask, mask_token_id):
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    seq_len = tokens.size(1)
    valid_positions = [pos for pos in positions_to_mask if 0 <= pos < seq_len]
    for pos in valid_positions:
        mask[:, pos] = True
    masked_tokens = tokens.clone()
    masked_tokens[mask] = mask_token_id
    return masked_tokens, mask

def apply_random_masking(tokens, mask_prob=0.15):
    masked = tokens.clone()
    mask   = torch.rand(tokens.shape, device=device) < mask_prob
    mask[:,  0] = False  # no mask on BOS token
    mask[:, -1] = False  # no mask on EOS token
    masked[mask] = tokenizer.mask_token_id
    return masked, mask

def train(lora_model, tokenizer, train_loader, test_loader, device):
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, lora_model.parameters()),
        lr=LEARNING_RATE,
    )
    loss_fn = nn.CrossEntropyLoss()
    
    with open(LOG_PATH, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch", "batch_index", "train_loss", "train_accuracy"])
        
        for epoch in range(EPOCHS):
            #id2tok = {v: k for k, v in tokenizer.get_vocab().items()}
            for batch_idx, (_, _, tokens) in enumerate(train_loader):
                masked_tokens, mask = apply_fixed_masking(
                    tokens, MASK_POSITIONS, tokenizer.mask_token_id
                )
                if mask.sum() == 0:
                    continue

                # <-- Debugging: print shapes and masked tokens -->
                #ic(masked_tokens.shape, mask.sum().item())
                # for pos in positions_to_mask:
                #     ids_at_pos = tokens[:, pos]
                #     toks_all = [id2tok[i.item()] for i in ids_at_pos]
                #     ic(f"pos {pos}", toks_all)

                #masked_tokens, mask = apply_random_masking(tokens)
                
                avg_plddt = torch.ones_like(tokens, dtype=torch.float32, device=device)
                per_res_plddt = torch.zeros_like(tokens, dtype=torch.float32, device=device)
                
                outputs = lora_model(
                    sequence_tokens=masked_tokens,
                    average_plddt=avg_plddt,
                    per_res_plddt=per_res_plddt,
                )
                logits = outputs.sequence_logits
                masked_logits = logits[mask]
                targets = tokens[mask]
                
                loss = loss_fn(masked_logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                loss_val = loss.item()

                preds = masked_logits.argmax(dim=-1)
                correct = (preds == targets).sum().item()
                acc = correct / targets.size(0)
                
                if batch_idx % 10 == 0:
                    writer.writerow([epoch, batch_idx, f"{loss_val:.4f}", f"{acc:.4f}"])
                    csvfile.flush()
                    print(f"Epoch {epoch}, Batch {batch_idx}, Training Loss {loss_val:.4f}, Training Accuracy {acc:.4f}")

            if (epoch + 1) % 10 == 0:
                save_model(lora_model)
                quick_eval(lora_model, tokenizer, test_loader, device)


def quick_eval(lora_model, tokenizer, test_loader, device):
    lora_model.eval()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    with torch.no_grad():
        for _, _, tokens in test_loader:
            masked_tokens, mask = apply_fixed_masking(
                tokens,
                MASK_POSITIONS,
                tokenizer.mask_token_id
            )
            if mask.sum().item() == 0:
                continue

            avg_plddt = torch.ones_like(tokens, dtype=torch.float32, device=device)
            per_res_plddt = torch.zeros_like(tokens, dtype=torch.float32, device=device)

            outputs = lora_model(
                sequence_tokens=masked_tokens,
                average_plddt=avg_plddt,
                per_res_plddt=per_res_plddt
            )
            logits = outputs.sequence_logits

            masked_logits = logits[mask]
            targets = tokens[mask]

            loss = loss_fn(masked_logits, targets)
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size

            preds = masked_logits.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_tokens += batch_size

    average_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens

    print(f"Test Loss {average_loss:.4f}, Test Accuracy {accuracy:.4f}")
    lora_model.train()


def sanity_check(lora_model, test_loader, device):
    lora_model.eval()
    with torch.no_grad():
        labels_t, seqs_t, tokens_t = next(iter(test_loader))
        out_t = lora_model(sequence_tokens=tokens_t)
        reps = out_t.embeddings
        cls_emb = reps[:, 0, :]
        mean_emb = reps.mean(dim=1)
        # ic(cls_emb.shape, mean_emb.shape)
    lora_model.train()


def save_model(lora_model):
    temp_model = copy.deepcopy(lora_model)
    merged_state = temp_model.merge_and_unload()
    torch.save(merged_state, MODEL_SAVE_PATH)
    del temp_model


def main():
    device = get_device()
    base_model = load_base_model(device)
    lora_model = wrap_with_lora(base_model)
    tokenizer, train_loader, test_loader = get_data_loaders(device)
    train(lora_model, tokenizer, train_loader, test_loader, device)


if __name__ == "__main__":
    main()
