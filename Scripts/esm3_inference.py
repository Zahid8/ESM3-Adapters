import torch
import csv
from esm.models.esm3 import ESM3
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from esm.data import FastaBatchedDataset
from torch.utils.data import DataLoader
from torch import nn

# Configuration
CHECKPOINT_PATH = "/scratch/user/zahidhussain909/esm3-darpins/finetuned_models/esm3_lora_delete.pt"
TEST_FASTA_PATH = "/scratch/user/zahidhussain909/esm3-darpins/fasta/test.fasta"
OUTPUT_CSV = "/scratch/user/zahidhussain909/esm3-darpins/Scripts/logs/LoRA_eval_random.csv"
BATCH_SIZE = 4
MASK_POSITIONS = [33, 35, 36, 38, 46, 47, 66, 68, 69, 71, 79, 80, 99, 101, 102, 104, 112, 113]
USE_FIXED_MASKING = False  # Set to False to use random masking
MASK_PROB = 0.15

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(checkpoint_path: str, device: torch.device) -> ESM3:
    print("Loading Model")
    loaded = torch.load(checkpoint_path, map_location=device)
    if isinstance(loaded, dict):
        model = ESM3.from_pretrained("esm3_sm_open_v1", device=device)
        model.load_state_dict(loaded)
    else:
        model = loaded
    model.to(device).to(torch.float32)
    model.eval()
    print("Model Load Complete")
    return model

def get_dataloader(fasta_path: str, batch_size: int, device: torch.device):
    tokenizer = EsmSequenceTokenizer()

    def collate_fn(batch):
        labels, seqs = zip(*batch)
        enc = tokenizer(list(seqs), return_tensors="pt", padding=True)
        tokens = enc["input_ids"].to(device)
        return labels, seqs, tokens

    dataset = FastaBatchedDataset.from_file(fasta_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return loader, tokenizer

def apply_fixed_masking(tokens: torch.Tensor, positions: list[int], mask_token_id: int):
    print("Applying Fixed Masking")
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    seq_len = tokens.size(1)
    valid = [p for p in positions if 0 <= p < seq_len]
    for p in valid:
        mask[:, p] = True
    masked = tokens.clone()
    masked[mask] = mask_token_id
    return masked, mask

def apply_random_masking(tokens: torch.Tensor, mask_token_id: int, mask_prob: float):
    print("Applying Random Masking")
    masked = tokens.clone()
    mask = torch.rand(tokens.shape, device=tokens.device) < mask_prob
    mask[:, 0] = False
    mask[:, -1] = False
    masked[mask] = mask_token_id
    return masked, mask

def evaluate(model: ESM3, tokenizer: EsmSequenceTokenizer, dataloader: DataLoader, output_csv: str):
    pad_id = tokenizer.pad_token_id
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

    total_loss_value = 0.0
    total_masked_tokens = 0
    total_correct_preds = 0

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["sequence_id", "position", "original_token", "predicted_token", "batch_loss"])
        print("Starting Evaluation")
        with torch.no_grad():
            for _, (labels, _, tokens) in enumerate(dataloader):
                if USE_FIXED_MASKING:
                    masked_tokens, mask = apply_fixed_masking(tokens, MASK_POSITIONS, tokenizer.mask_token_id)
                else:
                    masked_tokens, mask = apply_random_masking(tokens, tokenizer.mask_token_id, MASK_PROB)

                if mask.sum().item() == 0:
                    continue

                out = model(sequence_tokens=masked_tokens)
                logits = out.sequence_logits

                masked_logits = logits[mask]
                targets = tokens[mask]

                loss_val = loss_fn(masked_logits, targets)
                num_masked = targets.size(0)

                total_loss_value += loss_val.item() * num_masked
                total_masked_tokens += num_masked

                pred_indices = masked_logits.argmax(dim=-1)
                total_correct_preds += (pred_indices == targets).sum().item()

                orig_tokens = tokenizer.convert_ids_to_tokens(targets.tolist())
                pred_tokens = tokenizer.convert_ids_to_tokens(pred_indices.tolist())

                positions = mask.nonzero(as_tuple=False)
                for idx in range(positions.size(0)):
                    b, pos = positions[idx].tolist()
                    seq_id = labels[b]
                    writer.writerow([seq_id, pos, orig_tokens[idx], pred_tokens[idx], f"{loss_val.item():.4f}"])

        if total_masked_tokens > 0:
            avg_loss = total_loss_value / total_masked_tokens
            accuracy = total_correct_preds / total_masked_tokens
            print(f"Test Loss {avg_loss:.4f}, Test Accuracy {accuracy:.4f}")
        else:
            print("No masked tokens were evaluated; cannot compute loss or accuracy.")

def main():
    device = get_device()
    model = load_model(CHECKPOINT_PATH, device)
    dataloader, tokenizer = get_dataloader(TEST_FASTA_PATH, BATCH_SIZE, device)
    evaluate(model, tokenizer, dataloader, OUTPUT_CSV)
    print("Completed Evaluation")

if __name__ == "__main__":
    main()
