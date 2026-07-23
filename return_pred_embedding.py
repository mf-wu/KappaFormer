import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from models.kappaformer.kappaformer import Kappaformer
from datasets.lmdb_dataset import get_dataloader
import os

def main(args):
    # -------------------------
    # Step 1: Select label keys
    # -------------------------
    kappa_keys = ['id', 'kappa_log', 'density', 'V1', 'n_atoms', 'Ma']
    bg_keys = ['B_VRH', 'G_VRH']

    if args.label_type == "bg":
        label_keys = bg_keys
        model = Kappaformer(enable_kappa=False, label_keys=label_keys, return_embedding=True)
    elif args.label_type == "kappa":
        label_keys = kappa_keys
        model = Kappaformer(enable_kappa=True, label_keys=label_keys, return_embedding=True)
    else:
        raise ValueError("label_type must be 'bg' or 'kappa'")

    # -------------------------
    # Step 2: Load model
    # -------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(state_dict,strict=True)
    model.eval()

    # -------------------------
    # Step 3: Load dataset
    # -------------------------
    dataloader = get_dataloader(
        args.data_path,
        split=args.split,
        batch_size=1,
        label_keys=label_keys
    )

    # -------------------------
    # Step 4: Inference
    # -------------------------
    if args.label_type == "bg":
        # mp_ids, b_preds, g_preds, b_cals, g_cals, harm_embeddings = [], [], [], [], [], []
        b_preds, g_preds, b_cals, g_cals, harm_embeddings = [], [], [], [], []
        for data_batch in tqdm(dataloader):
            # Predict B, G, and extract embedding
            # mp = data_batch['mp_id']
            # if isinstance(mp, (list, tuple)) and len(mp) == 1:
            #     mp = mp[0]
            # if isinstance(mp, torch.Tensor):
            #     mp = mp.detach().cpu().item() if mp.dim() == 0 else mp.detach().cpu().numpy().tolist()
            b, g, harm_embedding = model(data_batch)
            
            # mp_ids.append(mp)
            # Ground truth values
            b_cals.append(data_batch['B_VRH'])
            g_cals.append(data_batch['G_VRH'])

            # Predictions
            b_preds.append(b.detach())
            g_preds.append(g.detach())

            # Embedding
            harm_embeddings.append(harm_embedding.detach())

        b_ps = [x.item() if isinstance(x, torch.Tensor) else float(x) for x in b_preds]
        g_ps = [x.item() if isinstance(x, torch.Tensor) else float(x) for x in g_preds]
        b_cs = [x.item() if isinstance(x, torch.Tensor) else float(x) for x in b_cals]
        g_cs = [x.item() if isinstance(x, torch.Tensor) else float(x) for x in g_cals]

        df_pred_bg = pd.DataFrame({
            # "mp_id": mp_ids,
            "b_pred": b_ps,
            "b_cal": b_cs,
            "g_pred": g_ps,
            "g_cal": g_cs
        })

        # Embeddings
        harm_ebeds = np.vstack([e.cpu().numpy() for e in harm_embeddings])
        df_harm_ebeds = pd.DataFrame(harm_ebeds, columns=[f"Harm_emb_{i}" for i in range(harm_ebeds.shape[1])])
        # Concatenate predictions and embeddings
        df_bg = pd.concat([df_pred_bg, df_harm_ebeds], axis=1)
        # -------------------------
        # Save results
        # -------------------------
        split = args.split
        original_path = args.output_path
        dir_path = os.path.dirname(original_path)
        file_name = os.path.basename(original_path)  # preds_embeddings.csv
        name, ext = os.path.splitext(file_name)  # name='preds_embeddings', ext='.csv'
        file_BG = os.path.join(dir_path, f"{name}_BG_{split}{ext}")
        
        df_bg.to_csv(file_BG, index=False)
        print(f"B/G results saved to {file_BG}")
        
    else:
        ids, kappa_preds, kappa_cals, harm_embeddings, anharm_embeddings = [], [], [], [], []
        for data_batch in tqdm(dataloader):
            # kappa_keys mode (incomplete, depends on model output structure)
            id = data_batch['id']
            if isinstance(id, (list, tuple)) and len(id) == 1:
                id = id[0]
            if isinstance(id, torch.Tensor):
                id = id.detach().cpu().item() if id.dim() == 0 else id.detach().cpu().numpy().tolist()

            b, g, kappa_log, harm_embedding, anharm_embedding = model(data_batch)

            ids.append(id)
            kappa_preds.append(kappa_log.detach())
            kappa_cals.append(data_batch['kappa_log'])
            harm_embeddings.append(harm_embedding.detach())
            anharm_embeddings.append(anharm_embedding.detach())

        kappa_ps = [x.item() if isinstance(x, torch.Tensor) else float(x) for x in kappa_preds]
        kappa_cs = [x.item() if isinstance(x, torch.Tensor) else float(x) for x in kappa_cals]
        df_pred_kappa = pd.DataFrame({
            "id": ids,
            "kappa_log_pred": kappa_ps,
            "kappa_log_cal": kappa_cs
        })

        # Embeddings
        harm_ebeds = np.vstack([e.cpu().numpy() for e in harm_embeddings])
        anharm_ebeds = np.vstack([e.cpu().numpy() for e in anharm_embeddings])
        df_harm_ebeds = pd.DataFrame(harm_ebeds, columns=[f"Harm_emb_{i}" for i in range(harm_ebeds.shape[1])])
        df_anharm_ebeds = pd.DataFrame(anharm_ebeds, columns=[f"Anharm_emb_{i}" for i in range(anharm_ebeds.shape[1])])
    
        # Concatenate predictions and embeddings
        df_kappa = pd.concat([df_pred_kappa, df_harm_ebeds, df_anharm_ebeds], axis=1)
        # -------------------------
        # Save results
        # -------------------------
        split = args.split
        original_path = args.output_path
        dir_path = os.path.dirname(original_path)
        file_name = os.path.basename(original_path)  # preds_embeddings.csv
        name, ext = os.path.splitext(file_name)  # name='preds_embeddings', ext='.csv'名
        file_kappa = os.path.join(dir_path, f"{name}_kappa_{split}{ext}")
        
        df_kappa.to_csv(file_kappa, index=False)
        print(f"Kappa results saved to {file_kappa}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract predictions and embeddings")

    parser.add_argument("--split", type=str, choices=["train", "valid", "test"], required=True,
                        help="Dataset split to use")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to dataset")
    parser.add_argument("--label_type", type=str, choices=["bg", "kappa"], required=True,
                        help="Type of label to use ('bg' or 'kappa')")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save the output CSV")

    args = parser.parse_args()
    main(args)
