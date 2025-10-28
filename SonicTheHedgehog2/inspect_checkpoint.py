# inspect_checkpoint.py
import sys, os, re, json
import torch

CKPT = "sonic_ppo_latest.pt"

def pretty(v):
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(str(x) for x in v[:8]) + (", ...]" if len(v) > 8 else "]")
    return str(v)

def main(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    # It might be either a raw state_dict or a dict with nested fields:
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and all(isinstance(k, str) and torch.is_tensor(v) for k, v in ckpt.items()):
        sd = ckpt
    else:
        # Best effort: find the biggest dict of tensors
        sd = None
        for k, v in (ckpt.items() if isinstance(ckpt, dict) else []):
            if isinstance(v, dict) and all(isinstance(kk, str) and torch.is_tensor(vv) for kk, vv in v.items()):
                sd = v; break
        if sd is None:
            print("Could not find a tensor state_dict in checkpoint keys:", list(ckpt.keys())[:20])
            return

    print("=== Top-level keys in checkpoint ===")
    if isinstance(ckpt, dict):
        print([k for k in ckpt.keys()])

    print("\n=== State dict summary (first 50 keys) ===")
    keys = list(sd.keys())
    for k in keys[:50]:
        t = sd[k]
        print(f"{k:60s}  {tuple(t.shape)}  {t.dtype}")

    # Infer action count: look for the final policy logits layer
    policy_candidates = [k for k in keys if re.search(r"(pi|policy|actor).*weight", k)]
    head = None
    for k in policy_candidates:
        W = sd[k]
        # Usually Linear(out, in) so out_features = W.shape[0]
        if W.ndim == 2:
            head = (k, W.shape[0], W.shape[1])
    if head:
        print("\n[Inference] Policy head:", head[0], "=> action_count =", head[1])
    else:
        # fallback: look for something named logits
        for k in keys:
            if "logits" in k and sd[k].ndim == 1:
                print("\n[Inference] Found logits bias:", k, "=> action_count =", sd[k].numel())
                break

    # Try to infer conv trunk input channels from first conv weight
    conv_candidates = [k for k in keys if re.search(r"(conv|features\.0|encoder\..*0).*weight", k)]
    if conv_candidates:
        W = sd[conv_candidates[0]]
        if W.ndim == 4:  # (out, in, kH, kW)
            print(f"\n[Inference] First conv in_channels: {W.shape[1]}  (likely frame stack)")
            print(f"[Inference] First conv kernel: {W.shape[2]}x{W.shape[3]}")

    # Look for running norm buffers commonly used by obs/reward normalizers
    norm_keys = [k for k in keys if re.search(r"(running_mean|running_var|obs_rms|ret_rms|rew_rms)", k)]
    if norm_keys:
        print("\n=== Normalization buffers (obs/reward) ===")
        for k in norm_keys:
            t = sd[k]
            print(f"{k:45s}  shape={tuple(t.shape)}  mean~{float(t.mean()):.4g} std~{float(t.std()):.4g}")

    # Sometimes people save action mappings or metadata in the checkpoint
    meta = {}
    for meta_key in ("action_meanings", "action_map", "combos", "buttons", "env_info", "config", "args"):
        if isinstance(ckpt, dict) and meta_key in ckpt and not torch.is_tensor(ckpt[meta_key]):
            meta[meta_key] = ckpt[meta_key]
    if meta:
        print("\n=== Metadata ===")
        for k, v in meta.items():
            try:
                print(k, "=", json.dumps(v))
            except Exception:
                print(k, "=", str(v))

if __name__ == "__main__":
    path = CKPT if len(sys.argv) == 1 else sys.argv[1]
    if not os.path.exists(path):
        print("File not found:", path)
        sys.exit(1)
    main(path)
