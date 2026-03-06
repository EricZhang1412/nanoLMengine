from pathlib import Path

def resolve_resume_ckpt(resume, ckpt_dir):
    if resume is None:
        return None

    resume = str(resume).strip()
    if resume.lower() in ("none", "null", ""):
        return None

    if resume.lower() != "auto":
        ckpt_path = Path(resume)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {ckpt_path}")
        return str(ckpt_path)

    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        return None

    # auto 模式下只认 last.ckpt
    last_ckpt = ckpt_dir / "last.ckpt"
    if last_ckpt.exists():
        return str(last_ckpt)

    # 再退化到普通 step*.ckpt，不包含 on_exception.ckpt
    all_ckpts = sorted(
        ckpt_dir.glob("step*.ckpt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if all_ckpts:
        return str(all_ckpts[0])

    return None

# restore/retrieve a particular run 
def load_aim_run_hash(ckpt_dir):
    p = Path(ckpt_dir) / "aim_run_hash.txt"
    if p.exists():
        run_hash = p.read_text(encoding="utf-8").strip()
        return run_hash or None
    return None

def save_aim_run_hash(ckpt_dir, run_hash):
    if not run_hash:
        return
    p = Path(ckpt_dir) / "aim_run_hash.txt"
    p.write_text(str(run_hash).strip(), encoding="utf-8")