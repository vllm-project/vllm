import os
if os.environ.get("VLLM_PERDIM_ROPE_FILE"):
    try:
        import patch_perdim_rope
        patch_perdim_rope.apply()
    except Exception as e:                       # nunca tumbar el arranque por el parche
        import sys; print("[perdim-rope] no aplicado:", e, file=sys.stderr)
