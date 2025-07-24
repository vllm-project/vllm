def filter_missing_classis(params, classi_idx, warn=True):
    # Filter the params that are not found.
    missing_params = []
    for i, p in enumerate(params):
        if p["name"] not in classi_idx:
            missing_params.append(i)
            if warn:
                print(f"WARNING: penalty classifiers not found: {p['name']}")

    if missing_params:
        params = [p for i, p in enumerate(params) if i not in missing_params]
    return params
