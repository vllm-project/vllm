import marimo

__generated_with = "0.15.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import pandas as pd
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    return Path, go, json, make_subplots, pd


@app.cell
def _():
    all_runs_dir = '/home/ubuntu/vllm/final_outputs/spec_outputs'
    return (all_runs_dir,)


@app.cell
def _(Path, json):
    def parse_run(dir, keep=None):
        dir = Path(dir)
        data = {}
        # read in args
        with open(dir / "args.jsonl") as f:
            for line in f:
                data |= json.loads(line.strip())

        # read in stats
        with open(dir / "stats.jsonl") as f:
            for line in f:
                data |= json.loads(line.strip())

        # parse stats
        if keep:
            data = {k: data[k] for k in keep if k in data}
        return data
    return (parse_run,)


@app.cell
def _(Path, parse_run, pd):
    def parse_all_runs(all_runs_dir, keep=None):
        all_runs_dir = Path(all_runs_dir)
        run_dirs = [all_runs_dir / d.name for d in Path(all_runs_dir).iterdir() if d.is_dir()]
        data = [parse_run(run_dir, keep) for run_dir in run_dirs]
        return pd.DataFrame(data)
    return (parse_all_runs,)


@app.function
def fix_dataframe(df):

    # convert to ints
    df['num_spec_tokens'] = df['num_spec_tokens'].astype(int)

    # sort by num_spec_tokens
    df = df.sort_values('num_spec_tokens')

    # add method column
    def get_method(row):
        if row['num_spec_tokens'] == 0:
            return 'vanilla'
        elif row['draft_vocab_frequency_keep_threshold'] == 'None':
            return 'fr-spec'
        else:
            return 'eagle'
    df['method'] = df.apply(get_method, axis=1)

    return df


@app.cell
def _(all_runs_dir, parse_all_runs):
    keep = ['num_spec_tokens', 'spec_token_tree', 'forward_ratio', 'total_time_sec', 'acceptance_length', 'output_speed', 'draft_vocab_frequency_path', 'draft_vocab_frequency_keep_threshold']
    df_raw = parse_all_runs(all_runs_dir, keep=keep)
    df_raw
    return (df_raw,)


@app.cell
def _(df_raw):
    df = fix_dataframe(df_raw)
    df
    return (df,)


@app.cell
def _(df, go, make_subplots):
    metrics = ['acceptance_length', 'output_speed', 'forward_ratio', 'total_time_sec']
    titles = ['Mean Acceptance Length', 'Decoding Throughput (toks/sec)', 'Forward Pass Ratio (D:T)', 'Total Time (sec)']
    colors = {'vanilla': 'blue', 'eagle': 'red', 'fr-spec': 'green'}

    fig = make_subplots(rows=2, cols=2, subplot_titles=titles)

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        row, col = (i//2) + 1, (i%2) + 1

        for method in ['vanilla', 'eagle', 'fr-spec']:
            data = df[df['method'] == method].sort_values('num_spec_tokens')
            fig.add_trace(
                go.Scatter(x=data['num_spec_tokens'], y=data[metric],
                          mode='lines+markers', name=method,
                          line=dict(color=colors[method]),
                          showlegend=(i==0)),
                row=row, col=col
            )

    fig.update_xaxes(title_text="Num Spec Tokens")
    fig.update_layout(height=600, title="Speculative Decoding Metrics")
    fig.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
