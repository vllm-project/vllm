// vLLM Windows GUI Launcher
// Compile: csc /target:winexe /out:vllm.exe launcher.cs
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Threading;
using System.Windows.Forms;

class VllmGui : Form
{
    private TextBox txtModel, txtPort, txtMaxModelLen, txtGpuMem, tvCacheDtype;
    private TextBox txtTpSize, txtCpuOffload, txtBlockSize, txtSeed, txtMaxBatch;
    private TextBox txtMaxTokens, txtTemp, txtTopP, txtTopK, txtRepPenalty;
    private TextBox txtFreqPenalty, txtPresPenalty, txtMinP, txtStop;
    private CheckBox chkEager, chkTrust, chkPrefixCache, chkChunked;
    private CheckBox chkNoStats, chkNoCustomReduce, chkLora, chkReasoning;
    private CheckBox chkHidden, chkIgnoreEos, chkSkipSpecial, chkLogprobs;
    private CheckBox chkFlashAttn, chkTritonAttn;
    private Button btnBrowse, btnStart;
    private RichTextBox txtLog;
    private Process serverProcess;
    private Thread serverThread;
    private bool serverRunning;
    private string repoRoot;
    private Panel scrollPanel;

    [STAThread]
    static void Main()
    {
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);
        Application.Run(new VllmGui());
    }

    public VllmGui()
    {
        Text = "vLLM Server Launcher";
        Size = new Size(920, 700);
        StartPosition = FormStartPosition.CenterScreen;
        Font = new Font("Segoe UI", 9);
        repoRoot = FindVllmRepo(AppDomain.CurrentDomain.BaseDirectory);
        BuildUI();
    }

    private void BuildUI()
    {
        scrollPanel = new Panel { Dock = DockStyle.Fill, AutoScroll = true };
        Controls.Add(scrollPanel);
        int col1 = 15, col2 = 340, y = 10;

        AddSectionLabel("Model", col1, ref y);
        AddField(col1, ref y, "Model path:", txtModel = new TextBox { Size = new Size(400, 22) });
        btnBrowse = AddButton("Browse...", col1 + 485, y - 26, 80);
        btnBrowse.Click += (s, e) => BrowseModel();

        y += 6; AddSectionLabel("Server", col1, ref y);
        AddField(col1, ref y, "Port:", txtPort = MkTb("8001"));
        AddField(col1, ref y, "Max model len:", txtMaxModelLen = MkTb("4096"));
        AddField(col1, ref y, "GPU memory util:", txtGpuMem = MkTb("0.95"));
        AddField(col1, ref y, "KV cache dtype:", tvCacheDtype = MkTb("auto"));
        AddField(col1, ref y, "Tensor parallel:", txtTpSize = MkTb("1"));
        AddField(col1, ref y, "CPU offload GB:", txtCpuOffload = MkTb("0"));
        AddField(col1, ref y, "Block size:", txtBlockSize = MkTb("16"));
        AddField(col1, ref y, "Seed:", txtSeed = MkTb("0"));
        AddField(col1, ref y, "Max batched tokens:", txtMaxBatch = MkTb("8192"));

        int sy = 10; AddSectionLabel("Sampling", col2, ref sy);
        AddField(col2, ref sy, "Max tokens:", txtMaxTokens = MkTb("512"));
        AddField(col2, ref sy, "Temperature:", txtTemp = MkTb("0.7"));
        AddField(col2, ref sy, "Top P:", txtTopP = MkTb("0.95"));
        AddField(col2, ref sy, "Top K:", txtTopK = MkTb("-1"));
        AddField(col2, ref sy, "Repetition penalty:", txtRepPenalty = MkTb("1.0"));
        AddField(col2, ref sy, "Frequency penalty:", txtFreqPenalty = MkTb("0.0"));
        AddField(col2, ref sy, "Presence penalty:", txtPresPenalty = MkTb("0.0"));
        AddField(col2, ref sy, "Min P:", txtMinP = MkTb("0.0"));

        int cy = y + 6, cw = 260;
        AddSectionLabel("Options", col1, ref cy);
        chkEager = AddChk("Enforce eager", col1, ref cy, cw, true);
        chkPrefixCache = AddChk("Prefix caching", col1, ref cy, cw, true);
        chkChunked = AddChk("Chunked prefill", col1, ref cy, cw, true);
        chkNoStats = AddChk("Disable log stats", col1, ref cy, cw, true);
        chkNoCustomReduce = AddChk("Disable custom all-reduce", col1, ref cy, cw, true);
        chkTrust = AddChk("Trust remote code", col1, ref cy, cw, false);
        chkLora = AddChk("Enable LoRA", col1, ref cy, cw, false);
        chkReasoning = AddChk("Enable reasoning", col1, ref cy, cw, false);
        chkHidden = AddChk("Return hidden states", col1, ref cy, cw, false);
        chkFlashAttn = AddChk("Flashinfer attention", col1, ref cy, cw, false);
        chkTritonAttn = AddChk("Triton attention", col1, ref cy, cw, true);
        chkIgnoreEos = AddChk("Ignore EOS", col2, ref cy, cw, false);
        chkSkipSpecial = AddChk("Skip special tokens", col2, ref cy, cw, true);
        chkLogprobs = AddChk("Enable logprobs", col2, ref cy, cw, false);

        int by = Math.Max(cy, sy) + 10;
        btnStart = AddButton("Start Server", col1, by, 140);
        btnStart.ForeColor = Color.Green;
        btnStart.Font = new Font("Segoe UI", 10, FontStyle.Bold);
        btnStart.Click += (s, e) => ToggleServer();
        AddStatusLabel("Status: Stopped", col1 + 155, by + 4, 400);

        int ly = by + 40;
        AddSectionLabel("Server Log", col1, ref ly);
        txtLog = new RichTextBox { Location = new Point(col1, ly), Size = new Size(870, 280),
            ReadOnly = true, BackColor = Color.Black, ForeColor = Color.Lime,
            Font = new Font("Consolas", 9), WordWrap = false,
            Anchor = AnchorStyles.Left | AnchorStyles.Top | AnchorStyles.Right | AnchorStyles.Bottom };
        scrollPanel.Controls.Add(txtLog);
    }

    private TextBox MkTb(string d) { return new TextBox { Text = d, Size = new Size(80, 22) }; }
    private void AddSectionLabel(string t, int x, ref int y) { var l = new Label { Text = t, Location = new Point(x, y), Size = new Size(600, 20), Font = new Font("Segoe UI", 9, FontStyle.Bold), ForeColor = Color.DarkBlue }; scrollPanel.Controls.Add(l); y += 22; }
    private void AddField(int x, ref int y, string label, TextBox tb) { var l = new Label { Text = label, Location = new Point(x, y), Size = new Size(132, 22), TextAlign = ContentAlignment.MiddleLeft }; scrollPanel.Controls.Add(l); tb.Location = new Point(x + 134, y - 1); scrollPanel.Controls.Add(tb); y += 26; }
    private Button AddButton(string t, int x, int y, int w) { var b = new Button { Text = t, Location = new Point(x, y), Size = new Size(w, 28) }; scrollPanel.Controls.Add(b); return b; }
    private CheckBox AddChk(string t, int x, ref int y, int w, bool d) { var c = new CheckBox { Text = t, Location = new Point(x, y), Size = new Size(w, 22), Checked = d }; scrollPanel.Controls.Add(c); y += 24; return c; }
    private Label AddStatusLabel(string t, int x, int y, int w) { var l = new Label { Text = "Status: " + t, Location = new Point(x, y), Size = new Size(w, 24), ForeColor = Color.Gray }; l.Tag = "status"; scrollPanel.Controls.Add(l); return l; }

    private void SetStatus(string t, Color? c = null) {
        foreach (Control x in scrollPanel.Controls)
            if (x is Label && x.Tag != null && x.Tag.ToString() == "status") {
                x.Text = "Status: " + t;
                if (c.HasValue) x.ForeColor = c.Value;
            }
    }

    private void BrowseModel() {
        var dlg = new FolderBrowserDialog { Description = "Select model folder" };
        if (dlg.ShowDialog() == DialogResult.OK) txtModel.Text = dlg.SelectedPath;
    }

    private void ToggleServer() {
        if (serverRunning) StopServer();
        else {
            if (string.IsNullOrEmpty(txtModel.Text) || !Directory.Exists(txtModel.Text))
            { MessageBox.Show("Select a model folder.", "vLLM"); return; }
            if (repoRoot == null)
            { MessageBox.Show("vLLM source not found. Clone the repo first.", "vLLM"); return; }
            StartServerAsync();
        }
    }

    private void StartServerAsync() {
        btnStart.Enabled = false;
        btnStart.Text = "Starting...";
        SetStatus("Starting...", Color.Orange);

        // Capture all UI state on UI thread before starting background work
        string modelPath = txtModel.Text;
        string port = txtPort.Text;
        string maxLen = txtMaxModelLen.Text;
        string gpuMem = txtGpuMem.Text;
        string cacheDt = tvCacheDtype.Text;
        string tpSz = txtTpSize.Text;
        string cpuOff = txtCpuOffload.Text;
        string blkSz = txtBlockSize.Text;
        string sd = txtSeed.Text;
        string maxB = txtMaxBatch.Text;
        bool eager = chkEager.Checked, prefixCache = chkPrefixCache.Checked;
        bool chunked = chkChunked.Checked, noStats = chkNoStats.Checked;
        bool noCustomReduce = chkNoCustomReduce.Checked, trust = chkTrust.Checked;
        bool lora = chkLora.Checked, reasoning = chkReasoning.Checked;
        bool hidden = chkHidden.Checked, flashAttn = chkFlashAttn.Checked;
        bool tritonAttn = chkTritonAttn.Checked;

        // Create a snapshot for the background thread
        var opts = new ServerOptions {
            ModelPath = modelPath, Port = port, MaxModelLen = maxLen,
            GpuMem = gpuMem, CacheDtype = cacheDt, TpSize = tpSz,
            CpuOffload = cpuOff, BlockSize = blkSz, Seed = sd,
            MaxBatch = maxB, Eager = eager, PrefixCache = prefixCache,
            Chunked = chunked, NoStats = noStats, NoCustomReduce = noCustomReduce,
            Trust = trust, Lora = lora, Reasoning = reasoning,
            Hidden = hidden, FlashAttn = flashAttn, TritonAttn = tritonAttn,
        };

        serverThread = new Thread(() => ServerWorker(opts));
        serverThread.IsBackground = true;
        serverThread.Start();
    }

    private class ServerOptions {
        public string ModelPath, Port, MaxModelLen, GpuMem, CacheDtype;
        public string TpSize, CpuOffload, BlockSize, Seed, MaxBatch;
        public bool Eager, PrefixCache, Chunked, NoStats, NoCustomReduce;
        public bool Trust, Lora, Reasoning, Hidden, FlashAttn, TritonAttn;
    }

    private void ServerWorker(ServerOptions o) {
        try {
            // 1. Python
            string python = FindPythonExe(repoRoot);
            if (python == null) { Log("ERROR: Python not found."); Done("Error"); return; }
            Log("Python: " + python);

            // 2. _C.pyd check
            string pyd = Path.Combine(repoRoot, "vllm", "_C.pyd");
            if (File.Exists(pyd)) Log("_C.pyd: OK (" + (new FileInfo(pyd).Length / 1048576) + " MB)");
            else Log("WARNING: _C.pyd not found");

            // 3. pip install -e .
            Log("Installing vLLM package...");
            int pipCode = RunProcess(python, "-m pip install -e .", repoRoot, 120000);
            if (pipCode != 0) Log("pip install: exit code " + pipCode);
            else Log("vLLM ready.");

            // 4. Build args from captured options
            List<string> args = new List<string>();
            args.Add("-m vllm.entrypoints.openai.api_server");
            args.Add("--model \"" + o.ModelPath + "\"");
            args.Add("--port " + o.Port);
            args.Add("--max-model-len " + o.MaxModelLen);
            args.Add("--gpu-memory-utilization " + o.GpuMem);
            args.Add("--dtype float16");
            args.Add("--seed " + o.Seed);
            if (o.CacheDtype != "auto") { args.Add("--kv-cache-dtype " + o.CacheDtype); }
            if (o.TpSize != "1") { args.Add("--tensor-parallel-size " + o.TpSize); }
            if (o.CpuOffload != "0") { args.Add("--cpu-offload-gb " + o.CpuOffload); }
            if (o.BlockSize != "16") { args.Add("--block-size " + o.BlockSize); }
            if (o.MaxBatch != "8192") { args.Add("--max-num-batched-tokens " + o.MaxBatch); }
            if (o.Eager) args.Add("--enforce-eager");
            if (o.PrefixCache) args.Add("--enable-prefix-caching");
            if (o.Chunked) args.Add("--enable-chunked-prefill");
            if (o.NoStats) args.Add("--disable-log-stats");
            if (o.NoCustomReduce) args.Add("--disable-custom-all-reduce");
            if (o.Trust) args.Add("--trust-remote-code");
            if (o.Lora) args.Add("--enable-lora");
            if (o.Reasoning) args.Add("--enable-reasoning");
            if (o.Hidden) args.Add("--return-hidden-states");
            if (o.FlashAttn) args.Add("--flashinfer-attention");
            if (o.TritonAttn) args.Add("--triton-attention");

            string argStr = string.Join(" ", args.ToArray());
            Log("Model: " + o.ModelPath);
            Log("Port: " + o.Port);
            Log("Starting server...");

            // 5. Set env — auto-detect ROCm
            string rocmPath = DetectRocm();
            if (rocmPath != null)
                Environment.SetEnvironmentVariable("HIP_PATH", rocmPath);
            Environment.SetEnvironmentVariable("PYTHONPATH", repoRoot);
            Environment.SetEnvironmentVariable("VLLM_NO_USAGE_STATS", "true");

            // 6. Launch server
            serverProcess = new Process();
            serverProcess.StartInfo.FileName = python;
            serverProcess.StartInfo.Arguments = argStr;
            serverProcess.StartInfo.WorkingDirectory = repoRoot;
            serverProcess.StartInfo.UseShellExecute = false;
            serverProcess.StartInfo.RedirectStandardOutput = true;
            serverProcess.StartInfo.RedirectStandardError = true;
            serverProcess.StartInfo.CreateNoWindow = true;
            serverProcess.EnableRaisingEvents = true;
            serverProcess.Exited += (s, e) => {
                serverRunning = false;
                int c = serverProcess.ExitCode;
                Log("Server exited with code: " + c + (c != 0 ? " (ERROR)" : ""));
                Done(c == 0 ? "Stopped" : "Error");
            };
            serverProcess.Start();
            serverRunning = true;
            this.BeginInvoke(new Action(() => {
                btnStart.Text = "Stop Server";
                btnStart.ForeColor = Color.Red;
                btnStart.Enabled = true;
                SetStatus("Running", Color.Green);
            }));

            // 7. Read output
            string line;
            while ((line = serverProcess.StandardOutput.ReadLine()) != null)
                Log(line);
            string rest = serverProcess.StandardOutput.ReadToEnd();
            if (!string.IsNullOrEmpty(rest)) Log(rest);
        }
        catch (Exception ex) { Log("ERROR: " + ex.Message); Done("Error"); }
    }

    private void Log(string msg) {
        this.BeginInvoke(new Action(() => {
            txtLog.AppendText(msg + "\n");
            txtLog.SelectionStart = txtLog.Text.Length;
            txtLog.ScrollToCaret();
        }));
    }

    private void Done(string status) {
        this.BeginInvoke(new Action(() => {
            btnStart.Text = "Start Server";
            btnStart.ForeColor = Color.Green;
            btnStart.Enabled = true;
            SetStatus(status, status == "Running" ? Color.Green : Color.Gray);
        }));
    }

    private void StopServer() {
        if (serverProcess != null && !serverProcess.HasExited)
        { try { serverProcess.Kill(); } catch { } serverProcess.WaitForExit(2000); }
        serverRunning = false;
        Done("Stopped");
        Log("Server stopped.");
    }

    private int RunProcess(string exe, string args, string cwd, int timeoutMs) {
        Process p = new Process();
        p.StartInfo.FileName = exe;
        p.StartInfo.Arguments = args;
        p.StartInfo.WorkingDirectory = cwd;
        p.StartInfo.UseShellExecute = false;
        p.StartInfo.RedirectStandardOutput = true;
        p.StartInfo.RedirectStandardError = true;
        p.StartInfo.CreateNoWindow = true;
        p.Start();
        string outStr = p.StandardOutput.ReadToEnd();
        string errStr = p.StandardError.ReadToEnd();
        p.WaitForExit(timeoutMs);
        if (!string.IsNullOrEmpty(outStr)) Log(outStr.Trim());
        if (!string.IsNullOrEmpty(errStr)) Log(errStr.Trim());
        return p.ExitCode;
    }

    private string FindVllmRepo(string startDir) {
        DirectoryInfo dir = new DirectoryInfo(startDir);
        for (int i = 0; i < 5; i++) {
            if (dir == null) break;
            if (File.Exists(Path.Combine(dir.FullName, "vllm", "__init__.py")))
                return dir.FullName;
            dir = dir.Parent;
        }
        return null;
    }

    private string FindPythonExe(string repoRoot) {
        string[] venvChecks = {
            Path.Combine(repoRoot, ".venv", "Scripts", "python.exe"),
            Path.Combine(repoRoot, "venv", "Scripts", "python.exe"),
        };
        foreach (string v in venvChecks)
            if (File.Exists(v)) return v;

        try {
            Process p = new Process();
            p.StartInfo.FileName = "python";
            p.StartInfo.Arguments = "--version";
            p.StartInfo.UseShellExecute = false;
            p.StartInfo.RedirectStandardOutput = true;
            p.StartInfo.CreateNoWindow = true;
            p.Start(); string ver = p.StandardOutput.ReadToEnd().Trim();
            p.WaitForExit(2000);
            if (p.ExitCode == 0) return "python";
        } catch {}
        return null;
    }

    private string DetectRocm() {
        // Check environment variables first
        string[] envVars = { "ROCM_HOME", "ROCM_PATH", "HIP_PATH" };
        foreach (string env in envVars) {
            string v = Environment.GetEnvironmentVariable(env);
            if (!string.IsNullOrEmpty(v) && File.Exists(v + "/bin/hipcc.exe"))
                return v;
        }
        // Check common install paths
        string[] paths = {
            @"C:\Program Files\AMD\ROCm\7.13",
            @"C:\Program Files\AMD\ROCm\7.12",
            @"C:\Program Files\AMD\ROCm\7.11",
            @"E:\ROCM-7.13.0-Windows",
            @"C:\ROCm\7.13",
        };
        foreach (string p in paths)
            if (File.Exists(p + "/bin/hipcc.exe"))
                return p;
        // Scan for any ROCm version
        try {
            string baseDir = @"C:\Program Files\AMD\ROCm";
            if (Directory.Exists(baseDir)) {
                foreach (string dir in Directory.GetDirectories(baseDir)) {
                    string candidate = dir + @"\bin\hipcc.exe";
                    if (File.Exists(candidate))
                        return dir;
                }
            }
        } catch {}
        return null;
    }
}
