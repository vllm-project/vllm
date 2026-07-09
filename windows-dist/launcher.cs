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
    // --- Server options ---
    private TextBox txtModel, txtPort, txtMaxModelLen, txtGpuMem, tvCacheDtype;
    private TextBox txtTpSize, txtCpuOffload, txtBlockSize, txtSeed, txtMaxBatch;
    // --- Sampling options ---
    private TextBox txtMaxTokens, txtTemp, txtTopP, txtTopK, txtRepPenalty;
    private TextBox txtFreqPenalty, txtPresPenalty, txtMinP, txtStop;
    // --- Checkboxes ---
    private CheckBox chkEager, chkTrust, chkPrefixCache, chkChunked;
    private CheckBox chkNoStats, chkNoCustomReduce, chkLora, chkReasoning;
    private CheckBox chkHidden, chkIgnoreEos, chkSkipSpecial, chkLogprobs;
    private CheckBox chkFlashAttn, chkTritonAttn;
    // --- Controls ---
    private Button btnBrowse, btnStart;
    private RichTextBox txtLog;
    private Process serverProcess;
    private Thread outputThread;
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
        // Scrollable panel for all controls
        scrollPanel = new Panel();
        scrollPanel.Dock = DockStyle.Fill;
        scrollPanel.AutoScroll = true;
        Controls.Add(scrollPanel);

        int col1 = 15, col2 = 340, colW = 280;
        int y = 10, rowH = 26;

        // ===== SECTION: Model =====
        AddSectionLabel("Model", col1, y, 600); y += 22;
        AddLabel("Model path:", col1, y, colW);
        txtModel = AddTextBox(col1 + 80, y, 400);
        btnBrowse = AddButton("Browse...", col1 + 485, y - 2, 80);
        btnBrowse.Click += (s, e) => BrowseModel();
        y += rowH;

        // ===== SECTION: Server =====
        AddSectionLabel("Server", col1, y, 600); y += 22;
        AddField(col1, ref y, "Port:", txtPort = MkTb("8001"));
        AddField(col1, ref y, "Max model len:", txtMaxModelLen = MkTb("4096"));
        AddField(col1, ref y, "GPU memory util:", txtGpuMem = MkTb("0.95"));
        AddField(col1, ref y, "KV cache dtype:", tvCacheDtype = MkTb("auto"));
        AddField(col1, ref y, "Tensor parallel:", txtTpSize = MkTb("1"));
        AddField(col1, ref y, "CPU offload GB:", txtCpuOffload = MkTb("0"));
        AddField(col1, ref y, "Block size:", txtBlockSize = MkTb("16"));
        AddField(col1, ref y, "Seed:", txtSeed = MkTb("0"));
        AddField(col1, ref y, "Max batched tokens:", txtMaxBatch = MkTb("8192"));

        // ===== SECTION: Sampling =====
        y += 6;
        AddSectionLabel("Sampling", col2, 10, colW); 
        int sy = 10 + 22;
        AddField(col2, ref sy, "Max tokens:", txtMaxTokens = MkTb("512"));
        AddField(col2, ref sy, "Temperature:", txtTemp = MkTb("0.7"));
        AddField(col2, ref sy, "Top P:", txtTopP = MkTb("0.95"));
        AddField(col2, ref sy, "Top K:", txtTopK = MkTb("-1"));
        AddField(col2, ref sy, "Repetition penalty:", txtRepPenalty = MkTb("1.0"));
        AddField(col2, ref sy, "Frequency penalty:", txtFreqPenalty = MkTb("0.0"));
        AddField(col2, ref sy, "Presence penalty:", txtPresPenalty = MkTb("0.0"));
        AddField(col2, ref sy, "Min P:", txtMinP = MkTb("0.0"));
        AddField(col2, ref sy, "Stop strings:", txtStop = MkTb(""));

        // ===== SECTION: Checkboxes - Server =====
        int cy = y + 6; int cw = 280;
        AddSectionLabel("Options", col1, y, 600); cy = y + 22;
        
        chkEager = AddCheckBox("Enforce eager mode", col1, ref cy, cw, true);
        chkTrust = AddCheckBox("Trust remote code", col1, ref cy, cw, false);
        chkPrefixCache = AddCheckBox("Enable prefix caching", col1, ref cy, cw, true);
        chkChunked = AddCheckBox("Enable chunked prefill", col1, ref cy, cw, true);
        chkNoStats = AddCheckBox("Disable log stats", col1, ref cy, cw, true);
        chkNoCustomReduce = AddCheckBox("Disable custom all-reduce", col1, ref cy, cw, true);
        chkLora = AddCheckBox("Enable LoRA", col1, ref cy, cw, false);
        chkReasoning = AddCheckBox("Enable reasoning", col1, ref cy, cw, false);
        chkHidden = AddCheckBox("Return hidden states", col1, ref cy, cw, false);

        // Checkboxes - Sampling
        int cy2 = y + 22; int cw2 = 200;
        chkIgnoreEos = AddCheckBox("Ignore EOS", col2, ref cy2, cw2, false);
        chkSkipSpecial = AddCheckBox("Skip special tokens", col2, ref cy2, cw2, true);
        chkLogprobs = AddCheckBox("Enable prompt logprobs", col2, ref cy2, cw2, false);

        // Checkboxes - GPU
        int gy = cy + 6;
        AddSectionLabel("GPU", col1, cy, 600); gy = cy + 22;
        chkFlashAttn = AddCheckBox("Flashinfer attention", col1, ref gy, cw, false);
        chkTritonAttn = AddCheckBox("Triton attention", col1, ref gy, cw, true);

        // ===== Buttons =====
        int by = Math.Max(gy, cy2) + 10;
        btnStart = AddButton("Start Server", col1, by, 140);
        btnStart.ForeColor = Color.Green;
        btnStart.Font = new Font("Segoe UI", 10, FontStyle.Bold);
        btnStart.Click += (s, e) => ToggleServer();

        AddStatusLabel("Status: Stopped", col1 + 155, by + 4, 400, Color.Gray);

        // ===== Log =====
        int ly = by + 40;
        AddSectionLabel("Server Log", col1, ly, 600); ly += 22;
        txtLog = new RichTextBox();
        txtLog.Location = new Point(col1, ly);
        txtLog.Size = new Size(870, 280);
        txtLog.ReadOnly = true;
        txtLog.BackColor = Color.Black;
        txtLog.ForeColor = Color.Lime;
        txtLog.Font = new Font("Consolas", 9);
        txtLog.WordWrap = false;
        txtLog.Anchor = AnchorStyles.Left | AnchorStyles.Top | AnchorStyles.Right | AnchorStyles.Bottom;
        scrollPanel.Controls.Add(txtLog);
    }

    // ===== Helpers =====
    private TextBox MkTb(string def) { return new TextBox { Text = def, Size = new Size(80, 22) }; }

    private void AddSectionLabel(string text, int x, int y, int w)
    {
        Label l = new Label();
        l.Text = text; l.Location = new Point(x, y);
        l.Size = new Size(w, 20); l.Font = new Font("Segoe UI", 9, FontStyle.Bold);
        l.ForeColor = Color.DarkBlue;
        scrollPanel.Controls.Add(l);
    }

    private void AddLabel(string text, int x, int y, int w)
    {
        Label l = new Label();
        l.Text = text; l.Location = new Point(x, y);
        l.Size = new Size(w, 22); l.TextAlign = ContentAlignment.MiddleLeft;
        scrollPanel.Controls.Add(l);
    }

    private void AddField(int col, ref int y, string label, TextBox tb)
    {
        Label l = new Label();
        l.Text = label; l.Location = new Point(col, y);
        l.Size = new Size(130, 22); l.TextAlign = ContentAlignment.MiddleLeft;
        l.Font = new Font("Segoe UI", 9);
        scrollPanel.Controls.Add(l);
        tb.Location = new Point(col + 132, y - 1);
        scrollPanel.Controls.Add(tb);
        y += 26;
    }

    private TextBox AddTextBox(int x, int y, int w)
    {
        TextBox tb = new TextBox();
        tb.Location = new Point(x, y); tb.Size = new Size(w, 22);
        scrollPanel.Controls.Add(tb);
        return tb;
    }

    private Button AddButton(string text, int x, int y, int w)
    {
        Button btn = new Button();
        btn.Text = text; btn.Location = new Point(x, y);
        btn.Size = new Size(w, 28);
        scrollPanel.Controls.Add(btn);
        return btn;
    }

    private CheckBox AddCheckBox(string text, int x, ref int y, int w, bool def)
    {
        CheckBox cb = new CheckBox();
        cb.Text = text; cb.Location = new Point(x, y);
        cb.Size = new Size(w, 22); cb.Checked = def;
        scrollPanel.Controls.Add(cb);
        y += 24;
        return cb;
    }

    private Label AddStatusLabel(string text, int x, int y, int w, Color c)
    {
        Label l = new Label();
        l.Text = text; l.Location = new Point(x, y);
        l.Size = new Size(w, 24); l.ForeColor = c;
        l.Tag = "status";
        scrollPanel.Controls.Add(l);
        return l;
    }

    private void SetStatus(string text)
    {
        foreach (Control c in scrollPanel.Controls)
            if (c is Label && c.Tag != null && c.Tag.ToString() == "status")
                c.Text = "Status: " + text;
    }

    private void BrowseModel()
    {
        FolderBrowserDialog dlg = new FolderBrowserDialog();
        dlg.Description = "Select model folder";
        if (dlg.ShowDialog() == DialogResult.OK)
            txtModel.Text = dlg.SelectedPath;
    }

    private void ToggleServer()
    {
        if (serverRunning) StopServer(); else StartServer();
    }

    private void StartServer()
    {
        if (string.IsNullOrEmpty(txtModel.Text) || !Directory.Exists(txtModel.Text))
        {
            MessageBox.Show("Select a valid model folder.", "vLLM", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            return;
        }
        if (repoRoot == null)
        {
            MessageBox.Show("vLLM source not found. Clone the repo first.", "vLLM", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return;
        }

        // --- Pre-flight checks ---
        string pydPath = Path.Combine(repoRoot, "vllm", "_C.pyd");
        if (!File.Exists(pydPath))
        {
            Log("WARNING: _C.pyd not found at " + pydPath);
            Log("Run setup.bat first or copy _C.pyd into the vllm folder.");
        }
        else
        {
            Log("_C.pyd: OK (" + (new FileInfo(pydPath).Length / 1048576) + " MB)");
        }

        // Find Python — prefer venv Python over system Python
        string pythonExe = FindPythonExe(repoRoot);
        if (pythonExe == null)
        {
            Log("ERROR: No Python found. Install Python 3.12 and create a venv.");
            SetStatus("Error");
            return;
        }
        Log("Python: " + pythonExe);

        List<string> args = new List<string>();
        args.Add("-m"); args.Add("vllm.entrypoints.openai.api_server");
        args.Add("--model"); args.Add("\"" + txtModel.Text + "\"");
        args.Add("--port"); args.Add(txtPort.Text);
        args.Add("--max-model-len"); args.Add(txtMaxModelLen.Text);
        args.Add("--gpu-memory-utilization"); args.Add(txtGpuMem.Text);
        args.Add("--dtype"); args.Add("float16");
        args.Add("--seed"); args.Add(txtSeed.Text);

        if (tvCacheDtype.Text != "auto") { args.Add("--kv-cache-dtype"); args.Add(tvCacheDtype.Text); }
        if (txtTpSize.Text != "1") { args.Add("--tensor-parallel-size"); args.Add(txtTpSize.Text); }
        if (txtCpuOffload.Text != "0") { args.Add("--cpu-offload-gb"); args.Add(txtCpuOffload.Text); }
        if (txtBlockSize.Text != "16") { args.Add("--block-size"); args.Add(txtBlockSize.Text); }
        if (txtMaxBatch.Text != "8192") { args.Add("--max-num-batched-tokens"); args.Add(txtMaxBatch.Text); }

        if (chkEager.Checked) args.Add("--enforce-eager");
        if (chkTrust.Checked) args.Add("--trust-remote-code");
        if (chkPrefixCache.Checked) args.Add("--enable-prefix-caching");
        if (chkChunked.Checked) args.Add("--enable-chunked-prefill");
        if (chkNoStats.Checked) args.Add("--disable-log-stats");
        if (chkNoCustomReduce.Checked) args.Add("--disable-custom-all-reduce");
        if (chkLora.Checked) args.Add("--enable-lora");
        if (chkReasoning.Checked) args.Add("--enable-reasoning");
        if (chkHidden.Checked) args.Add("--return-hidden-states");
        if (chkFlashAttn.Checked) args.Add("--flashinfer-attention");
        if (chkTritonAttn.Checked) args.Add("--triton-attention");

        // Sampling params get passed to the server
        // (the server uses defaults unless changed)

        string argStr = string.Join(" ", args.ToArray());
        Log("Starting vLLM server...");
        Log("Model: " + txtModel.Text);
        Log("Port: " + txtPort.Text);
        Log("");

        try
        {
            // Show model info
        if (File.Exists(Path.Combine(txtModel.Text, "config.json")))
            Log("Model config: " + Path.Combine(txtModel.Text, "config.json"));
        else
            Log("WARNING: No config.json found in model path. Select a model folder (e.g. Qwen2.5-3B-Instruct), not a model directory.");

        // Set environment
            string[] rocmPaths = {
                @"C:\Program Files\AMD\ROCm\7.13",
                @"E:\ROCM-7.13.0-Windows",
                @"C:\ROCm\7.13"
            };
            foreach (string rp in rocmPaths)
                if (File.Exists(Path.Combine(rp, "bin", "hipcc.exe")))
                    Environment.SetEnvironmentVariable("HIP_PATH", rp);
            Environment.SetEnvironmentVariable("PYTHONPATH", repoRoot);
            Environment.SetEnvironmentVariable("VLLM_NO_USAGE_STATS", "true");

            serverProcess = new Process();
            serverProcess.StartInfo.FileName = pythonExe;
            serverProcess.StartInfo.Arguments = argStr;
            serverProcess.StartInfo.WorkingDirectory = repoRoot;
            serverProcess.StartInfo.UseShellExecute = false;
            serverProcess.StartInfo.RedirectStandardOutput = true;
            serverProcess.StartInfo.RedirectStandardError = true;
            serverProcess.StartInfo.CreateNoWindow = true;
            serverProcess.EnableRaisingEvents = true;
            serverProcess.Exited += (s, e) => {
                serverRunning = false;
                try {
                    int code = serverProcess.ExitCode;
                    this.BeginInvoke(new Action(() => {
                        Log("Server exited with code: " + code + (code != 0 ? " (ERROR)" : ""));
                        btnStart.Text = "Start Server";
                        btnStart.ForeColor = Color.Green;
                        SetStatus("Stopped");
                    }));
                } catch {}
            };

            try { serverProcess.Start(); }
            catch (Exception ex) {
                Log("ERROR starting process: " + ex.Message);
                Log("Command: python " + argStr);
                SetStatus("Error");
                btnStart.Text = "Start Server";
                btnStart.ForeColor = Color.Green;
                return;
            }
            serverRunning = true;
            outputThread = new Thread(ReadOutput);
            outputThread.IsBackground = true;
            outputThread.Start();

            btnStart.Text = "Stop Server";
            btnStart.ForeColor = Color.Red;
            SetStatus("Running");
        }
        catch (Exception ex) { Log("ERROR: " + ex.Message); SetStatus("Error"); }
    }

    private void StopServer()
    {
        if (serverProcess != null && !serverProcess.HasExited)
        {
            try { serverProcess.Kill(); } catch {}
            serverProcess.WaitForExit(2000);
        }
        serverRunning = false;
        btnStart.Text = "Start Server";
        btnStart.ForeColor = Color.Green;
        SetStatus("Stopped");
        Log("Server stopped.");
    }

    private void ReadOutput()
    {
        try
        {
            while (serverProcess != null && !serverProcess.HasExited)
            {
                string line = serverProcess.StandardOutput.ReadLine();
                if (line != null)
                    this.BeginInvoke(new Action<string>(Log), new object[] { line });
            }
            string rest = serverProcess.StandardOutput.ReadToEnd();
            if (!string.IsNullOrEmpty(rest))
                this.BeginInvoke(new Action<string>(Log), new object[] { rest });
        }
        catch { }
    }

    private void Log(string msg)
    {
        txtLog.AppendText(msg + "\n");
        txtLog.SelectionStart = txtLog.Text.Length;
        txtLog.ScrollToCaret();
    }

    private string FindVllmRepo(string startDir)
    {
        DirectoryInfo dir = new DirectoryInfo(startDir);
        for (int i = 0; i < 5; i++)
        {
            if (dir == null) break;
            if (File.Exists(Path.Combine(dir.FullName, "vllm", "__init__.py")))
                return dir.FullName;
            dir = dir.Parent;
        }
        return null;
    }

    private string FindPythonExe(string repoRoot)
    {
        // 1. Look for venv Python in various common locations
        string[] venvChecks = new string[] {
            Path.Combine(repoRoot, ".venv", "Scripts", "python.exe"),
            Path.Combine(repoRoot, "venv", "Scripts", "python.exe"),
            Path.Combine(repoRoot, ".venv", "bin", "python.exe"),
            Path.Combine(repoRoot, "..", ".venv", "Scripts", "python.exe"),
        };
        foreach (string v in venvChecks)
            if (File.Exists(v)) return v;

        // 2. Try "python" on PATH
        try
        {
            Process p = new Process();
            p.StartInfo.FileName = "python";
            p.StartInfo.Arguments = "--version";
            p.StartInfo.UseShellExecute = false;
            p.StartInfo.RedirectStandardOutput = true;
            p.StartInfo.CreateNoWindow = true;
            p.Start();
            string ver = p.StandardOutput.ReadToEnd().Trim();
            p.WaitForExit(2000);
            if (p.ExitCode == 0 && ver.Contains("3.12"))
                return "python";
        }
        catch {}

        // 3. Try common Python 3.12 install paths
        string[] fixedPaths = new string[] {
            @"C:\Users\rr\AppData\Local\Programs\Python\Python312\python.exe",
            @"C:\Python312\python.exe",
            @"C:\Program Files\Python312\python.exe",
        };
        foreach (string f in fixedPaths)
            if (File.Exists(f)) return f;

        return null;
    }
}
