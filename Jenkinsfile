// vLLM CI/CD Pipeline for DGX Spark 2 (SM121/GB10)
// ================================================
// Builds and tests vLLM on spark2 (ARM64 + 8x A100)
// Jenkins master on node5 SSHs to spark2 for execution
//
// Target: NVIDIA GB10 Blackwell (SM121) + ARM64 + CUDA 13.1
// Model: Qwen3-Next-80B-A3B-FP8

pipeline {
    agent any
    
    options {
        timestamps()
        ansiColor('xterm')
        buildDiscarder(logRotator(numToKeepStr: '30', artifactNumToKeepStr: '10'))
        timeout(time: 180, unit: 'MINUTES')
        disableConcurrentBuilds()
    }
    
    environment {
        // Spark2 connection (ARM64 + GPU build server)
        SPARK2_HOST = '192.168.4.208'
        SPARK2_USER = 'seli'
        
        // vLLM server endpoint
        VLLM_API_URL = 'http://192.168.4.208:8000'
        
        // Remote paths on spark2
        VLLM_DIR = '/data/vllm'
        VLLM_ENV = '/data/vllm-env'
        CONTAINER_DIR = '/data/vllm-container'
        
        // Local paths for results
        RESULTS_DIR = 'test-results'
        ALLURE_RESULTS = 'test-results/allure-results'
        METRICS_DIR = 'test-results/metrics'
        
        // Build settings
        TORCH_CUDA_ARCH_LIST = '12.1'
        MAX_JOBS = '20'
    }
    
    parameters {
        booleanParam(
            name: 'REBUILD_VLLM',
            defaultValue: false,
            description: 'Rebuild vLLM from source (preserves torch)'
        )
        booleanParam(
            name: 'REBUILD_FLASHINFER',
            defaultValue: false,
            description: 'Rebuild FlashInfer from source'
        )
        booleanParam(
            name: 'REBUILD_DOCKER',
            defaultValue: false,
            description: 'Rebuild Docker container image'
        )
        booleanParam(
            name: 'RUN_BENCHMARKS',
            defaultValue: true,
            description: 'Run performance benchmarks'
        )
        booleanParam(
            name: 'SYNC_CODE',
            defaultValue: true,
            description: 'Pull latest code on spark2 before build'
        )
        string(
            name: 'BENCHMARK_PROMPTS',
            defaultValue: '20',
            description: 'Number of prompts for benchmark'
        )
    }
    
    stages {
        stage('Prepare') {
            steps {
                echo '📁 Preparing test environment...'
                sh """
                    mkdir -p ${RESULTS_DIR}
                    mkdir -p ${ALLURE_RESULTS}
                    mkdir -p ${METRICS_DIR}
                    rm -f ${RESULTS_DIR}/*.xml ${RESULTS_DIR}/*.html 2>/dev/null || true
                    rm -rf ${ALLURE_RESULTS}/* 2>/dev/null || true
                    rm -f ${METRICS_DIR}/*.csv 2>/dev/null || true
                """
            }
        }
        
        stage('Spark2 Health Check') {
            steps {
                echo '🔍 Checking spark2 connectivity and GPU status...'
                sshagent(['ssh-credentials']) {
                    sh '''
                        ssh -o StrictHostKeyChecking=no ${SPARK2_USER}@${SPARK2_HOST} "
                            echo '=== System Info ==='
                            uname -a
                            echo ''
                            echo '=== GPU Status ==='
                            nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv
                            echo ''
                            echo '=== Docker Status ==='
                            docker ps --format 'table {{.Names}}\\t{{.Status}}' | head -10
                        "
                    '''
                }
            }
        }
        
        stage('vLLM Server Health') {
            steps {
                echo '🔍 Checking vLLM server status and collecting metrics...'
                script {
                    def health = sh(
                        script: "curl -sf ${VLLM_API_URL}/health && echo 'OK' || echo 'DOWN'",
                        returnStdout: true
                    ).trim()
                    
                    def models = sh(
                        script: "curl -sf ${VLLM_API_URL}/v1/models | jq -r '.data[0].id' 2>/dev/null || echo 'UNKNOWN'",
                        returnStdout: true
                    ).trim()
                    
                    // Collect GPU metrics for plotting
                    sshagent(['ssh-credentials']) {
                        sh '''
                            ssh -o StrictHostKeyChecking=no ${SPARK2_USER}@${SPARK2_HOST} "
                                nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
                            " > ${METRICS_DIR}/gpu_initial.csv || true
                        '''
                    }
                    
                    if (health.contains('OK')) {
                        echo "✅ vLLM Server: Running"
                        echo "📦 Loaded Model: ${models}"
                    } else {
                        echo "⚠️ vLLM Server: Not responding"
                    }
                }
            }
        }
        
        stage('Sync Code') {
            when {
                expression { params.SYNC_CODE == true }
            }
            steps {
                echo '📥 Syncing latest code to spark2...'
                sshagent(['ssh-credentials']) {
                    sh '''
                        ssh -o StrictHostKeyChecking=no ${SPARK2_USER}@${SPARK2_HOST} "
                            cd ${VLLM_DIR} && \
                            git fetch origin && \
                            git status && \
                            git pull origin \\$(git branch --show-current) && \
                            git log -1 --pretty=format:'%h - %s (%an, %ar)'
                        "
                    '''
                }
            }
        }
        
        stage('Verify Torch') {
            steps {
                echo '🔧 Verifying CUDA torch installation...'
                sshagent(['ssh-credentials']) {
                    sh '''
                        ssh -o StrictHostKeyChecking=no ${SPARK2_USER}@${SPARK2_HOST} "
                            source ${VLLM_ENV}/bin/activate && \
                            python3 -c \\"
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
\\"
                        "
                    '''
                }
            }
        }
        
        stage('Build FlashInfer') {
            when {
                expression { params.REBUILD_FLASHINFER == true }
            }
            steps {
                echo '🔨 Rebuilding FlashInfer on spark2...'
                sshagent(['ssh-credentials']) {
                    sh '''
                        ssh -o StrictHostKeyChecking=no ${SPARK2_USER}@${SPARK2_HOST} "
                            cd ${CONTAINER_DIR} && \
                            ./safe-rebuild-vllm.sh --flashinfer-only --clear-cache -y
                        "
                    '''
                }
            }
        }
        
        stage('Build vLLM') {
            when {
                expression { params.REBUILD_VLLM == true }
            }
            steps {
                echo '🔨 Rebuilding vLLM on spark2 (preserving torch)...'
                sshagent(['ssh-credentials']) {
                    sh '''
                        ssh -o StrictHostKeyChecking=no ${SPARK2_USER}@${SPARK2_HOST} "
                            cd ${CONTAINER_DIR} && \
                            ./safe-rebuild-vllm.sh --vllm-only -y
                        "
                    '''
                }
            }
        }
        
        stage('Build Docker') {
            when {
                expression { params.REBUILD_DOCKER == true }
            }
            steps {
                echo '🐳 Rebuilding Docker container on spark2...'
                sshagent(['ssh-credentials']) {
                    sh '''
                        ssh -o StrictHostKeyChecking=no ${SPARK2_USER}@${SPARK2_HOST} "
                            cd ${CONTAINER_DIR} && \
                            ./safe-rebuild-vllm.sh --sync --docker -y 2>&1 | tee build.log
                        "
                    '''
                }
            }
        }
        
        stage('API Health Tests') {
            steps {
                echo '🌐 Running API tests against live vLLM server...'
                script {
                    // Test health endpoint
                    def healthResult = sh(
                        script: "curl -sf ${VLLM_API_URL}/health && echo 'PASS' || echo 'FAIL'",
                        returnStdout: true
                    ).trim()
                    
                    // Test models endpoint
                    def modelsResult = sh(
                        script: "curl -sf ${VLLM_API_URL}/v1/models | jq -e '.data | length > 0' && echo 'PASS' || echo 'FAIL'",
                        returnStdout: true
                    ).trim()
                    
                    // Quick inference test with timing
                    def inferStart = System.currentTimeMillis()
                    def inferResult = sh(
                        script: '''
                            curl -sf -X POST "${VLLM_API_URL}/v1/chat/completions" \
                                -H "Content-Type: application/json" \
                                -d '{"model":"/models/Qwen3-Next-80B-A3B-FP8","messages":[{"role":"user","content":"Say hello in one word"}],"max_tokens":10}' \
                                | jq -e '.choices[0].message.content' && echo 'PASS' || echo 'FAIL'
                        ''',
                        returnStdout: true
                    ).trim()
                    def inferTime = System.currentTimeMillis() - inferStart
                    
                    echo "Health Check: ${healthResult.contains('PASS') ? '✅' : '❌'}"
                    echo "Models Check: ${modelsResult.contains('PASS') ? '✅' : '❌'}"
                    echo "Inference Check: ${inferResult.contains('PASS') ? '✅' : '❌'} (${inferTime}ms)"
                    
                    // Write API test metrics CSV for plotting
                    writeFile file: "${METRICS_DIR}/api_tests.csv", text: """test,result,latency_ms
health,${healthResult.contains('PASS') ? 1 : 0},0
models,${modelsResult.contains('PASS') ? 1 : 0},0
inference,${inferResult.contains('PASS') ? 1 : 0},${inferTime}
"""
                    
                    // Write JUnit result
                    def failures = [healthResult, modelsResult, inferResult].count { it.contains('FAIL') }
                    writeFile file: "${RESULTS_DIR}/api_results.xml", text: """<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="vLLM API Tests" tests="3" failures="${failures}" time="${inferTime/1000.0}">
    <testcase classname="api" name="health_check" time="0.1">${healthResult.contains('FAIL') ? '<failure message="Health check failed"/>' : ''}</testcase>
    <testcase classname="api" name="models_list" time="0.1">${modelsResult.contains('FAIL') ? '<failure message="Models list failed"/>' : ''}</testcase>
    <testcase classname="api" name="inference" time="${inferTime/1000.0}">${inferResult.contains('FAIL') ? '<failure message="Inference failed"/>' : ''}</testcase>
</testsuite>"""
                }
            }
            post {
                always {
                    junit(
                        testResults: "${RESULTS_DIR}/api_results.xml",
                        allowEmptyResults: true,
                        skipPublishingChecks: true
                    )
                }
            }
        }
        
        stage('Benchmarks') {
            when {
                anyOf {
                    expression { params.RUN_BENCHMARKS == true }
                    triggeredBy 'TimerTrigger'
                }
            }
            steps {
                echo "⚡ Running performance benchmarks (${params.BENCHMARK_PROMPTS} prompts)..."
                script {
                    // Run benchmark and capture detailed metrics
                    sshagent(['ssh-credentials']) {
                        sh '''
                            ssh -o StrictHostKeyChecking=no ${SPARK2_USER}@${SPARK2_HOST} "
                                # Warmup request
                                curl -sf -X POST '${VLLM_API_URL}/v1/chat/completions' \
                                    -H 'Content-Type: application/json' \
                                    -d '{\"model\":\"/models/Qwen3-Next-80B-A3B-FP8\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":5}' > /dev/null
                                
                                echo 'Warmup complete, starting benchmark...'
                            "
                        '''
                        
                        // Run multiple inference requests and collect timing
                        sh """
                            echo 'prompt_id,tokens,latency_ms,tokens_per_sec' > ${METRICS_DIR}/benchmark_detailed.csv
                            
                            for i in \$(seq 1 ${params.BENCHMARK_PROMPTS}); do
                                PROMPT="Write a haiku about number \$i"
                                START=\$(date +%s%3N)
                                
                                RESPONSE=\$(curl -sf -X POST "${VLLM_API_URL}/v1/chat/completions" \
                                    -H "Content-Type: application/json" \
                                    -d "{\\\"model\\\":\\\"/models/Qwen3-Next-80B-A3B-FP8\\\",\\\"messages\\\":[{\\\"role\\\":\\\"user\\\",\\\"content\\\":\\\"\$PROMPT\\\"}],\\\"max_tokens\\\":50}" 2>/dev/null)
                                
                                END=\$(date +%s%3N)
                                LATENCY=\$((END - START))
                                
                                TOKENS=\$(echo "\$RESPONSE" | jq -r '.usage.completion_tokens // 0' 2>/dev/null || echo "0")
                                if [ "\$LATENCY" -gt 0 ] && [ "\$TOKENS" -gt 0 ]; then
                                    TPS=\$(echo "scale=2; \$TOKENS * 1000 / \$LATENCY" | bc)
                                else
                                    TPS=0
                                fi
                                
                                echo "\$i,\$TOKENS,\$LATENCY,\$TPS" >> ${METRICS_DIR}/benchmark_detailed.csv
                                echo "Request \$i: \${TOKENS} tokens in \${LATENCY}ms (\${TPS} tok/s)"
                            done
                        """
                        
                        // Collect final GPU metrics
                        sh '''
                            ssh -o StrictHostKeyChecking=no ${SPARK2_USER}@${SPARK2_HOST} "
                                nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
                            " > ${METRICS_DIR}/gpu_final.csv || true
                        '''
                    }
                    
                    // Calculate summary statistics
                    sh '''
                        if [ -f ${METRICS_DIR}/benchmark_detailed.csv ]; then
                            # Calculate averages using awk
                            awk -F',' 'NR>1 {
                                sum_tokens+=$2; sum_latency+=$3; sum_tps+=$4; count++
                            } END {
                                if(count>0) {
                                    printf "avg_tokens,avg_latency_ms,avg_tokens_per_sec\\n"
                                    printf "%.1f,%.1f,%.2f\\n", sum_tokens/count, sum_latency/count, sum_tps/count
                                }
                            }' ${METRICS_DIR}/benchmark_detailed.csv > ${METRICS_DIR}/benchmark_summary.csv
                            
                            echo "=== Benchmark Summary ==="
                            cat ${METRICS_DIR}/benchmark_summary.csv
                        fi
                    '''
                }
            }
            post {
                always {
                    archiveArtifacts(
                        artifacts: "${METRICS_DIR}/*.csv",
                        allowEmptyArchive: true
                    )
                }
            }
        }
        
        stage('Plot Metrics') {
            steps {
                echo '📊 Generating performance plots...'
                script {
                    // Plot benchmark latency over requests
                    plot(
                        csvFileName: 'benchmark_latency.csv',
                        csvSeries: [[
                            file: "${METRICS_DIR}/benchmark_detailed.csv",
                            inclusionFlag: 'OFF',
                            displayTableFlag: false,
                            exclusionValues: 'prompt_id',
                            url: ''
                        ]],
                        group: 'vLLM Performance',
                        title: 'Inference Latency (ms)',
                        style: 'line',
                        yaxis: 'Latency (ms)',
                        numBuilds: '30',
                        useDescr: false
                    )
                    
                    // Plot tokens per second
                    plot(
                        csvFileName: 'benchmark_tps.csv',
                        csvSeries: [[
                            file: "${METRICS_DIR}/benchmark_detailed.csv",
                            inclusionFlag: 'OFF',
                            displayTableFlag: false,
                            exclusionValues: 'prompt_id,tokens,latency_ms',
                            url: ''
                        ]],
                        group: 'vLLM Performance',
                        title: 'Tokens per Second',
                        style: 'line',
                        yaxis: 'Tokens/sec',
                        numBuilds: '30',
                        useDescr: false
                    )
                }
            }
        }
        
        stage('Allure Report') {
            steps {
                echo '📈 Generating Allure report...'
                script {
                    try {
                        allure([
                            includeProperties: false,
                            jdk: '',
                            properties: [],
                            reportBuildPolicy: 'ALWAYS',
                            results: [[path: "${ALLURE_RESULTS}"]]
                        ])
                    } catch (e) {
                        echo "Allure report generation skipped: ${e.message}"
                    }
                }
            }
        }
    }
    
    post {
        always {
            echo '🧹 Archiving results...'
            archiveArtifacts(
                artifacts: "${RESULTS_DIR}/**/*",
                allowEmptyArchive: true
            )
        }
        success {
            echo '''
╔══════════════════════════════════════════════════════════════╗
║  ✅ vLLM Pipeline completed successfully!                    ║
║                                                              ║
║  Server: http://192.168.4.208:8000                          ║
║  Model:  Qwen3-Next-80B-A3B-FP8                             ║
║  GPU:    NVIDIA GB10 (SM121) - 115GB                        ║
╚══════════════════════════════════════════════════════════════╝
'''
        }
        failure {
            echo '❌ vLLM Pipeline failed! Check logs for details.'
        }
        unstable {
            echo '⚠️ vLLM Pipeline unstable (some tests failed)'
        }
    }
}
