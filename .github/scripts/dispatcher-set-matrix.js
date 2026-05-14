const fs = require("fs");

/**
 * Parse one models CSV token: "model_id" or "model_id:tpN" (case-insensitive tp).
 * Returns { base, explicitTp } where explicitTp is null if absent.
 */
function parseModelToken(core, token) {
  const trimmed = token.trim();
  if (!trimmed) {
    return null;
  }
  const validSuffix = /^(.+):tp(\d+)$/i;
  const m = trimmed.match(validSuffix);
  if (m) {
    const base = m[1].trim();
    const n = parseInt(m[2], 10);
    if (!base || !Number.isInteger(n) || n <= 0) {
      core.setFailed(
        `Invalid model token '${trimmed}': :tpN requires a positive integer`,
      );
      return null;
    }
    return { base, explicitTp: n };
  }
  if (/:tp/i.test(trimmed)) {
    core.setFailed(
      `Invalid model token '${trimmed}': use suffix :tpN with a positive integer at the end of the token (e.g. model:tp2)`,
    );
    return null;
  }
  return { base: trimmed, explicitTp: null };
}

/**
 * Decide whether to use per-model recommended_tp or a single explicit TP from suffixes.
 * Returns { useRecommendedTp, requestedTp } or null if validation failed (setFailed called).
 */
function resolveTpMode(core, parsedEntries) {
  const explicitVals = [
    ...new Set(
      parsedEntries.map((e) => e.explicitTp).filter((x) => x != null),
    ),
  ];

  if (explicitVals.length === 0) {
    return { useRecommendedTp: true, requestedTp: null };
  }

  if (explicitVals.length > 1) {
    core.setFailed(
      `Conflicting tensor parallel suffixes in models list: ${explicitVals.join(", ")}. Use a single :tpN for all entries, or omit suffixes to use per-model recommended_tp.`,
    );
    return null;
  }

  const requestedTp = explicitVals[0];
  const anyMissing = parsedEntries.some((e) => e.explicitTp == null);
  if (anyMissing) {
    core.setFailed(
      `Mixed model list: when using :tpN, every comma-separated model entry must use the same suffix (e.g. all ...:tp${requestedTp})`,
    );
    return null;
  }

  return { useRecommendedTp: false, requestedTp };
}

function isPublicHfModel(model) {
  return model.includes("/");
}

function setMatrixOutput({ core, inputs }) {
  const inp = inputs;

  // Load <test, runner_label> mapping for the specified GPU type.
  const MAP_PATH = "tests/cohere/configs/runner_map.json";
  const runner_map = JSON.parse(fs.readFileSync(MAP_PATH, "utf8"));

  // Load TP size to model mapping
  const TP_MODEL_MAP_PATH = "tests/cohere/configs/tp_model_map.json";
  const tp_model_map = JSON.parse(fs.readFileSync(TP_MODEL_MAP_PATH, "utf8"));

  const device = inp.device.toString().trim();

  const table = runner_map[device];
  if (!table) {
    core.setFailed(`No runner mapping for device='${device}' in ${MAP_PATH}`);
    return;
  }

  // Parse input models (optional :tpN suffix per entry)
  const models_input = inp.models.toString().trim();
  const rawTokens = models_input
    .split(",")
    .map((t) => t.trim())
    .filter(Boolean);

  const parsedEntries = [];
  for (const t of rawTokens) {
    const p = parseModelToken(core, t);
    if (p === null) {
      return;
    }
    parsedEntries.push(p);
  }

  const tpMode = resolveTpMode(core, parsedEntries);
  if (tpMode === null) {
    return;
  }
  const { useRecommendedTp, requestedTp } = tpMode;
  const features = inp.features.toString().trim();
  const benchmarks = inp.benchmarks.toString().trim();
  const runPerf100 = benchmarks === "perf_100";
  const runPerf1000 = benchmarks === "perf_1000";
  const runLmEval = benchmarks === "lm_eval";
  const runBeeEval = benchmarks === "bee_eval";
  const performanceOnlyBenchmarks = runPerf100 || runPerf1000;

  const include = [];

  // Determine which feature tests to run based on features input
  const featureTests = [];
  if (features === "cpu_check" || features === "all") {
    featureTests.push("cpu_check");
  }
  if (features === "fast_check") {
    featureTests.push("fast_check");
  }
  if (features === "model_arch" || features === "all") {
    featureTests.push("model_arch_reward");
    featureTests.push("model_arch_c5_3a30t");
    featureTests.push("model_arch_c5_lora");
  }
  if (features === "quantization" || features === "all") {
    featureTests.push("quantization_32bit_logits");
  }
  if (features === "GG" || features === "all") {
    featureTests.push("guided_generation");
  }
  if (features === "thinking_budget" || features === "all") {
    featureTests.push("thinking_budget");
    featureTests.push("bee_sample_tb_check");
  }
  if (features === "speculative_decoding" || features === "all") {
    featureTests.push("speculative_decoding");
  }
  if (features === "vision" || features === "all") {
    featureTests.push("vision");
  }
  // Handle feature tests
  for (const tg of featureTests) {
    const labels = table[tg];
    if (!labels) {
      core.warning(
        `No runner labels for device='${device}', test_group='${tg}' in ${MAP_PATH}. Skipping this test group.`,
      );
      continue;
    }

    include.push({
      test_group: tg,
      runner_labels: labels,
      device,
      tp_size: 0,
      models: "",
      hardware_profiles_override: inp.hardware_profiles_override,
    });
  }

  // Handle test groups with TP size support
  const device_config = tp_model_map[device];
  if (!device_config) {
    core.warning(
      `No model mapping found for device='${device}' in ${TP_MODEL_MAP_PATH}. Skipping.`,
    );
  } else {
    const modelsByTp = new Map();

    for (const { base: model } of parsedEntries) {
      const modelConfig = device_config[model];
      if (!modelConfig) {
        if (isPublicHfModel(model)) {
          if (!performanceOnlyBenchmarks) {
            core.setFailed(
              `Public Hugging Face model '${model}' is only supported for performance benchmarks. Use build-and-bench with an explicit :tpN suffix.`,
            );
            return;
          }
          if (useRecommendedTp) {
            core.setFailed(
              `Public Hugging Face model '${model}' is not present in ${TP_MODEL_MAP_PATH}. Specify an explicit tensor parallel suffix such as '${model}:tp1'.`,
            );
            return;
          }
          if (!modelsByTp.has(requestedTp)) {
            modelsByTp.set(requestedTp, []);
          }
          modelsByTp.get(requestedTp).push(model);
          continue;
        }
        core.warning(
          `No TP config found for device='${device}', model='${model}' in ${TP_MODEL_MAP_PATH}. Skipping this model.`,
        );
        continue;
      }

      const minimumTp = Number(modelConfig.minimum_tp);
      const recommendedTp = Number(modelConfig.recommended_tp);
      if (
        !Number.isInteger(minimumTp) ||
        !Number.isInteger(recommendedTp) ||
        minimumTp <= 0 ||
        recommendedTp <= 0 ||
        recommendedTp < minimumTp
      ) {
        core.setFailed(
          `Invalid TP config for device='${device}', model='${model}' in ${TP_MODEL_MAP_PATH}: minimum_tp=${modelConfig.minimum_tp}, recommended_tp=${modelConfig.recommended_tp}`,
        );
        return;
      }

      const selectedTp = useRecommendedTp ? recommendedTp : requestedTp;
      if (!useRecommendedTp && selectedTp < minimumTp) {
        core.setFailed(
          `Invalid tensor parallel size '${selectedTp}' for model='${model}' on device='${device}'. Minimum supported tp is ${minimumTp}.`,
        );
        return;
      }

      if (!modelsByTp.has(selectedTp)) {
        modelsByTp.set(selectedTp, []);
      }
      modelsByTp.get(selectedTp).push(model);
    }

    for (const [tp_size, modelsForTp] of modelsByTp.entries()) {
      const models = modelsForTp.join(",");

      // Determine which benchmark/eval tests to run based on benchmarks input
      // Handle performance with TP size
      if (runPerf100) {
        const tg = "performance";
        const labels = table[`performance_tp${tp_size}`];
        if (!labels) {
          core.warning(
            `No runner labels for device='${device}', test_group='${tg}', tp_size='${tp_size}' in ${MAP_PATH}. Skipping this test group.`,
          );
        } else {
          include.push({
            test_group: tg,
            runner_labels: labels,
            tp_size,
            models,
            device,
            benchmark_output_len: "100",
            hardware_profiles_override: inp.hardware_profiles_override,
          });
        }
      }
      if (runPerf1000) {
        const tg = "performance";
        const labels = table[`performance_tp${tp_size}`];
        if (!labels) {
          core.warning(
            `No runner labels for device='${device}', test_group='${tg}', tp_size='${tp_size}' in ${MAP_PATH}. Skipping this test group.`,
          );
        } else {
          include.push({
            test_group: tg,
            runner_labels: labels,
            tp_size,
            models,
            device,
            benchmark_output_len: "1000",
            hardware_profiles_override: inp.hardware_profiles_override,
          });
        }
      }

      // Handle lm_eval with TP size
      if (runLmEval) {
        const tg = "lm_eval";
        const labels = table[`eval_tp${tp_size}`];
        if (!labels) {
          core.warning(
            `No runner labels for device='${device}', test_group='${tg}', tp_size='${tp_size}' in ${MAP_PATH}. Skipping this test group.`,
          );
          continue;
        }
        include.push({
          test_group: tg,
          runner_labels: labels,
          tp_size,
          models,
          device,
          hardware_profiles_override: inp.hardware_profiles_override,
        });
      }

      // Handle bee_eval with TP size — one matrix entry per model so they
      // run on separate runners in parallel (bee_eval is multi-hour per model).
      if (runBeeEval) {
        const tg = "bee_eval";
        const labels = table[`eval_tp${tp_size}`];
        if (!labels) {
          core.warning(
            `No runner labels for device='${device}', test_group='${tg}', tp_size='${tp_size}' in ${MAP_PATH}. Skipping this test group.`,
          );
        } else {
          for (const singleModel of modelsForTp) {
            include.push({
              test_group: tg,
              runner_labels: labels,
              tp_size,
              models: singleModel,
              device,
              hardware_profiles_override: inp.hardware_profiles_override,
            });
          }
        }
      }
    }
  }

  const matrix = { include };
  core.info(`matrix=${JSON.stringify(matrix)}`);
  core.setOutput("matrix", JSON.stringify(matrix));
}

module.exports = {
  setMatrixOutput,
};
