// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { test } = require('node:test');
const { script, families } = JSON.parse(readFileSync(0, 'utf8'));
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const labelIssue = new AsyncFunction('github', 'context', 'core', script);

async function run(title, body = '', labels = []) {
  const added = [];
  await labelIssue(
    { rest: { issues: { addLabels: async (params) => added.push(...params.labels) } } },
    { repo: { owner: 'vllm-project', repo: 'vllm' }, issue: { number: 1 }, payload: { issue: { title, body, labels } } },
    { notice() {}, setOutput() {} },
  );
  return added;
}

for (const [family, examples] of Object.entries(families)) {
  for (const title of examples.titles || []) {
    test(`${family}: ${title}`, async () => {
      assert.ok((await run(title)).includes(family), `missing ${family}`);
    });
  }
  for (const title of examples.not_titles || []) {
    test(`exclude ${family}: ${title}`, async () => {
      assert.ok(!(await run(title)).includes(family), `unexpected ${family}`);
    });
  }
}

test('pasted model names in bodies do not set model labels', async () => {
  const body = Object.values(families).flatMap(example => example.titles).join('\n');
  const added = await run('Documentation link is broken', body);
  for (const family of Object.keys(families)) assert.ok(!added.includes(family), `unexpected ${family}`);
});

test('matching preserves existing manual labels', async () => {
  assert.deepEqual(await run('Qwen99 model support', '', [{ name: 'qwen' }, { name: 'manual-label' }]), []);
});
