import test from "node:test";
import assert from "node:assert/strict";
import {
  createModel, encode, forward, lossAndGradient, workspace, train,
  serialize, deserialize, random, memoryPlan,
} from "../tgcloud/lib/model.js";
import probe from "../tgcloud/endpoints/memoryProbe.js";

test("backpropagation matches finite differences, including repeated characters", () => {
  const model = createModel();
  const work = workspace();
  const context = encode("кот кот ");
  const target = encode("с")[0];
  lossAndGradient(model, context, target, work);
  const gradient = work.gradient.slice();
  const selected = [...gradient.keys()].filter(i => Math.abs(gradient[i]) > 0.001);
  for (let n = 0; n < 40; n++) {
    const i = selected[Math.floor(n * selected.length / 40)];
    const old = model.weights[i];
    const eps = 0.003;
    model.weights[i] = old + eps;
    const plus = -Math.log(forward(model, context, work)[target]);
    model.weights[i] = old - eps;
    const minus = -Math.log(forward(model, context, work)[target]);
    model.weights[i] = old;
    assert.ok(Math.abs((plus - minus) / (2 * eps) - gradient[i]) < 0.0003, `gradient ${i}`);
  }
});

test("learns a repeated corpus and resumes exactly from its checkpoint", () => {
  const model = createModel();
  const corpus = "кот спит. кот ест. ".repeat(20);
  const first = train(model, corpus, { steps: 128, budgetMs: Infinity, rng: random(1) });
  train(model, corpus, { steps: 1800, budgetMs: Infinity, rng: random(2) });
  const last = train(model, corpus, { steps: 128, budgetMs: Infinity, rng: random(3) });
  assert.ok(last.meanLoss < first.meanLoss * 0.3);
  const restored = deserialize(serialize(model));
  assert.deepEqual(restored, model);
  train(model, corpus, { steps: 10, budgetMs: Infinity, rng: random(4) });
  train(restored, corpus, { steps: 10, budgetMs: Infinity, rng: random(4) });
  assert.deepEqual(restored, model);
  assert.ok(memoryPlan().trainingArraysBytes < 160 * 1024);
});

test("rejects corrupt checkpoints and tests both memory allocation paths", () => {
  assert.throws(() => deserialize('{"format":0}'));
  assert.equal(probe({ mib: 1 }).success, true);
  assert.equal(probe({ mib: 1, kind: "heap" }).success, true);
  assert.throws(() => probe({ mib: 257 }));
});
