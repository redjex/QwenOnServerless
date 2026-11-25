import { createModel, train, generate, random, serialize, memoryPlan } from "./tgcloud/lib/model.js";

const model = createModel();
const corpus = "кот спит. кот ест. кот спит. кот ест. ".repeat(30);
const rng = random(7);
let first;
let last;
for (let i = 0; i < 40; i++) {
  last = train(model, corpus, { steps: 128, budgetMs: Infinity, rng });
  first ??= last;
}
console.log(JSON.stringify({
  ...memoryPlan(), steps: model.steps,
  firstTrainingLoss: first.meanLoss, lastTrainingLoss: last.meanLoss,
  checkpointUtf8Bytes: Buffer.byteLength(serialize(model)),
  sample: generate(model, "кот ", { rng: random(9), budgetMs: Infinity }),
}, null, 2));
