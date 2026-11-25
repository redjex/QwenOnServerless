// Diagnostic invocation through the authenticated CLI; not deployed as an endpoint.
import { qwenChat } from "../lib/qwen-chat.js";
import { saveProgress } from "../lib/qwen-progress.js";

export default async function ({ prompt = "Привет!", maxTokens = 1 } = {}) {
  const result = await qwenChat(prompt, { maxTokens, trace: async partial => {
    if (partial.generated % 4 === 0) {
      await saveProgress("diagnostic", prompt, partial);
      console.log(partial);
    }
  } });
  await saveProgress("diagnostic", prompt, result, result.finished ? "finished" : "length_limit");
  return result;
}
