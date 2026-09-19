// Official Typesafe TS SDK pointed at (a) our local jev-compatible server and
// (b) the real Jev API. Smoke-tests one request: all question IDs answered,
// selected field types correct, Choice probabilities normalized. This does not
// establish full API compatibility, equivalent judgments, or calibration.
//
//   node test.ts                       (local server on :8399 must be running)
//   node test.ts --with-jev            (also hits the real API; key from ../../.env)

import { readFileSync } from "node:fs";
import * as sdk from "@typesafe-ai/sdk";

const LOCAL_URL = process.env.LOCAL_JEV_URL ?? "http://127.0.0.1:8399";

function jevKey(): string | undefined {
  if (process.env.TYPESAFE_API_KEY) return process.env.TYPESAFE_API_KEY;
  try {
    const env = readFileSync(new URL("../../.env", import.meta.url), "utf8");
    const line = env.split("\n").find((l) => l.startsWith("TYPESAFE_API_KEY="));
    return line?.split("=", 2)[1]?.trim().replace(/^['"]|['"]$/g, "");
  } catch {
    return undefined;
  }
}

// helpers if the SDK exports them, raw question objects otherwise
const noul =
  (sdk as any).noul ??
  ((instructions: string, criteria?: object) => ({ type: "noul", instructions, criteria }));
const choice =
  (sdk as any).choice ??
  ((instructions: string, criteria: object) => ({ type: "choice", instructions, criteria }));
const score =
  (sdk as any).score ??
  ((instructions: string, criteria: string[]) => ({ type: "score", instructions, criteria }));

const request = {
  state: {
    ticket_message:
      "I've been charged twice for my flight to Berlin and nobody is answering " +
      "the phone. I want my money back immediately or I am disputing this with my bank.",
  },
  questions: {
    refund_requested: noul("Does ticket_message request a refund?"),
    request_type: choice("What is the main request in ticket_message?", {
      refund: "The customer wants money returned.",
      rebooking: "The customer wants a replacement flight.",
      information: "The customer wants information only.",
    }),
    frustration: score("How frustrated is the customer?", [
      "Calm; neutral tone, no complaints.",
      "Annoyed; complains but remains cooperative.",
      "Angry; threats, ultimatums, or escalation demands.",
    ]),
  },
};

function check(name: string, response: any) {
  const answers = response.answers;
  const missing = Object.keys(request.questions).filter((q) => !(q in answers));
  if (missing.length) throw new Error(`${name}: missing answers: ${missing}`);

  const { refund_requested, request_type, frustration } = answers;
  if (typeof refund_requested.noul !== "number") throw new Error(`${name}: bad noul`);
  if (typeof request_type.choice !== "string") throw new Error(`${name}: bad choice`);
  const sum = (Object.values(request_type.probabilities) as number[]).reduce((a, b) => a + b, 0);
  if (Math.abs(sum - 1) > 1e-3) throw new Error(`${name}: probabilities sum ${sum}`);
  if (typeof frustration.score !== "number" || !frustration.legend)
    throw new Error(`${name}: bad score`);

  console.log(`\n== ${name} ==`);
  console.log(`refund_requested.noul  ${refund_requested.noul.toFixed(3)}`);
  console.log(
    `request_type.choice    ${request_type.choice} ` +
      `(conf ${request_type.confidence.toFixed(3)})`
  );
  console.log(
    `frustration.score      ${frustration.score.toFixed(2)} ` +
      `(conf ${frustration.confidence.toFixed(3)})`
  );
}

const local = new (sdk as any).TypeSafeClient({ apiKey: "local", baseURL: LOCAL_URL });
check("local (ours)", await local.systemOne(request));

if (process.argv.includes("--with-jev")) {
  const key = jevKey();
  if (!key) throw new Error("no TYPESAFE_API_KEY found");
  const jev = new (sdk as any).TypeSafeClient({ apiKey: key });
  check("jev (real API)", await jev.systemOne(request));
}

console.log("\nall checks passed");
