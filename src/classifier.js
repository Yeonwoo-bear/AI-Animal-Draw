import { loadLiteRt, loadAndCompile, Tensor } from "@litertjs/core";
import { ANIMALS } from "./animals";

let modelPromise = null;

export async function loadAnimalModel() {
  if (modelPromise) return modelPromise;

  modelPromise = (async () => {
    await loadLiteRt(
      "https://cdn.jsdelivr.net/npm/@litertjs/core@2.5.3/wasm/",
      { jspi: true }
    );

    const model = await loadAndCompile("/model/animal_draw_v11.tflite", {
      accelerator: "wasm",
    });

    console.log("LiteRT input:", model.getInputDetails?.());
    console.log("LiteRT output:", model.getOutputDetails?.());
    return model;
  })().catch((error) => {
    modelPromise = null;
    throw error;
  });

  return modelPromise;
}

function render(strokes, { lineWidth = 5, padRatio = 0.18, scaleAdjust = 1 } = {}) {
  const pts = strokes.flat().filter(Boolean);
  if (!pts.length) return null;

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  for (const p of pts) {
    minX = Math.min(minX, p.x);
    minY = Math.min(minY, p.y);
    maxX = Math.max(maxX, p.x);
    maxY = Math.max(maxY, p.y);
  }

  const w = Math.max(1, maxX - minX);
  const h = Math.max(1, maxY - minY);
  const side = Math.max(w, h);
  const pad = side * padRatio;
  const scale = (52 / (side + 2 * pad)) * scaleAdjust;
  const dx = (64 - w * scale) / 2;
  const dy = (64 - h * scale) / 2;

  const canvas = document.createElement("canvas");
  canvas.width = 64;
  canvas.height = 64;

  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, 64, 64);
  ctx.strokeStyle = "#000";
  ctx.lineWidth = lineWidth;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  for (const s of strokes) {
    if (!s || s.length < 2) continue;

    ctx.beginPath();
    ctx.moveTo(dx + (s[0].x - minX) * scale, dy + (s[0].y - minY) * scale);

    for (let i = 1; i < s.length; i++) {
      ctx.lineTo(dx + (s[i].x - minX) * scale, dy + (s[i].y - minY) * scale);
    }

    ctx.stroke();
  }

  return canvas;
}

function toFloatInput(canvas) {
  const { data } = canvas
    .getContext("2d", { willReadFrequently: true })
    .getImageData(0, 0, 64, 64);

  const input = new Float32Array(64 * 64);

  for (let i = 0, j = 0; i < data.length; i += 4, j++) {
    const gray = (data[i] + data[i + 1] + data[i + 2]) / 3 / 255;
    // Training format: white stroke = 1, black background = 0
    input[j] = 1 - gray;
  }

  return input;
}

async function runOne(model, canvas) {
  const input = new Tensor(toFloatInput(canvas), [1, 64, 64, 1]);
  let results;

  try {
    results = await model.run(input);
    const first = results[0];
    const cpu = await first.moveTo("wasm");

    try {
      return Array.from(cpu.toTypedArray()).map(Number);
    } finally {
      cpu.delete();
    }
  } finally {
    input.delete();
    results?.delete?.();
  }
}

function normalizeProbabilities(values) {
  const safe = values.map((x) => Number.isFinite(x) ? x : 0);
  const sum = safe.reduce((a, b) => a + b, 0);

  // 이미 softmax 확률처럼 보이면 그대로 정규화
  if (safe.every((x) => x >= 0 && x <= 1) && sum > 0.8 && sum < 1.2) {
    return safe.map((x) => x / sum);
  }

  // logits일 가능성에 대비해 softmax 적용
  const max = Math.max(...safe);
  const exp = safe.map((x) => Math.exp(x - max));
  const expSum = exp.reduce((a, b) => a + b, 0) || 1;
  return exp.map((x) => x / expSum);
}

export async function predictAnimals(strokes) {
  if (!strokes?.flat()?.length) return [];

  const model = await loadAnimalModel();

  // Air Drawing과 QuickDraw 간 차이를 줄이기 위한 test-time ensemble
  const configs = [
    { lineWidth: 4, padRatio: 0.12, scaleAdjust: 1.02 },
    { lineWidth: 4.5, padRatio: 0.16, scaleAdjust: 1.00 },
    { lineWidth: 5, padRatio: 0.20, scaleAdjust: 0.98 },
    { lineWidth: 5.5, padRatio: 0.24, scaleAdjust: 0.96 },
    { lineWidth: 6, padRatio: 0.28, scaleAdjust: 0.94 },
  ];

  const outputs = [];

  for (const config of configs) {
    const canvas = render(strokes, config);
    if (!canvas) continue;
    outputs.push(normalizeProbabilities(await runOne(model, canvas)));
  }

  if (!outputs.length) return [];

  const mean = new Array(ANIMALS.length).fill(0);

  for (const output of outputs) {
    for (let i = 0; i < mean.length; i++) {
      mean[i] += Number(output[i] ?? 0);
    }
  }

  for (let i = 0; i < mean.length; i++) {
    mean[i] /= outputs.length;
  }

  const total = mean.reduce((a, b) => a + b, 0) || 1;

  return mean
    .map((p, i) => ({ ...ANIMALS[i], p: p / total }))
    .sort((a, b) => b.p - a.p);
}
