const KEY = "ai-animal-draw-v14";

function normalize(rows) {
  return [...rows].sort((a, b) => {
    const scoreDiff = Number(b.similarity || 0) - Number(a.similarity || 0);
    if (Math.abs(scoreDiff) > 1e-9) return scoreDiff;
    return Number(a.elapsed || 999) - Number(b.elapsed || 999);
  });
}

export function getScores() {
  try {
    return normalize(JSON.parse(localStorage.getItem(KEY) || "[]"));
  } catch {
    return [];
  }
}

export function saveScore(entry) {
  const row = {
    ...entry,
    id: crypto.randomUUID?.() || `${Date.now()}-${Math.random()}`,
    createdAt: new Date().toISOString(),
  };

  const rows = normalize([...getScores(), row]).slice(0, 200);
  localStorage.setItem(KEY, JSON.stringify(rows));

  return {
    row,
    rows,
    rank: rows.findIndex((x) => x.id === row.id) + 1,
  };
}
