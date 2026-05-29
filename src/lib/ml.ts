// Lightweight ML toolkit (no deps) for the in-browser predictive modeling demo.
// Algorithms: Logistic Regression (multinomial softmax), K-Nearest Neighbors,
// Decision Tree (CART, gini). All produce class probabilities so we can draw
// per-class ROC curves.

import type { IrisRow } from "./iris-data";

export type Sample = { x: number[]; y: number };
export type Algo = "logreg" | "knn" | "tree";

// --- Deterministic PRNG so train/test split is reproducible ---
function mulberry32(seed: number) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = seed;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function trainTestSplit(rows: IrisRow[], testRatio: number, seed = 7) {
  const rand = mulberry32(seed);
  const idx = rows.map((_, i) => i);
  // Fisher–Yates
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  const cut = Math.round(rows.length * (1 - testRatio));
  const train = idx.slice(0, cut).map((i) => ({ x: rows[i].x.slice(), y: rows[i].y as number }));
  const test = idx.slice(cut).map((i) => ({ x: rows[i].x.slice(), y: rows[i].y as number }));
  return { train, test };
}

export function standardize(train: Sample[], test: Sample[]) {
  const d = train[0].x.length;
  const mean = new Array(d).fill(0);
  const std = new Array(d).fill(0);
  for (const r of train) for (let j = 0; j < d; j++) mean[j] += r.x[j];
  for (let j = 0; j < d; j++) mean[j] /= train.length;
  for (const r of train) for (let j = 0; j < d; j++) std[j] += (r.x[j] - mean[j]) ** 2;
  for (let j = 0; j < d; j++) std[j] = Math.sqrt(std[j] / train.length) || 1;
  const scale = (s: Sample): Sample => ({
    x: s.x.map((v, j) => (v - mean[j]) / std[j]),
    y: s.y,
  });
  return { train: train.map(scale), test: test.map(scale) };
}

// ---------------- Multinomial Logistic Regression ----------------

export type LogRegModel = { W: number[][]; b: number[]; nClasses: number };

function softmax(z: number[]) {
  const m = Math.max(...z);
  const ex = z.map((v) => Math.exp(v - m));
  const s = ex.reduce((a, b) => a + b, 0);
  return ex.map((v) => v / s);
}

export function trainLogReg(train: Sample[], nClasses: number, opts = { lr: 0.1, epochs: 400, l2: 0.01 }) {
  const d = train[0].x.length;
  const W: number[][] = Array.from({ length: nClasses }, () => new Array(d).fill(0));
  const b: number[] = new Array(nClasses).fill(0);
  const n = train.length;
  for (let epoch = 0; epoch < opts.epochs; epoch++) {
    const gW: number[][] = Array.from({ length: nClasses }, () => new Array(d).fill(0));
    const gB: number[] = new Array(nClasses).fill(0);
    for (const s of train) {
      const z = W.map((w, k) => w.reduce((acc, wj, j) => acc + wj * s.x[j], 0) + b[k]);
      const p = softmax(z);
      for (let k = 0; k < nClasses; k++) {
        const err = p[k] - (s.y === k ? 1 : 0);
        for (let j = 0; j < d; j++) gW[k][j] += err * s.x[j];
        gB[k] += err;
      }
    }
    for (let k = 0; k < nClasses; k++) {
      for (let j = 0; j < d; j++) W[k][j] -= (opts.lr / n) * (gW[k][j] + opts.l2 * W[k][j]);
      b[k] -= (opts.lr / n) * gB[k];
    }
  }
  return { W, b, nClasses };
}

export function predictLogRegProba(model: LogRegModel, x: number[]) {
  const z = model.W.map((w, k) => w.reduce((a, wj, j) => a + wj * x[j], 0) + model.b[k]);
  return softmax(z);
}

// ---------------- KNN ----------------

export type KnnModel = { train: Sample[]; k: number; nClasses: number };

export function trainKnn(train: Sample[], nClasses: number, k = 5): KnnModel {
  return { train, k, nClasses };
}

export function predictKnnProba(model: KnnModel, x: number[]) {
  const dists = model.train.map((s) => ({
    d: Math.sqrt(s.x.reduce((acc, v, j) => acc + (v - x[j]) ** 2, 0)),
    y: s.y,
  }));
  dists.sort((a, b) => a.d - b.d);
  const top = dists.slice(0, model.k);
  const counts = new Array(model.nClasses).fill(0);
  for (const t of top) counts[t.y] += 1;
  return counts.map((c) => c / model.k);
}

// ---------------- Decision Tree (CART, gini) ----------------

export type TreeNode =
  | { leaf: true; proba: number[] }
  | { leaf: false; feature: number; threshold: number; left: TreeNode; right: TreeNode };

function gini(counts: number[], total: number) {
  let g = 1;
  for (const c of counts) {
    const p = c / total;
    g -= p * p;
  }
  return g;
}

function classCounts(rows: Sample[], nClasses: number) {
  const c = new Array(nClasses).fill(0);
  for (const r of rows) c[r.y] += 1;
  return c;
}

function buildTree(rows: Sample[], nClasses: number, depth: number, maxDepth: number, minLeaf: number): TreeNode {
  const counts = classCounts(rows, nClasses);
  const total = rows.length;
  const probaLeaf = (): TreeNode => ({ leaf: true, proba: counts.map((c) => c / total) });
  if (depth >= maxDepth || rows.length <= minLeaf || gini(counts, total) === 0) return probaLeaf();

  const d = rows[0].x.length;
  let best: { feature: number; threshold: number; gain: number; left: Sample[]; right: Sample[] } | null = null;
  const baseG = gini(counts, total);
  for (let f = 0; f < d; f++) {
    const vals = Array.from(new Set(rows.map((r) => r.x[f]))).sort((a, b) => a - b);
    for (let i = 0; i < vals.length - 1; i++) {
      const thr = (vals[i] + vals[i + 1]) / 2;
      const left: Sample[] = []; const right: Sample[] = [];
      for (const r of rows) (r.x[f] <= thr ? left : right).push(r);
      if (left.length < minLeaf || right.length < minLeaf) continue;
      const gL = gini(classCounts(left, nClasses), left.length);
      const gR = gini(classCounts(right, nClasses), right.length);
      const g = (left.length / total) * gL + (right.length / total) * gR;
      const gain = baseG - g;
      if (!best || gain > best.gain) best = { feature: f, threshold: thr, gain, left, right };
    }
  }
  if (!best || best.gain <= 0) return probaLeaf();
  return {
    leaf: false,
    feature: best.feature,
    threshold: best.threshold,
    left: buildTree(best.left, nClasses, depth + 1, maxDepth, minLeaf),
    right: buildTree(best.right, nClasses, depth + 1, maxDepth, minLeaf),
  };
}

export type TreeModel = { root: TreeNode; nClasses: number; maxDepth: number };

export function trainTree(train: Sample[], nClasses: number, maxDepth = 5, minLeaf = 2): TreeModel {
  return { root: buildTree(train, nClasses, 0, maxDepth, minLeaf), nClasses, maxDepth };
}

export function predictTreeProba(model: TreeModel, x: number[]): number[] {
  let node: TreeNode = model.root;
  while (!node.leaf) node = x[node.feature] <= node.threshold ? node.left : node.right;
  return node.proba;
}

// ---------------- Metrics ----------------

export function argmax(a: number[]) {
  let best = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[best]) best = i;
  return best;
}

export function confusionMatrix(yTrue: number[], yPred: number[], nClasses: number) {
  const m: number[][] = Array.from({ length: nClasses }, () => new Array(nClasses).fill(0));
  for (let i = 0; i < yTrue.length; i++) m[yTrue[i]][yPred[i]] += 1;
  return m;
}

export function accuracy(yTrue: number[], yPred: number[]) {
  let n = 0;
  for (let i = 0; i < yTrue.length; i++) if (yTrue[i] === yPred[i]) n += 1;
  return n / yTrue.length;
}

export type ClassReportRow = { className: string; precision: number; recall: number; f1: number; support: number };

export function classificationReport(cm: number[][], classNames: string[]): ClassReportRow[] {
  const n = cm.length;
  return classNames.map((name, k) => {
    const tp = cm[k][k];
    let fp = 0, fn = 0, support = 0;
    for (let i = 0; i < n; i++) {
      if (i !== k) fp += cm[i][k];
      if (i !== k) fn += cm[k][i];
      support += cm[k][i];
    }
    const precision = tp / Math.max(tp + fp, 1);
    const recall = tp / Math.max(tp + fn, 1);
    const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
    return { className: name, precision, recall, f1, support };
  });
}

// ROC curve (one-vs-rest) — returns FPR/TPR points sorted by threshold, plus AUC.
export function rocCurveOvR(yTrue: number[], scores: number[]) {
  const pairs = scores.map((s, i) => ({ s, y: yTrue[i] })).sort((a, b) => b.s - a.s);
  const P = yTrue.reduce((acc, y) => acc + (y === 1 ? 1 : 0), 0);
  const N = yTrue.length - P;
  const pts: { fpr: number; tpr: number }[] = [{ fpr: 0, tpr: 0 }];
  let tp = 0, fp = 0;
  let prevScore = Number.POSITIVE_INFINITY;
  for (const { s, y } of pairs) {
    if (s !== prevScore) {
      pts.push({ fpr: N ? fp / N : 0, tpr: P ? tp / P : 0 });
      prevScore = s;
    }
    if (y === 1) tp += 1; else fp += 1;
  }
  pts.push({ fpr: N ? fp / N : 0, tpr: P ? tp / P : 0 });
  // AUC via trapezoidal rule
  let auc = 0;
  for (let i = 1; i < pts.length; i++) auc += (pts[i].fpr - pts[i - 1].fpr) * (pts[i].tpr + pts[i - 1].tpr) / 2;
  return { points: pts, auc };
}