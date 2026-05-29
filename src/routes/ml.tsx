import { createFileRoute, Link } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import {
  CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, Legend,
} from "recharts";

import { IRIS_CLASSES, IRIS_FEATURES, loadIris } from "@/lib/iris-data";
import {
  accuracy, argmax, classificationReport, confusionMatrix,
  predictKnnProba, predictLogRegProba, predictTreeProba,
  rocCurveOvR, standardize, trainKnn, trainLogReg, trainTestSplit, trainTree,
  type Algo,
} from "@/lib/ml";

export const Route = createFileRoute("/ml")({
  head: () => ({
    meta: [
      { title: "Predictive Modeling — Iris Classifier" },
      { name: "description", content: "Train logistic regression, KNN, and decision tree models on the Iris dataset. Evaluate with confusion matrices, ROC curves, and per-class metrics — all in the browser." },
      { property: "og:title", content: "Predictive Modeling — Iris Classifier" },
      { property: "og:description", content: "Supervised learning, in the browser. Confusion matrix + ROC + per-class precision/recall." },
    ],
  }),
  component: MLPage,
});

const ALGO_LABELS: Record<Algo, string> = {
  logreg: "Logistic Regression",
  knn: "K-Nearest Neighbors",
  tree: "Decision Tree (CART)",
};

const CLASS_COLORS = [
  "oklch(0.82 0.17 180)",
  "oklch(0.75 0.18 50)",
  "oklch(0.7 0.2 320)",
];

function MLPage() {
  const data = useMemo(() => loadIris(), []);
  const [algo, setAlgo] = useState<Algo>("logreg");
  const [testRatio, setTestRatio] = useState(0.3);
  const [k, setK] = useState(5);
  const [maxDepth, setMaxDepth] = useState(4);
  const [seed, setSeed] = useState(7);

  const result = useMemo(() => {
    const split = trainTestSplit(data, testRatio, seed);
    const { train, test } = standardize(split.train, split.test);
    const nClasses = IRIS_CLASSES.length;

    let probaFn: (x: number[]) => number[];
    if (algo === "logreg") {
      const m = trainLogReg(train, nClasses);
      probaFn = (x) => predictLogRegProba(m, x);
    } else if (algo === "knn") {
      const m = trainKnn(train, nClasses, k);
      probaFn = (x) => predictKnnProba(m, x);
    } else {
      const m = trainTree(train, nClasses, maxDepth);
      probaFn = (x) => predictTreeProba(m, x);
    }

    const evalSet = (rows: typeof train) => {
      const yTrue = rows.map((r) => r.y);
      const probs = rows.map((r) => probaFn(r.x));
      const yPred = probs.map(argmax);
      const cm = confusionMatrix(yTrue, yPred, nClasses);
      const acc = accuracy(yTrue, yPred);
      const report = classificationReport(cm, [...IRIS_CLASSES]);
      const roc = IRIS_CLASSES.map((_, c) => {
        const yBin = yTrue.map((y) => (y === c ? 1 : 0));
        const sc = probs.map((p) => p[c]);
        return rocCurveOvR(yBin, sc);
      });
      return { cm, acc, report, roc, n: rows.length };
    };

    return {
      train: evalSet(train),
      test: evalSet(test),
    };
  }, [data, algo, testRatio, k, maxDepth, seed]);

  // Build ROC chart data (50 interpolated points per class).
  const rocChart = useMemo(() => {
    const pts: Array<{ fpr: number } & Record<string, number>> = [];
    const N = 50;
    for (let i = 0; i <= N; i++) {
      const fpr = i / N;
      const row: { fpr: number } & Record<string, number> = { fpr };
      result.test.roc.forEach((curve, c) => {
        // Step interpolation: TPR at the largest point with fpr <= target
        let tpr = 0;
        for (const p of curve.points) if (p.fpr <= fpr && p.tpr >= tpr) tpr = p.tpr;
        row[`c${c}`] = tpr;
      });
      pts.push(row);
    }
    return pts;
  }, [result]);

  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="border-b border-border/40 bg-card/40 backdrop-blur">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3 px-4 py-4">
          <div>
            <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
              Project · Supervised Learning
            </p>
            <h1 className="font-display text-2xl font-semibold tracking-tight">
              Iris Classifier <span className="text-muted-foreground">/ ML Studio</span>
            </h1>
          </div>
          <Link
            to="/"
            className="rounded-md border border-border/60 px-3 py-1.5 text-xs font-medium text-muted-foreground transition hover:text-foreground"
          >
            ← Pulse dashboard
          </Link>
        </div>
      </header>

      <main className="mx-auto max-w-6xl space-y-8 px-4 py-8">
        <section className="rounded-2xl border border-border/40 bg-card/30 p-5">
          <h2 className="font-display text-lg font-semibold">1 · Configure experiment</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            150 samples · 4 features · 3 classes. Choose an algorithm and hyperparameters; the model retrains instantly.
          </p>
          <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Control label="Algorithm">
              <select
                value={algo}
                onChange={(e) => setAlgo(e.target.value as Algo)}
                className="w-full rounded-md border border-border/60 bg-background px-3 py-2 text-sm"
              >
                {(Object.keys(ALGO_LABELS) as Algo[]).map((a) => (
                  <option key={a} value={a}>{ALGO_LABELS[a]}</option>
                ))}
              </select>
            </Control>
            <Control label={`Test split: ${(testRatio * 100).toFixed(0)}%`}>
              <input
                type="range" min={0.1} max={0.5} step={0.05}
                value={testRatio}
                onChange={(e) => setTestRatio(Number(e.target.value))}
                className="w-full accent-[color:var(--primary)]"
              />
            </Control>
            {algo === "knn" && (
              <Control label={`Neighbors (k): ${k}`}>
                <input
                  type="range" min={1} max={25} step={2}
                  value={k}
                  onChange={(e) => setK(Number(e.target.value))}
                  className="w-full accent-[color:var(--primary)]"
                />
              </Control>
            )}
            {algo === "tree" && (
              <Control label={`Max depth: ${maxDepth}`}>
                <input
                  type="range" min={1} max={8} step={1}
                  value={maxDepth}
                  onChange={(e) => setMaxDepth(Number(e.target.value))}
                  className="w-full accent-[color:var(--primary)]"
                />
              </Control>
            )}
            <Control label={`Random seed: ${seed}`}>
              <input
                type="range" min={1} max={50} step={1}
                value={seed}
                onChange={(e) => setSeed(Number(e.target.value))}
                className="w-full accent-[color:var(--primary)]"
              />
            </Control>
          </div>
        </section>

        <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Kpi label="Train accuracy" value={`${(result.train.acc * 100).toFixed(1)}%`} sub={`${result.train.n} samples`} />
          <Kpi label="Test accuracy" value={`${(result.test.acc * 100).toFixed(1)}%`} sub={`${result.test.n} samples`} accent />
          <Kpi
            label="Macro F1 (test)"
            value={(result.test.report.reduce((s, r) => s + r.f1, 0) / result.test.report.length).toFixed(3)}
            sub="harmonic mean"
          />
          <Kpi
            label="Mean AUC (test)"
            value={(result.test.roc.reduce((s, r) => s + r.auc, 0) / result.test.roc.length).toFixed(3)}
            sub="one-vs-rest"
          />
        </section>

        <section className="grid gap-4 lg:grid-cols-2">
          <Card title="Confusion matrix (test set)" subtitle="rows = actual · columns = predicted">
            <ConfusionMatrix cm={result.test.cm} />
          </Card>
          <Card title="ROC curves (one-vs-rest)" subtitle="closer to the top-left = better separation">
            <div className="h-72">
              <ResponsiveContainer>
                <LineChart data={rocChart} margin={{ top: 10, right: 16, bottom: 4, left: -16 }}>
                  <CartesianGrid stroke="oklch(0.3 0.02 250 / 0.4)" strokeDasharray="3 3" />
                  <XAxis
                    dataKey="fpr" type="number" domain={[0, 1]}
                    tickFormatter={(v) => v.toFixed(1)}
                    stroke="oklch(0.65 0.02 250)" fontSize={11}
                    label={{ value: "False positive rate", position: "insideBottom", offset: -2, fontSize: 11, fill: "oklch(0.65 0.02 250)" }}
                  />
                  <YAxis
                    domain={[0, 1]} tickFormatter={(v) => v.toFixed(1)}
                    stroke="oklch(0.65 0.02 250)" fontSize={11}
                  />
                  <Tooltip contentStyle={tooltipStyle} formatter={(v: number) => v.toFixed(3)} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  {IRIS_CLASSES.map((name, c) => (
                    <Line
                      key={c} type="monotone" dataKey={`c${c}`}
                      name={`${name} (AUC ${result.test.roc[c].auc.toFixed(3)})`}
                      stroke={CLASS_COLORS[c]} strokeWidth={2} dot={false} isAnimationActive={false}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </section>

        <section>
          <Card title="Per-class report (test set)" subtitle="precision · recall · F1 · support">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border/40 text-left text-xs uppercase tracking-wider text-muted-foreground">
                    <th className="py-2 pr-4">Class</th>
                    <th className="py-2 pr-4">Precision</th>
                    <th className="py-2 pr-4">Recall</th>
                    <th className="py-2 pr-4">F1</th>
                    <th className="py-2 pr-4">Support</th>
                  </tr>
                </thead>
                <tbody>
                  {result.test.report.map((r, i) => (
                    <tr key={r.className} className="border-b border-border/20 last:border-0">
                      <td className="py-2 pr-4">
                        <span className="inline-block h-2 w-2 rounded-full align-middle" style={{ background: CLASS_COLORS[i] }} />
                        <span className="ml-2 font-medium">{r.className}</span>
                      </td>
                      <td className="py-2 pr-4 font-mono">{r.precision.toFixed(3)}</td>
                      <td className="py-2 pr-4 font-mono">{r.recall.toFixed(3)}</td>
                      <td className="py-2 pr-4 font-mono">{r.f1.toFixed(3)}</td>
                      <td className="py-2 pr-4 font-mono text-muted-foreground">{r.support}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </section>

        <section className="rounded-2xl border border-border/40 bg-card/30 p-5">
          <h2 className="font-display text-lg font-semibold">Notes</h2>
          <ul className="mt-3 space-y-2 text-sm text-muted-foreground">
            <li>• Features are standardized (zero mean, unit variance) using train-set statistics only.</li>
            <li>• Logistic regression: multinomial softmax trained with batch gradient descent + L2 regularization (400 epochs).</li>
            <li>• KNN: Euclidean distance on standardized features, uniform vote across <code>k</code> neighbors.</li>
            <li>• Decision tree: CART with Gini impurity, capped at <code>max_depth</code>.</li>
            <li>• ROC curves and AUC are computed one-vs-rest using class probabilities.</li>
            <li>• Dataset: classic Fisher Iris (<code>{IRIS_FEATURES.length}</code> features → <code>{IRIS_CLASSES.length}</code> classes).</li>
          </ul>
        </section>
      </main>
    </div>
  );
}

const tooltipStyle = {
  background: "oklch(0.21 0.025 250)",
  border: "1px solid oklch(0.3 0.02 250)",
  borderRadius: 12,
  fontSize: 12,
  color: "oklch(0.95 0.01 240)",
};

function Control({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs font-medium text-muted-foreground">{label}</span>
      {children}
    </label>
  );
}

function Kpi({ label, value, sub, accent }: { label: string; value: string; sub?: string; accent?: boolean }) {
  return (
    <div className={`rounded-2xl border p-4 ${accent ? "border-primary/40 bg-primary/5" : "border-border/40 bg-card/30"}`}>
      <p className="text-xs uppercase tracking-wider text-muted-foreground">{label}</p>
      <p className={`mt-1 font-display text-2xl font-semibold ${accent ? "text-primary" : ""}`}>{value}</p>
      {sub && <p className="mt-0.5 text-xs text-muted-foreground">{sub}</p>}
    </div>
  );
}

function Card({ title, subtitle, children }: { title: string; subtitle?: string; children: React.ReactNode }) {
  return (
    <div className="rounded-2xl border border-border/40 bg-card/30 p-5">
      <div className="mb-3">
        <h3 className="font-display text-base font-semibold">{title}</h3>
        {subtitle && <p className="text-xs text-muted-foreground">{subtitle}</p>}
      </div>
      {children}
    </div>
  );
}

function ConfusionMatrix({ cm }: { cm: number[][] }) {
  const max = Math.max(1, ...cm.flat());
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-separate border-spacing-1 text-center text-sm">
        <thead>
          <tr>
            <th></th>
            {IRIS_CLASSES.map((c) => (
              <th key={c} className="px-2 py-1 text-xs font-medium text-muted-foreground">pred: {c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {cm.map((row, i) => (
            <tr key={i}>
              <th className="pr-2 text-right text-xs font-medium text-muted-foreground">actual: {IRIS_CLASSES[i]}</th>
              {row.map((v, j) => {
                const intensity = v / max;
                const isDiag = i === j;
                const bg = isDiag
                  ? `oklch(0.55 0.18 180 / ${0.15 + intensity * 0.7})`
                  : `oklch(0.55 0.2 25 / ${0.05 + intensity * 0.55})`;
                return (
                  <td
                    key={j}
                    className="h-14 w-14 rounded-md font-mono text-base font-semibold tabular-nums"
                    style={{ background: bg }}
                  >
                    {v}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}