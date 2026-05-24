import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, Pie, PieChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import {
  generateRawDataset, cleanDataset, kpis, revenueByMonth,
  revenueByCategory, ordersByRegion, avgOrderValueByCategory,
} from "@/lib/sales-data";

export const Route = createFileRoute("/")({ component: Dashboard });

const CHART_COLORS = [
  "oklch(0.82 0.17 180)",
  "oklch(0.75 0.18 50)",
  "oklch(0.7 0.2 320)",
  "oklch(0.8 0.18 130)",
  "oklch(0.7 0.15 230)",
];

const fmtCurrency = (n: number) =>
  new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 }).format(n);
const fmtNumber = (n: number) => new Intl.NumberFormat("en-US").format(n);

const tooltipStyle = {
  background: "oklch(0.21 0.025 250)",
  border: "1px solid oklch(0.3 0.02 250)",
  borderRadius: 12,
  fontSize: 12,
  color: "oklch(0.95 0.01 240)",
};

function Dashboard() {
  const raw = useMemo(() => generateRawDataset(), []);
  const { clean, report } = useMemo(() => cleanDataset(raw), [raw]);
  const [region, setRegion] = useState<string>("All");

  const filtered = useMemo(
    () => (region === "All" ? clean : clean.filter((r) => r.region === region)),
    [clean, region],
  );

  const k = useMemo(() => kpis(filtered), [filtered]);
  const monthData = useMemo(() => revenueByMonth(filtered), [filtered]);
  const catData = useMemo(() => revenueByCategory(filtered), [filtered]);
  const regionData = useMemo(() => ordersByRegion(filtered), [filtered]);
  const aovData = useMemo(() => avgOrderValueByCategory(filtered), [filtered]);
  const regions = ["All", "North", "South", "East", "West"];

  return (
    <main className="min-h-screen bg-background text-foreground">
      <header className="relative overflow-hidden border-b border-border">
        <div className="absolute inset-0 pointer-events-none" style={{ background: "var(--gradient-glow)" }} />
        <div className="relative mx-auto max-w-7xl px-6 py-12">
          <div className="flex items-center gap-2 text-xs font-mono uppercase tracking-[0.2em] text-primary">
            <span className="inline-block h-2 w-2 rounded-full bg-primary animate-pulse" />
            Pulse / Analyst Console
          </div>
          <h1 className="mt-4 text-4xl md:text-6xl font-semibold tracking-tight">
            From raw CSV<br />
            <span className="text-primary">to clean insight.</span>
          </h1>
          <p className="mt-4 max-w-2xl text-muted-foreground">
            An end-to-end data cleaning and visualization walkthrough on a synthetic e-commerce sales dataset
            (1,200+ orders) — handling missing values, duplicates, and outliers, then telling the story visually.
          </p>
        </div>
      </header>

      <div className="mx-auto max-w-7xl px-6 py-10 space-y-12">
        <section>
          <SectionHeader number="01" title="Data cleaning pipeline" subtitle="Pandas-style preprocessing, reproduced in the browser" />
          <div className="grid gap-4 md:grid-cols-3">
            <StatCard label="Raw rows ingested" value={fmtNumber(report.rawCount)} tone="muted" />
            <StatCard label="Clean rows kept" value={fmtNumber(report.cleanCount)} tone="primary" />
            <StatCard
              label="Dirt removed"
              value={fmtNumber(report.duplicatesRemoved + report.missingDropped + report.outliersClipped)}
              tone="accent"
              hint={`${report.duplicatesRemoved} dupes · ${report.missingDropped} missing · ${report.outliersClipped} outliers`}
            />
          </div>

          <div className="mt-6 rounded-2xl border border-border bg-card p-6">
            <ol className="space-y-4">
              {report.steps.map((s, i) => {
                const pct = (s.after / report.rawCount) * 100;
                return (
                  <li key={s.label} className="grid gap-3 md:grid-cols-[auto_1fr_auto] md:items-center">
                    <span className="font-mono text-xs text-muted-foreground">STEP {String(i + 1).padStart(2, "0")}</span>
                    <div>
                      <div className="font-medium">{s.label}</div>
                      <div className="text-sm text-muted-foreground">{s.note}</div>
                      <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-secondary">
                        <div className="h-full rounded-full bg-primary transition-all" style={{ width: `${pct}%` }} />
                      </div>
                    </div>
                    <span className="font-mono text-sm text-muted-foreground md:text-right">
                      {fmtNumber(s.before)} → {fmtNumber(s.after)}
                    </span>
                  </li>
                );
              })}
            </ol>
          </div>
        </section>

        <section>
          <div className="flex items-end justify-between gap-4 flex-wrap">
            <SectionHeader number="02" title="Key performance indicators" subtitle="Filter the cleaned dataset by region" />
            <div className="flex gap-1 rounded-full border border-border bg-card p-1">
              {regions.map((r) => (
                <button
                  key={r}
                  onClick={() => setRegion(r)}
                  className={`px-4 py-1.5 text-xs font-mono uppercase tracking-wider rounded-full transition ${
                    region === r ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {r}
                </button>
              ))}
            </div>
          </div>

          <div className="mt-4 grid gap-4 md:grid-cols-4">
            <StatCard label="Total revenue" value={fmtCurrency(k.revenue)} tone="primary" />
            <StatCard label="Orders" value={fmtNumber(k.orders)} tone="muted" />
            <StatCard label="Unique customers" value={fmtNumber(k.customers)} tone="muted" />
            <StatCard label="Avg order value" value={fmtCurrency(k.aov)} tone="accent" />
          </div>
        </section>

        <section>
          <SectionHeader number="03" title="Visual storytelling" subtitle="Where the revenue actually comes from" />
          <div className="grid gap-6 lg:grid-cols-2">
            <ChartCard title="Revenue trend" subtitle="Monthly revenue across 2024">
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={monthData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                  <CartesianGrid stroke="oklch(0.3 0.02 250)" strokeDasharray="3 3" />
                  <XAxis dataKey="month" stroke="oklch(0.65 0.02 250)" fontSize={12} />
                  <YAxis stroke="oklch(0.65 0.02 250)" fontSize={12} tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`} />
                  <Tooltip contentStyle={tooltipStyle} formatter={(v: number) => fmtCurrency(v)} />
                  <Line type="monotone" dataKey="revenue" stroke={CHART_COLORS[0]} strokeWidth={2.5} dot={{ r: 3, fill: CHART_COLORS[0] }} />
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="Revenue by category" subtitle="Top-performing product lines">
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={catData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                  <CartesianGrid stroke="oklch(0.3 0.02 250)" strokeDasharray="3 3" />
                  <XAxis dataKey="category" stroke="oklch(0.65 0.02 250)" fontSize={12} />
                  <YAxis stroke="oklch(0.65 0.02 250)" fontSize={12} tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`} />
                  <Tooltip contentStyle={tooltipStyle} formatter={(v: number) => fmtCurrency(v)} />
                  <Bar dataKey="revenue" radius={[6, 6, 0, 0]}>
                    {catData.map((_, i) => <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="Orders by region" subtitle="Geographic distribution of demand">
              <ResponsiveContainer width="100%" height={280}>
                <PieChart>
                  <Pie data={regionData} dataKey="orders" nameKey="region" cx="50%" cy="50%" innerRadius={60} outerRadius={100} paddingAngle={3}>
                    {regionData.map((_, i) => <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />)}
                  </Pie>
                  <Tooltip contentStyle={tooltipStyle} />
                  <Legend wrapperStyle={{ fontSize: 12, color: "oklch(0.65 0.02 250)" }} />
                </PieChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="Average order value" subtitle="By category — where customers spend most per order">
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={aovData} layout="vertical" margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
                  <CartesianGrid stroke="oklch(0.3 0.02 250)" strokeDasharray="3 3" />
                  <XAxis type="number" stroke="oklch(0.65 0.02 250)" fontSize={12} tickFormatter={(v) => `$${v}`} />
                  <YAxis type="category" dataKey="category" stroke="oklch(0.65 0.02 250)" fontSize={12} width={80} />
                  <Tooltip contentStyle={tooltipStyle} formatter={(v: number) => fmtCurrency(v)} />
                  <Bar dataKey="aov" fill={CHART_COLORS[1]} radius={[0, 6, 6, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>
          </div>
        </section>

        <section>
          <SectionHeader number="04" title="What the data is telling us" subtitle="Insights extracted after cleaning" />
          <div className="grid gap-4 md:grid-cols-3">
            <InsightCard
              title={`${catData[0]?.category ?? "Top category"} leads revenue`}
              body={`${catData[0]?.category ?? "—"} contributes ${fmtCurrency(catData[0]?.revenue ?? 0)} — roughly ${Math.round(((catData[0]?.revenue ?? 0) / (k.revenue || 1)) * 100)}% of total revenue in the cleaned dataset.`}
            />
            <InsightCard
              title="Outliers were skewing AOV"
              body={`${report.outliersClipped} extreme price/units values were winsorized via the 1.5×IQR rule, stabilising average order value at ${fmtCurrency(k.aov)}.`}
            />
            <InsightCard
              title="Data quality recovered"
              body={`After dropping ${report.duplicatesRemoved} duplicates and ${report.missingDropped} incomplete rows, ${Math.round((report.cleanCount / report.rawCount) * 100)}% of the dataset is usable for analysis.`}
            />
          </div>
        </section>

        <footer className="border-t border-border pt-6 text-xs font-mono text-muted-foreground">
          Pulse · Built with React, TanStack Start & Recharts · Synthetic dataset, deterministic seed
        </footer>
      </div>
    </main>
  );
}

function SectionHeader({ number, title, subtitle }: { number: string; title: string; subtitle: string }) {
  return (
    <div className="mb-6">
      <div className="font-mono text-xs uppercase tracking-[0.2em] text-primary">/ {number}</div>
      <h2 className="mt-1 text-2xl md:text-3xl font-semibold">{title}</h2>
      <p className="text-sm text-muted-foreground">{subtitle}</p>
    </div>
  );
}

function StatCard({ label, value, tone, hint }: { label: string; value: string; tone: "primary" | "accent" | "muted"; hint?: string }) {
  const accentClass = tone === "primary" ? "text-primary" : tone === "accent" ? "text-accent" : "text-foreground";
  return (
    <div className="rounded-2xl border border-border bg-card p-5">
      <div className="font-mono text-xs uppercase tracking-wider text-muted-foreground">{label}</div>
      <div className={`mt-2 text-3xl font-semibold tracking-tight ${accentClass}`}>{value}</div>
      {hint && <div className="mt-1 text-xs text-muted-foreground">{hint}</div>}
    </div>
  );
}

function ChartCard({ title, subtitle, children }: { title: string; subtitle: string; children: React.ReactNode }) {
  return (
    <div className="rounded-2xl border border-border bg-card p-5">
      <div className="mb-4">
        <div className="font-semibold">{title}</div>
        <div className="text-xs text-muted-foreground">{subtitle}</div>
      </div>
      {children}
    </div>
  );
}

function InsightCard({ title, body }: { title: string; body: string }) {
  return (
    <div className="rounded-2xl border border-border bg-card p-5">
      <div className="font-semibold">{title}</div>
      <p className="mt-2 text-sm text-muted-foreground leading-relaxed">{body}</p>
    </div>
  );
}