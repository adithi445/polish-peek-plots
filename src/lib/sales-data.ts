// Synthetic e-commerce sales dataset with intentional dirt (missing values,
// duplicates, outliers) so we can demonstrate a real cleaning pipeline.

export type RawRow = {
  order_id: string;
  date: string | null;
  category: string | null;
  region: string | null;
  units: number | null;
  unit_price: number | null;
  customer_id: string | null;
};

export type CleanRow = {
  order_id: string;
  date: string;
  category: string;
  region: string;
  units: number;
  unit_price: number;
  revenue: number;
  customer_id: string;
  month: string;
};

const CATEGORIES = ["Electronics", "Apparel", "Home", "Beauty", "Sports"];
const REGIONS = ["North", "South", "East", "West"];

// Deterministic PRNG so the "dataset" is stable across reloads.
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

export function generateRawDataset(): RawRow[] {
  const rand = mulberry32(42);
  const rows: RawRow[] = [];
  const start = new Date("2024-01-01").getTime();
  const end = new Date("2024-12-31").getTime();

  for (let i = 0; i < 1200; i++) {
    const cat = CATEGORIES[Math.floor(rand() * CATEGORIES.length)];
    const basePrice =
      cat === "Electronics" ? 220 :
      cat === "Apparel" ? 55 :
      cat === "Home" ? 90 :
      cat === "Beauty" ? 30 : 70;
    const date = new Date(start + rand() * (end - start)).toISOString().slice(0, 10);
    rows.push({
      order_id: `ORD-${10000 + i}`,
      date,
      category: cat,
      region: REGIONS[Math.floor(rand() * REGIONS.length)],
      units: Math.max(1, Math.round(1 + rand() * 5)),
      unit_price: Math.round((basePrice * (0.7 + rand() * 0.6)) * 100) / 100,
      customer_id: `CUST-${1000 + Math.floor(rand() * 300)}`,
    });
  }

  // Inject dirt: missing values
  for (let i = 0; i < 90; i++) {
    const idx = Math.floor(rand() * rows.length);
    const field = ["date", "category", "region", "units", "unit_price"][Math.floor(rand() * 5)] as keyof RawRow;
    (rows[idx] as Record<string, unknown>)[field] = null;
  }

  // Inject duplicates
  for (let i = 0; i < 35; i++) {
    const idx = Math.floor(rand() * rows.length);
    rows.push({ ...rows[idx] });
  }

  // Inject outliers (absurd unit prices and unit counts)
  for (let i = 0; i < 12; i++) {
    const idx = Math.floor(rand() * rows.length);
    rows[idx] = { ...rows[idx], unit_price: 9999 + rand() * 5000, units: 250 + Math.floor(rand() * 400) };
  }

  return rows;
}

export type CleaningReport = {
  rawCount: number;
  cleanCount: number;
  duplicatesRemoved: number;
  missingDropped: number;
  outliersClipped: number;
  steps: { label: string; before: number; after: number; note: string }[];
};

export function cleanDataset(raw: RawRow[]): { clean: CleanRow[]; report: CleaningReport } {
  const rawCount = raw.length;

  // 1. Drop duplicates by order_id (keep first)
  const seen = new Set<string>();
  const deduped = raw.filter((r) => {
    if (seen.has(r.order_id)) return false;
    seen.add(r.order_id);
    return true;
  });
  const duplicatesRemoved = rawCount - deduped.length;

  // 2. Drop rows missing any critical field
  const complete = deduped.filter(
    (r) => r.date && r.category && r.region && r.units != null && r.unit_price != null && r.customer_id,
  );
  const missingDropped = deduped.length - complete.length;

  // 3. Clip outliers using IQR on unit_price and units
  const prices = complete.map((r) => r.unit_price!).sort((a, b) => a - b);
  const units = complete.map((r) => r.units!).sort((a, b) => a - b);
  const q = (arr: number[], p: number) => arr[Math.floor(arr.length * p)];
  const priceLo = q(prices, 0.25);
  const priceHi = q(prices, 0.75);
  const priceIQR = priceHi - priceLo;
  const priceMax = priceHi + 1.5 * priceIQR;
  const unitLo = q(units, 0.25);
  const unitHi = q(units, 0.75);
  const unitMax = unitHi + 1.5 * (unitHi - unitLo);

  let outliersClipped = 0;
  const clean: CleanRow[] = complete.map((r) => {
    let price = r.unit_price!;
    let u = r.units!;
    if (price > priceMax) { price = priceMax; outliersClipped++; }
    if (u > unitMax) { u = Math.round(unitMax); outliersClipped++; }
    return {
      order_id: r.order_id,
      date: r.date!,
      category: r.category!,
      region: r.region!,
      units: u,
      unit_price: Math.round(price * 100) / 100,
      revenue: Math.round(price * u * 100) / 100,
      customer_id: r.customer_id!,
      month: r.date!.slice(0, 7),
    };
  });

  return {
    clean,
    report: {
      rawCount,
      cleanCount: clean.length,
      duplicatesRemoved,
      missingDropped,
      outliersClipped,
      steps: [
        { label: "Deduplicate by order_id", before: rawCount, after: deduped.length, note: `${duplicatesRemoved} duplicate rows removed` },
        { label: "Drop rows with missing critical fields", before: deduped.length, after: complete.length, note: `${missingDropped} incomplete rows removed` },
        { label: "Clip outliers via 1.5×IQR (price & units)", before: complete.length, after: clean.length, note: `${outliersClipped} values winsorized` },
      ],
    },
  };
}

// ---- Aggregations for the dashboard ----

export function revenueByMonth(rows: CleanRow[]) {
  const map = new Map<string, number>();
  rows.forEach((r) => map.set(r.month, (map.get(r.month) ?? 0) + r.revenue));
  return Array.from(map.entries())
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([month, revenue]) => ({ month, revenue: Math.round(revenue) }));
}

export function revenueByCategory(rows: CleanRow[]) {
  const map = new Map<string, number>();
  rows.forEach((r) => map.set(r.category, (map.get(r.category) ?? 0) + r.revenue));
  return Array.from(map.entries())
    .map(([category, revenue]) => ({ category, revenue: Math.round(revenue) }))
    .sort((a, b) => b.revenue - a.revenue);
}

export function ordersByRegion(rows: CleanRow[]) {
  const map = new Map<string, number>();
  rows.forEach((r) => map.set(r.region, (map.get(r.region) ?? 0) + 1));
  return Array.from(map.entries()).map(([region, orders]) => ({ region, orders }));
}

export function avgOrderValueByCategory(rows: CleanRow[]) {
  const agg = new Map<string, { sum: number; n: number }>();
  rows.forEach((r) => {
    const a = agg.get(r.category) ?? { sum: 0, n: 0 };
    a.sum += r.revenue; a.n += 1;
    agg.set(r.category, a);
  });
  return Array.from(agg.entries()).map(([category, { sum, n }]) => ({
    category,
    aov: Math.round(sum / n),
  }));
}

export function kpis(rows: CleanRow[]) {
  const revenue = rows.reduce((s, r) => s + r.revenue, 0);
  const orders = rows.length;
  const customers = new Set(rows.map((r) => r.customer_id)).size;
  const aov = revenue / Math.max(orders, 1);
  return {
    revenue: Math.round(revenue),
    orders,
    customers,
    aov: Math.round(aov),
  };
}