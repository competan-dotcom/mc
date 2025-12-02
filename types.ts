export interface Coin {
  symbol: string;
  price: number;
  change: number;
}

export interface Stats {
  current: number;
  projectedMedian: number;
  projectedLow: number;
  projectedHigh: number;
  volatility: string;
}

export interface SimDataPoint {
  day: number;
  median: number;
  range: [number, number];
  innerRange: [number, number];
  p5: number;
  p95: number;
  outlierMin: number;
  outlierMax: number;
  [key: string]: any;
}

export interface SimResult {
  data: SimDataPoint[];
  stats: Stats;
}

export interface SimulationConfig {
  days: number;
  numSimulations: number;
}

export interface BinanceTicker {
  symbol: string;
  lastPrice: string;
  priceChangePercent: string;
  quoteVolume: string;
}