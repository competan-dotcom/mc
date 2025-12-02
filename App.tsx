import React, { useState, useEffect, useMemo, useRef, memo } from 'react';
import { 
  ComposedChart, 
  Area, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
} from 'recharts';
import { 
  Search, 
  AlertTriangle, 
  TrendingUp, 
  TrendingDown, 
  Loader2,
  Activity,
  ChevronDown,
  Info,
  Cpu,
  Feather, 
  Newspaper,
  ArrowUp
} from 'lucide-react';
import { GoogleGenAI } from "@google/genai";
import { Coin, SimResult, Stats, SimDataPoint, SimulationConfig, BinanceTicker } from './types';

// --- CONFIGURATION ---
const EXPONENTIAL_BACKOFF_MAX_RETRIES = 3;

// --- UTILS ---
const getCoinLogo = (symbol: string) => {
  if (!symbol) return ''; 
  const rawSymbol = symbol.replace('USDT', '').toLowerCase();
  return `https://raw.githubusercontent.com/spothq/cryptocurrency-icons/master/128/color/${rawSymbol}.png`;
};

const formatCurrency = (val: number | undefined) => {
  if (val === undefined || val === null) return '$0,00';
  let digits = 2;
  
  if (val < 1) digits = 4;
  if (val < 0.001) digits = 7;

  return new Intl.NumberFormat('tr-TR', { 
    style: 'currency', 
    currency: 'USD',
    minimumFractionDigits: digits,
    maximumFractionDigits: digits
  }).format(val).replace('$', '$'); 
};

// --- MATH ENGINE ---
const generateGaussian = (): number => {
  let u = 0, v = 0;
  while(u === 0) u = Math.random();
  while(v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
};

const runGBM = (historicalPrices: number[], days = 20, numSimulations = 500): SimResult | null => {
  if (!historicalPrices || historicalPrices.length === 0) return null;

  const returns: number[] = [];
  for (let i = 1; i < historicalPrices.length; i++) {
    const current = historicalPrices[i];
    const prev = historicalPrices[i - 1];
    returns.push(Math.log(current / prev));
  }

  const n = returns.length;
  const meanReturn = returns.reduce((a, b) => a + b, 0) / n;
  const variance = returns.reduce((acc, val) => acc + Math.pow(val - meanReturn, 2), 0) / (n - 1);
  const volatility = Math.sqrt(variance);
  const drift = meanReturn - (variance / 2);

  const lastPrice = historicalPrices[historicalPrices.length - 1];
  const allPaths: number[][] = [];

  for (let sim = 0; sim < numSimulations; sim++) {
    const path: number[] = [lastPrice];
    let currentPrice = lastPrice;
    for (let t = 1; t <= days; t++) {
      const Z = generateGaussian();
      const shock = drift + volatility * Z;
      currentPrice = currentPrice * Math.exp(shock);
      path.push(currentPrice);
    }
    allPaths.push(path);
  }

  const sortedPathIndices = allPaths
    .map((path, index) => ({ index, finalPrice: path[days] }))
    .sort((a, b) => a.finalPrice - b.finalPrice);

  const minPathIndex = sortedPathIndices[0].index; 
  const maxPathIndex = sortedPathIndices[sortedPathIndices.length - 1].index; 
  
  const randomIndices = Array.from({ length: 30 }, () => Math.floor(Math.random() * numSimulations));

  const aggregatedData: SimDataPoint[] = [];
  for (let t = 0; t <= days; t++) {
    const pricesAtT = allPaths.map(path => path[t]).sort((a, b) => a - b);
    const p5 = pricesAtT[Math.floor(numSimulations * 0.05)];
    const p25 = pricesAtT[Math.floor(numSimulations * 0.25)];
    const p50 = pricesAtT[Math.floor(numSimulations * 0.50)];
    const p75 = pricesAtT[Math.floor(numSimulations * 0.75)];
    const p95 = pricesAtT[Math.floor(numSimulations * 0.95)];

    const dayData: SimDataPoint = {
      day: t,
      median: p50,
      range: [p5, p95],
      innerRange: [p25, p75],
      p5: p5,
      p95: p95,
      outlierMin: allPaths[minPathIndex][t],
      outlierMax: allPaths[maxPathIndex][t],
      simLines: {}
    };

    randomIndices.forEach((simIndex, i) => {
      dayData[`sim${i}`] = allPaths[simIndex][t];
    });

    aggregatedData.push(dayData);
  }

  return {
    data: aggregatedData,
    stats: {
      current: lastPrice,
      projectedMedian: aggregatedData[days].median,
      projectedLow: aggregatedData[days].p5,
      projectedHigh: aggregatedData[days].p95,
      volatility: (volatility * 100).toFixed(2)
    }
  };
};

// --- GEMINI API HELPERS ---
const getGenAI = () => {
  const apiKey = process.env.API_KEY;
  if (!apiKey) {
    console.error("API_KEY is missing in environment variables.");
    return null;
  }
  return new GoogleGenAI({ apiKey });
};

async function fetchAIInsight(symbol: string, stats: Stats, duration: number): Promise<string> {
  const ai = getGenAI();
  if (!ai || !stats) return "API Anahtarı eksik veya veri yok.";

  const userQuery = `
    Aşağıdaki Monte Carlo simülasyon sonuçlarını analiz et ve kullanıcının yatırım yapmaması gerektiğini vurgulayan, tarafsız, tek paragraftan oluşan bir risk değerlendirmesi yap.
    Coin: ${symbol}
    Süre: ${duration} gün
    Güncel Fiyat: ${stats.current}
    Medyan Tahmin (P50): ${stats.projectedMedian}
    Kötü Senaryo (P5): ${stats.projectedLow}
    İyi Senaryo (P95): ${stats.projectedHigh}
  `;

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: userQuery,
      config: {
        systemInstruction: "Sen bir Finansal AI Analistisin. Kısa, öz ve tarafsız ol. Yatırım tavsiyesi verme.",
      }
    });
    return response.text || "Yapay zeka yanıtı alınamadı.";
  } catch (error) {
    console.error("Gemini Error:", error);
    return "Analiz servisi şu anda kullanılamıyor.";
  }
}

async function fetchNewsSummary(symbol: string): Promise<string> {
  const ai = getGenAI();
  if (!ai) return "API Anahtarı eksik.";

  const userQuery = `${symbol} kripto parası için son piyasa gelişmeleri ve duyarlılık özeti (3 kısa madde).`;
  
  try {
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: userQuery,
      config: {
        systemInstruction: "Güncel verilere dayan. Yatırım tavsiyesi verme.",
        tools: [{googleSearch: {}}],
      }
    });
    return response.text || "Haber verisi alınamadı.";
  } catch (error) {
    console.error("Gemini News Error:", error);
    return "Haber servisi şu anda kullanılamıyor.";
  }
}

// --- COMPONENTS ---

interface CollapsibleCardProps {
  title: string;
  icon: React.ElementType;
  children: React.ReactNode;
  loading?: boolean;
  colorClass?: string;
  defaultOpen?: boolean;
  minimizedText?: string;
}

const CollapsibleCard: React.FC<CollapsibleCardProps> = ({ 
  title, 
  icon: Icon, 
  children,
  loading = false, 
  colorClass = "text-slate-500",
  defaultOpen = false,
  minimizedText = "" 
}) => {
  const [isOpen, setIsOpen] = useState(defaultOpen);
  const hasContent = !loading && children;
  const showMinimizedText = !isOpen && minimizedText;

  return (
    <div className={`border rounded-xl overflow-hidden transition-all duration-300 mb-3 ${isOpen ? 'bg-white border-slate-200 shadow-md' : 'bg-slate-50 border-slate-100'}`}>
      <button 
        onClick={() => (hasContent || !loading) && setIsOpen(!isOpen)}
        disabled={loading}
        className={`w-full p-3 flex items-center justify-between text-left ${loading ? 'cursor-wait' : 'cursor-pointer hover:bg-slate-100'} transition-colors`}
      >
        <div className="flex items-center gap-3">
          <div className="p-0">
            {loading ? (
              <Loader2 size={16} className={`animate-spin ${colorClass}`} />
            ) : (
              <Icon size={16} className={colorClass} />
            )}
          </div>
          
          <div className="flex flex-col">
            <span className="text-sm font-bold text-slate-800 capitalize">{title}</span> 
            {loading ? (
              <span className="text-[10px] text-slate-400 font-medium animate-pulse">Yükleniyor...</span>
            ) : (
              showMinimizedText && (
                <span className={`text-[10px] font-medium flex items-center gap-1 ${colorClass}`}>
                   {minimizedText}
                </span>
              )
            )}
          </div>
        </div>

        {(hasContent || !loading) && (
          <div className={`transition-transform duration-300 ${isOpen ? 'rotate-180' : ''}`}>
            <ChevronDown size={16} className="text-slate-400" />
          </div>
        )}
      </button>

      <div className={`overflow-hidden transition-[max-height] duration-500 ease-in-out ${isOpen ? 'max-h-[500px]' : 'max-h-0'}`}>
        <div className="p-4 pt-0 text-xs text-slate-600 leading-relaxed font-medium border-t border-slate-100 mt-2 text-justify">
          <div className="pt-2">
            {children}
          </div>
        </div>
      </div>
    </div>
  );
};

interface AnalysisSectionProps {
  simResult: SimResult | null;
  selectedCoin: Coin | null;
  duration: number;
  handleDurationChange: (d: number) => void;
  aiInsight: string;
  newsSummary: string;
  aiLoading: boolean;
  newsLoading: boolean;
}

const AnalysisSection = memo(({
  simResult, 
  selectedCoin, 
  duration, 
  handleDurationChange, 
  aiInsight, 
  newsSummary, 
  aiLoading, 
  newsLoading
}: AnalysisSectionProps) => {
  if (!simResult || !selectedCoin) return null;

  return (
    <div className="animate-in slide-in-from-bottom-8 duration-700 flex flex-col pb-10">
      
      {/* SÜRE SEÇİMİ */}
      <div className="w-full bg-slate-100 p-1 rounded-xl flex mb-6 shadow-inner">
        {[20, 50, 100].map(d => (
          <button
            key={d}
            onClick={() => handleDurationChange(d)}
            className={`
              flex-1 py-2 rounded-lg text-xs font-bold transition-all duration-200 ease-in-out
              ${duration === d 
                ? 'bg-white text-rose-700 shadow-sm ring-1 ring-black/5 scale-100' 
                : 'text-slate-400 hover:text-slate-600'
              }
            `}
          >
            {d} Gün
          </button>
        ))}
      </div>

      {/* Coin Başlık */}
      <div className="mb-6 flex justify-between items-center">
        <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-full overflow-hidden flex items-center justify-center bg-white shadow-md border border-slate-100">
              {selectedCoin?.symbol && (
                  <img 
                    src={getCoinLogo(selectedCoin.symbol)} 
                    alt={selectedCoin.symbol}
                    className="w-full h-full object-cover"
                    onError={(e) => {
                      const target = e.target as HTMLImageElement;
                      target.style.display = 'none';
                      if(target.nextSibling) (target.nextSibling as HTMLElement).style.display = 'flex';
                    }}
                  />
              )}
              <div className={`w-full h-full items-center justify-center bg-slate-100 text-xs font-bold text-slate-400 ${selectedCoin?.symbol ? 'flex' : 'hidden'}`}>
                {selectedCoin?.symbol.substring(0,1)}
              </div>
            </div>
            
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-2xl font-bold text-slate-900">
                  {selectedCoin?.symbol}
                </h2>
                <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${selectedCoin?.change >= 0 ? 'bg-emerald-100 text-emerald-700' : 'bg-rose-100 text-rose-700'}`}>
                  {selectedCoin?.change > 0 ? '+' : ''}{selectedCoin?.change.toFixed(2)}%
                </span>
              </div>
              <div className="text-slate-500 font-medium text-xs flex items-center gap-1">
                <span className="opacity-70">Güncel Değer:</span>
                <span className="text-slate-900 font-bold">{formatCurrency(simResult?.stats.current)}</span>
              </div>
            </div>
        </div>
      </div>

      {/* GRAFİK KARTI */}
      <div className="bg-white rounded-[2rem] border border-slate-100 shadow-xl shadow-slate-200/50 p-2 mb-8 h-[320px] relative overflow-hidden">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={simResult?.data} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="mainGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#6366f1" stopOpacity={0.05}/>
                <stop offset="95%" stopColor="#6366f1" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
            <XAxis dataKey="day" hide padding={{ left: 0, right: 20 }} />
            <YAxis 
                orientation="right" 
                tickFormatter={(val) => val < 1 ? val.toFixed(4) : val >= 1000 ? `${(val/1000).toFixed(1)}k` : val.toFixed(2)}
                tick={{ fontSize: 10, fill: '#94a3b8', fontFamily: 'IBM Plex Sans' }} 
                axisLine={false}
                tickLine={false}
                domain={['auto', 'auto']}
                width={50}
            />
            <Tooltip 
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload;
                    return (
                      <div className="bg-white/95 backdrop-blur border border-slate-100 p-3 rounded-xl shadow-2xl text-xs min-w-[160px] font-binance">
                        <div className="font-bold text-slate-800 mb-2 border-b pb-1">{data.day}. Gün Simülasyonu</div>
                        <div className="space-y-1.5">
                          <div className="flex justify-between items-center">
                            <span className="text-emerald-600 font-medium flex items-center gap-1"><TrendingUp size={10}/> İyi (P95)</span>
                            <span className="font-mono font-bold">{formatCurrency(data.p95)}</span>
                          </div>
                          <div className="flex justify-between items-center bg-indigo-50 p-1 rounded">
                            <span className="text-indigo-600 font-bold">MEDYAN</span>
                            <span className="font-mono font-bold text-indigo-700">{formatCurrency(data.median)}</span>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-rose-600 font-medium flex items-center gap-1"><TrendingDown size={10}/> Kötü (P5)</span>
                            <span className="font-mono font-bold">{formatCurrency(data.p5)}</span>
                          </div>
                        </div>
                      </div>
                    );
                  }
                  return null;
                }}
            />
            
            <Area type="monotone" dataKey="range" stroke="none" fill="url(#mainGradient)" fillOpacity={0.2} isAnimationActive={false} />
            
            {Array.from({ length: 30 }).map((_, i) => (
              <Line 
                key={`sim${i}`}
                type="monotone"
                dataKey={`sim${i}`}
                stroke="#6366f1"
                strokeWidth={0.4}
                strokeOpacity={0.6} 
                dot={false}
                isAnimationActive={false}
              />
            ))}

            <Line type="monotone" dataKey="outlierMax" stroke="#10b981" strokeWidth={1.5} strokeOpacity={1} strokeDasharray="4 4" dot={false} isAnimationActive={false} />
            <Line type="monotone" dataKey="outlierMin" stroke="#f43f5e" strokeWidth={1.5} strokeOpacity={1} strokeDasharray="4 4" dot={false} isAnimationActive={false} />

            <Line 
              type="monotone" 
              dataKey="median" 
              stroke="#4f46e5" 
              strokeWidth={2.5} 
              dot={false}
              isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* İSTATİSTİK KARTLARI */}
      <div className="flex gap-2 justify-between items-stretch mt-4 mb-8">
        <div className="flex-1 bg-rose-50 rounded-2xl p-3 flex flex-col items-center justify-center border border-rose-100 relative mt-2">
          <div className="absolute -top-3 bg-white text-rose-600 text-[9px] font-bold px-2 py-0.5 rounded-full border border-rose-100 shadow-sm whitespace-nowrap">
            Dip Değer
          </div>
          <TrendingDown size={18} className="text-rose-500 mb-1 mt-1" />
          <div className="text-rose-900 font-bold text-sm truncate w-full text-center tracking-tight font-binance">
            {formatCurrency(simResult?.stats.projectedLow)}
          </div>
          <div className="text-rose-400 text-[8px] font-medium mt-0.5 text-center">5% Olasılık Havuzu</div>
        </div>

        <div className="flex-[1.3] bg-indigo-50 rounded-2xl p-3 flex flex-col items-center justify-center border-2 border-indigo-100 shadow-lg relative z-10 scale-105">
          <div className="absolute -top-3 bg-white text-indigo-600 text-[9px] font-bold px-2 py-0.5 rounded-full border border-indigo-100 shadow-sm whitespace-nowrap">
            Denge Noktası
          </div>
          <Activity size={22} className="text-indigo-500 mb-1 mt-1" />
          <div className="text-indigo-700 font-black text-base truncate w-full text-center tracking-tight font-binance">
            {formatCurrency(simResult?.stats.projectedMedian)}
          </div>
          <div className="text-indigo-400 text-[8px] font-bold mt-0.5 text-center">90% Olasılık Medyan Değer</div>
        </div>

        <div className="flex-1 bg-emerald-50 rounded-2xl p-3 flex flex-col items-center justify-center border border-emerald-100 relative mt-2">
          <div className="absolute -top-3 bg-white text-emerald-600 text-[9px] font-bold px-2 py-0.5 rounded-full border border-emerald-100 shadow-sm whitespace-nowrap">
            Pik Değer
          </div>
          <TrendingUp size={18} className="text-emerald-500 mb-1 mt-1" />
          <div className="text-emerald-900 font-bold text-sm truncate w-full text-center tracking-tight font-binance">
            {formatCurrency(simResult?.stats.projectedHigh)}
          </div>
          <div className="text-emerald-400 text-[8px] font-medium mt-0.5 text-center">5% Olasılık Havuzu</div>
        </div>
      </div>

      {/* AÇILIR KAPANIR ALANLAR */}
      <div className="space-y-1">
        {/* 1. Bilgi Notu */}
        <CollapsibleCard 
          title="Model Detayları & Bilgilendirme" 
          icon={Info} 
          colorClass="text-indigo-500"
          defaultOpen={false}
        >
          <p className="mb-2"><strong className="text-slate-800">Yapılan İşlem:</strong> Analiz edilen kripto paranın son 365 günlük fiyat verileri (Binance Klines - 1D) kullanılarak, Geometric Brownian Motion (GBM) adı verilen stokastik bir süreç ile gelecekteki olası fiyat hareketleri simüle edilir. Bu simülasyon 1000 farklı yolu kapsar.</p>
          <p className="mb-2"><strong className="text-slate-800">Monte Carlo (MC) Nedir?</strong> Matematikte ve finans mühendisliğinde kullanılan bir yöntemdir. Karmaşık sistemlerde belirsizliğin etkisini modellemek için rastgele örneklemeyi kullanır. </p>
          <p><strong className="text-slate-800">Grafik Açıklamaları:</strong> Grafik, simülasyonun hesapladığı olasılık aralığını gösterir. Medyan (Mavi Çizgi), 1000 yolun ortalama değerini temsil eder. Dip/Pik Değerler (Kesikli Çizgiler) ve Spagetti Görünümü ise belirsizlik konisinin sınırlarını ve dağılımını gösterir.</p>
        </CollapsibleCard>

        {/* 2. AI Insight */}
        <CollapsibleCard 
          title="AI Risk Değerlendirmesi" 
          icon={Feather} 
          colorClass="text-indigo-600"
          loading={aiLoading}
          defaultOpen={true}
        >
          {aiInsight || "Analiz bekleniyor..."}
        </CollapsibleCard>

        {/* 3. Haberler */}
        <CollapsibleCard 
          title="İlgili Coine Ait Piyasa Bülteni" 
          icon={Newspaper} 
          colorClass="text-emerald-600"
          loading={newsLoading}
        >
          {newsSummary || "Haberler bekleniyor..."}
        </CollapsibleCard>
      </div>
      
      <div className="mt-6 text-center text-slate-400 text-[10px] px-8 leading-relaxed font-medium">
        * Simülasyon son 365 gün verisiyle 1,000 kez çalıştırılmıştır (GBM Modeli).
      </div>
    </div>
  );
});

// --- MAIN APP ---

export default function CryptoMonteCarloAI() {
  const [view, setView] = useState<'home' | 'analysis'>('home'); 
  const [coins, setCoins] = useState<Coin[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCoin, setSelectedCoin] = useState<Coin | null>(null);
  const [duration, setDuration] = useState(20); 
  const [simResult, setSimResult] = useState<SimResult | null>(null);
  const [loading, setLoading] = useState(false); 
  const [historicalData, setHistoricalData] = useState<number[]>([]); 
  
  const [isSimulating, setIsSimulating] = useState(false);
  const [simulationText, setSimulationText] = useState('');
  
  const [aiInsight, setAiInsight] = useState('');
  const [newsSummary, setNewsSummary] = useState('');
  const [aiLoading, setAiLoading] = useState(false);
  const [newsLoading, setNewsLoading] = useState(false);

  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const fetchCoins = async () => {
      setLoading(true);
      try {
        const res = await fetch('https://api.binance.com/api/v3/ticker/24hr');
        const data = await res.json();
        const filtered = data
          .filter((t: BinanceTicker) => t.symbol.endsWith('USDT'))
          .sort((a: BinanceTicker, b: BinanceTicker) => parseFloat(b.quoteVolume) - parseFloat(a.quoteVolume))
          .slice(0, 300)
          .map((t: BinanceTicker) => ({
            symbol: t.symbol,
            price: parseFloat(t.lastPrice),
            change: parseFloat(t.priceChangePercent)
          }));
        setCoins(filtered);
      } catch (err) {
        console.error("Coin listesi alınamadı", err);
      } finally {
        setLoading(false);
      }
    };
    fetchCoins();
  }, []);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsDropdownOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [dropdownRef]);

  const triggerSimulation = (dataToSimulate: number[], dayDuration: number) => {
    if (!selectedCoin) return; 

    setIsSimulating(true);
    setAiInsight(''); 
    setNewsSummary(''); 
    
    setSimulationText("Monte Carlo Motoru Isınıyor...");

    setTimeout(() => { setSimulationText("Geçmiş Volatilite Taranıyor..."); }, 1500);
    setTimeout(() => { setSimulationText("10,000+ Olasılık Hesaplanıyor..."); }, 2800);

    setTimeout(() => {
      const result = runGBM(dataToSimulate, dayDuration);
      setSimResult(result);
      setIsSimulating(false);
      
      if (selectedCoin && result && result.stats) {
        setAiLoading(true);
        fetchAIInsight(selectedCoin.symbol, result.stats, dayDuration)
          .then(setAiInsight)
          .finally(() => setAiLoading(false));

        setNewsLoading(true);
        fetchNewsSummary(selectedCoin.symbol)
          .then(setNewsSummary)
          .finally(() => setNewsLoading(false));
      }
    }, 4000);
  };

  const runAnalysis = (coin: Coin) => {
    setIsDropdownOpen(false);
    setSearchTerm(coin.symbol);
    setSelectedCoin(coin);
    setAiInsight('');
    setNewsSummary('');
    
    setIsSimulating(true);
    setSimulationText("Bağlantı Kuruluyor...");

    setTimeout(async () => {
        setTimeout(() => setSimulationText("Monte Carlo Motoru Isınıyor..."), 1000);
        setTimeout(() => setSimulationText("Geçmiş Volatilite Taranıyor..."), 2500);
        
        try {
          const res = await fetch(`https://api.binance.com/api/v3/klines?symbol=${coin.symbol}&interval=1d&limit=365`);
          if (!res.ok) throw new Error("Binance Klines verisi çekilemedi.");
          const klinesData = await res.json();
          const closingPrices = klinesData.map((d: (string | number)[]) => parseFloat(d[4] as string));
          
          setHistoricalData(closingPrices); 
          
          setTimeout(() => {
             setSimulationText("10,000+ Olasılık Hesaplanıyor...");
             
             setTimeout(() => {
                 const result = runGBM(closingPrices, duration);
                 setSimResult(result);
                 
                 setView('analysis'); 
                 setIsSimulating(false); 
                 
                 setAiLoading(true);
                 fetchAIInsight(coin.symbol, result!.stats, duration)
                  .then(setAiInsight)
                  .finally(() => setAiLoading(false));

                 setNewsLoading(true);
                 fetchNewsSummary(coin.symbol)
                  .then(setNewsSummary)
                  .finally(() => setNewsLoading(false));
             }, 1500); 
             
          }, 2500); 

        } catch (err: any) {
          alert(`Veri çekilemedi: ${err.message}. Lütfen tekrar deneyin.`);
          setIsSimulating(false);
        }
    }, 50); 
  };
  
  const handleDurationChange = (newDuration: number) => {
    if (newDuration === duration) return;
    setDuration(newDuration);
    if (historicalData.length > 0 && selectedCoin) {
         triggerSimulation(historicalData, newDuration);
    }
  };

  const filteredCoins = useMemo(() => {
    if (!searchTerm) return [];
    return coins.filter(c => c.symbol.includes(searchTerm.toUpperCase())).slice(0, 5);
  }, [coins, searchTerm]);

  return (
    <div className="min-h-screen bg-white font-binance text-slate-900 flex justify-center">
      <div className="w-full max-w-md bg-white min-h-screen relative flex flex-col p-6 shadow-2xl">
        
        {/* --- HEADER --- */}
        <div className="relative bg-white z-30 pb-2 -mx-6 px-6 pt-6 border-b border-transparent transition-all">
          <header className="mb-4 flex flex-col items-start justify-center relative">
            
            {/* BAŞLIK SATIRI */}
            <div className="flex items-center justify-start">
                <h1 className="text-3xl font-bold tracking-tight text-slate-900">
                  Crypto Monte Carlo AI
                </h1>
            </div>
            
            {/* SLOGAN SATIRI */}
            <p className="text-slate-500 text-base font-medium mt-1 text-left">
              Kripto Geleceği Simüle Ediyoruz!..
            </p>
          </header>
          
          {/* Yasal Uyarı */}
          <div className="mb-6">
             <CollapsibleCard 
                title="Yasal Uyarı / Disclaimer" 
                icon={AlertTriangle} 
                colorClass="text-orange-500"
                defaultOpen={false} 
                minimizedText="Lütfen okuyun." 
             >
               Bu uygulama kesinlikle yatırım tavsiyesi değildir. Veriler Binance üzerinden alınır ve geçmiş volatilite kullanılarak saf matematiksel olasılıklar (Monte Carlo) hesaplanır. Geleceği kimse bilemez, ama simüle edebilir.
             </CollapsibleCard>
          </div>

          {/* Arama Alanı */}
          <div className="relative z-40" ref={dropdownRef}>
            <div className="relative">
              <div className="absolute inset-y-0 left-4 flex items-center pointer-events-none">
                {loading ? <Loader2 className="animate-spin text-rose-600" size={18}/> : <Search className="text-slate-400" size={18} />}
              </div>
              <input 
                type="text" 
                placeholder="Crypto coin ara (örn:ETH)" 
                className="w-full bg-slate-50 text-slate-900 placeholder:text-slate-400 py-3.5 pl-11 pr-4 rounded-2xl border-2 border-transparent focus:border-rose-700 focus:bg-white outline-none transition-all shadow-sm focus:shadow-lg focus:shadow-rose-700/20 font-binance text-sm tracking-wide font-medium"
                value={searchTerm}
                onChange={(e) => {
                  setSearchTerm(e.target.value);
                  setIsDropdownOpen(true); 
                }}
                onFocus={() => setIsDropdownOpen(true)}
              />
            </div>
            
            {isDropdownOpen && searchTerm && filteredCoins.length > 0 && (
              <div className="absolute top-full left-0 right-0 mt-2 bg-white rounded-xl border border-slate-100 shadow-xl overflow-hidden animate-in fade-in zoom-in-95 duration-200 max-h-[400px] overflow-y-auto">
                {filteredCoins.map(coin => (
                  <button
                    key={coin.symbol}
                    onClick={() => runAnalysis(coin)}
                    className="w-full p-3 flex items-center justify-between hover:bg-slate-50 border-b border-slate-50 last:border-0 transition-colors text-left group"
                  >
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 rounded-full overflow-hidden flex items-center justify-center bg-slate-100 border border-slate-200 shrink-0">
                        <img 
                          src={getCoinLogo(coin.symbol)} 
                          alt={coin.symbol}
                          className="w-full h-full object-cover"
                          onError={(e) => {
                            const target = e.target as HTMLImageElement;
                            target.style.display = 'none';
                            if(target.nextSibling) (target.nextSibling as HTMLElement).style.display = 'flex';
                          }}
                        />
                        <div className="hidden w-full h-full items-center justify-center bg-slate-200 text-[10px] font-bold text-slate-500">
                          {coin.symbol.substring(0,1)}
                        </div>
                      </div>
                      <div>
                        <div className="font-bold text-slate-800 group-hover:text-rose-700 transition-colors">{coin.symbol}</div>
                        <div className="text-xs text-slate-400 font-medium">Binance Spot</div>
                      </div>
                    </div>
                    <div className="flex flex-col items-end">
                      <span className="font-medium text-sm text-slate-800">{formatCurrency(coin.price)}</span>
                      <span className={`text-[10px] font-bold ${coin.change >= 0 ? 'text-emerald-500' : 'text-rose-500'}`}>
                        {coin.change > 0 ? '+' : ''}{coin.change.toFixed(2)}%
                      </span>
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* --- İÇERİK --- */}
        <div className="flex-1 flex flex-col mt-4 transition-all duration-500 relative min-h-[400px]">
          
          {/* SİMÜLASYON MOTORU (Loading Overlay) */}
          {isSimulating && (
            <div className="absolute inset-0 z-50 bg-white/95 backdrop-blur-sm flex flex-col items-center justify-center rounded-3xl animate-in fade-in duration-300">
              <div className="relative mb-6">
                <div className="absolute inset-0 bg-rose-100 rounded-full animate-ping opacity-75"></div>
                <div className="relative bg-rose-50 p-4 rounded-full border-2 border-rose-100 shadow-lg">
                  <Cpu size={48} className="text-rose-600 animate-spin-slow" />
                </div>
              </div>
              <h3 className="text-lg font-bold text-slate-800 mb-2 animate-pulse">
                {simulationText}
              </h3>
              <div className="w-48 h-1.5 bg-slate-100 rounded-full overflow-hidden">
                <div className="h-full bg-rose-500 animate-progress-indeterminate"></div>
              </div>
            </div>
          )}

          {view === 'home' ? (
            <div className="flex-1 flex flex-col items-center justify-center opacity-40 mb-20 animate-in fade-in slide-in-from-bottom-4">
              <ArrowUp size={64} strokeWidth={1} className="text-slate-500 opacity-60 animate-bounce-slow mb-4" />
              <p className="text-slate-500 font-medium">Bir coin seç ve simülasyonu başlat.</p>
            </div>
          ) : (
            /* ANALYSIS SECTION */
            !isSimulating && (
              <AnalysisSection 
                simResult={simResult}
                selectedCoin={selectedCoin}
                duration={duration}
                handleDurationChange={handleDurationChange}
                aiInsight={aiInsight}
                newsSummary={newsSummary}
                aiLoading={aiLoading}
                newsLoading={newsLoading}
              />
            )
          )}
        </div>

      </div>
    </div>
  );
}