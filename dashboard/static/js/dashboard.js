
"use strict";

const TICKER_DATABASE = [
  { sym: "AAPL",  name: "Apple Inc." },
  { sym: "MSFT",  name: "Microsoft Corporation" },
  { sym: "GOOGL", name: "Alphabet Inc." },
  { sym: "AMZN",  name: "Amazon.com Inc." },
  { sym: "NVDA",  name: "NVIDIA Corporation" },
  { sym: "META",  name: "Meta Platforms Inc." },
  { sym: "TSLA",  name: "Tesla Inc." },
  { sym: "BRK-B", name: "Berkshire Hathaway B" },
  { sym: "JPM",   name: "JPMorgan Chase" },
  { sym: "V",     name: "Visa Inc." },
  { sym: "UNH",   name: "UnitedHealth Group" },
  { sym: "XOM",   name: "Exxon Mobil" },
  { sym: "JNJ",   name: "Johnson & Johnson" },
  { sym: "WMT",   name: "Walmart Inc." },
  { sym: "MA",    name: "Mastercard Inc." },
  { sym: "PG",    name: "Procter & Gamble" },
  { sym: "LLY",   name: "Eli Lilly and Company" },
  { sym: "HD",    name: "Home Depot" },
  { sym: "AVGO",  name: "Broadcom Inc." },
  { sym: "CVX",   name: "Chevron Corporation" },
  { sym: "MRK",   name: "Merck & Co." },
  { sym: "ABBV",  name: "AbbVie Inc." },
  { sym: "COST",  name: "Costco Wholesale" },
  { sym: "KO",    name: "Coca-Cola Company" },
  { sym: "PEP",   name: "PepsiCo Inc." },
  { sym: "ADBE",  name: "Adobe Inc." },
  { sym: "NFLX",  name: "Netflix Inc." },
  { sym: "AMD",   name: "Advanced Micro Devices" },
  { sym: "INTC",  name: "Intel Corporation" },
  { sym: "ORCL",  name: "Oracle Corporation" },
  { sym: "CSCO",  name: "Cisco Systems" },
  { sym: "CRM",   name: "Salesforce Inc." },
  { sym: "QCOM",  name: "Qualcomm Inc." },
  { sym: "TXN",   name: "Texas Instruments" },
  { sym: "NOW",   name: "ServiceNow Inc." },
  { sym: "INTU",  name: "Intuit Inc." },
  { sym: "AMAT",  name: "Applied Materials" },
  { sym: "LRCX",  name: "Lam Research" },
  { sym: "MU",    name: "Micron Technology" },
  { sym: "KLAC",  name: "KLA Corporation" },
  { sym: "PANW",  name: "Palo Alto Networks" },
  { sym: "SNOW",  name: "Snowflake Inc." },
  { sym: "PLTR",  name: "Palantir Technologies" },
  { sym: "SQ",    name: "Block Inc." },
  { sym: "PYPL",  name: "PayPal Holdings" },
  { sym: "SHOP",  name: "Shopify Inc." },
  { sym: "UBER",  name: "Uber Technologies" },
  { sym: "LYFT",  name: "Lyft Inc." },
  { sym: "COIN",  name: "Coinbase Global" },
  { sym: "HOOD",  name: "Robinhood Markets" },
  { sym: "BAC",   name: "Bank of America" },
  { sym: "GS",    name: "Goldman Sachs" },
  { sym: "MS",    name: "Morgan Stanley" },
  { sym: "WFC",   name: "Wells Fargo" },
  { sym: "C",     name: "Citigroup Inc." },
  { sym: "SCHW",  name: "Charles Schwab" },
  { sym: "BLK",   name: "BlackRock Inc." },
  { sym: "SPY",   name: "SPDR S&P 500 ETF" },
  { sym: "QQQ",   name: "Invesco QQQ Trust" },
  { sym: "DIA",   name: "SPDR Dow Jones ETF" },
  { sym: "IWM",   name: "iShares Russell 2000" },
  { sym: "GLD",   name: "SPDR Gold Shares" },
  { sym: "SLV",   name: "iShares Silver Trust" },
  { sym: "TLT",   name: "iShares 20+ Year Treasury" },
  { sym: "VTI",   name: "Vanguard Total Market ETF" },
  { sym: "VOO",   name: "Vanguard S&P 500 ETF" },
  { sym: "ARKK",  name: "ARK Innovation ETF" },
  { sym: "BABA",  name: "Alibaba Group" },
  { sym: "TSM",   name: "Taiwan Semiconductor" },
  { sym: "ASML",  name: "ASML Holding" },
  { sym: "NVO",   name: "Novo Nordisk" },
  { sym: "SAP",   name: "SAP SE" },
  { sym: "TM",    name: "Toyota Motor" },
];


//State

const STATE = {
  ticker:    "AAPL",
  range:     "1D",
  chartType: "line",
  horizon:   "1H",
  tapeMode:  "tape",
  activeTab: "news",
  watchlist: ["AAPL", "TSLA", "NVDA", "GOOGL", "MSFT", "AMZN", "SPY", "META"],
  quoteData:      {},
  _lastChartData: null,
  _lastPrice:     null,
  _perfOpen:      false,
};

//Chart indicator state
const CHART_STATE = {
  sma20: false,
  sma50: false,
  ema12: false,
  ema26: false,
  bb:    false,
  vol:   true,
  rsi:   false,
  macd:  false,
};


const PLOT_CONFIG = {
  displayModeBar: true,
  modeBarButtonsToRemove: ["toImage", "sendDataToCloud", "select2d", "lasso2d"],
  displaylogo: false, responsive: true,
};

function computeSMA(prices, period) {
  const result = new Array(prices.length).fill(null);
  for (let i = period - 1; i < prices.length; i++) {
    let sum = 0;
    for (let j = i - period + 1; j <= i; j++) sum += prices[j];
    result[i] = sum / period;
  }
  return result;
}

function computeEMA(prices, period) {
  const result = new Array(prices.length).fill(null);
  if (prices.length < period) return result;
  const k = 2 / (period + 1);
  let sum = 0;
  for (let i = 0; i < period; i++) sum += prices[i];
  result[period - 1] = sum / period;
  for (let i = period; i < prices.length; i++) {
    result[i] = prices[i] * k + result[i - 1] * (1 - k);
  }
  return result;
}

function computeRSI(prices, period = 14) {
  const result = new Array(prices.length).fill(null);
  if (prices.length < period + 2) return result;
  const gains = [], losses = [];
  for (let i = 1; i < prices.length; i++) {
    const diff = prices[i] - prices[i - 1];
    gains.push(Math.max(diff, 0));
    losses.push(Math.max(-diff, 0));
  }
  let avgG = gains.slice(0, period).reduce((a, b) => a + b, 0) / period;
  let avgL = losses.slice(0, period).reduce((a, b) => a + b, 0) / period;
  const rsi = (g, l) => l === 0 ? 100 : 100 - 100 / (1 + g / l);
  result[period] = rsi(avgG, avgL);
  for (let i = period + 1; i < prices.length; i++) {
    avgG = (avgG * (period - 1) + gains[i - 1]) / period;
    avgL = (avgL * (period - 1) + losses[i - 1]) / period;
    result[i] = rsi(avgG, avgL);
  }
  return result;
}

function computeMACD(prices) {
  const ema12 = computeEMA(prices, 12);
  const ema26 = computeEMA(prices, 26);
  const macd  = prices.map((_, i) =>
    ema12[i] != null && ema26[i] != null ? ema12[i] - ema26[i] : null);
  const validMACD = macd.filter(v => v != null);
  const emaSignal = computeEMA(validMACD, 9);
  const signal = new Array(prices.length).fill(null);
  let j = 0;
  for (let i = 0; i < prices.length; i++) {
    if (macd[i] != null) { signal[i] = emaSignal[j] ?? null; j++; }
  }
  const hist = prices.map((_, i) =>
    macd[i] != null && signal[i] != null ? macd[i] - signal[i] : null);
  return { macd, signal, hist };
}

function computeBollinger(prices, period = 20, mult = 2) {
  const upper = new Array(prices.length).fill(null);
  const mid   = new Array(prices.length).fill(null);
  const lower = new Array(prices.length).fill(null);
  for (let i = period - 1; i < prices.length; i++) {
    const window = prices.slice(i - period + 1, i + 1);
    const m = window.reduce((a, b) => a + b, 0) / period;
    const std = Math.sqrt(window.reduce((a, b) => a + (b - m) ** 2, 0) / period);
    mid[i]   = m;
    upper[i] = m + mult * std;
    lower[i] = m - mult * std;
  }
  return { upper, mid, lower };
}


let _chartInit = false;

function priceYRange(data) {
  const close = data.close || [];
  const low   = data.low   || [];
  const high  = data.high  || [];

  //For candlestick use full wick range; for line use close prices
  const allVals = [
    ...close.filter(v => v != null),
    ...low.filter(v => v != null),
    ...high.filter(v => v != null),
  ];
  if (!allVals.length) return null;

  let minP = Math.min(...allVals);
  let maxP = Math.max(...allVals);

  //Edge case: flat data
  if (minP === maxP) {
    const pad = minP * 0.01 || 1;
    return [minP - pad, maxP + pad];
  }

  const pad = (maxP - minP) * 0.05;
  return [minP - pad, maxP + pad];
}

function chartRangeBreaks(interval) {
  //Intraday intervals: hide weekends AND overnight non-trading hours
  if (["1m", "5m", "15m", "30m", "1h"].includes(interval)) {
    return [
      { bounds: ["sat", "mon"] },               //hide Sat–Sun
      { bounds: [16, 9.5], pattern: "hour" },   //hide 4 PM – 9:30 AM
    ];
  }
  //Daily intervals: hide weekends only (yfinance skips them, but Plotly still gaps)
  if (["1d"].includes(interval)) {
    return [{ bounds: ["sat", "mon"] }];
  }
  return [];
}

function buildChartLayout(h, data) {
  const hasVol  = CHART_STATE.vol;
  const hasRSI  = CHART_STATE.rsi;
  const hasMACD = CHART_STATE.macd;

  //Build yaxis domains bottom-up
  const panels = [];
  if (hasMACD) panels.push("macd");
  if (hasRSI)  panels.push("rsi");
  if (hasVol)  panels.push("vol");
  panels.push("price"); //price always on top

  const panelH = { price: 0.55, vol: 0.15, rsi: 0.15, macd: 0.15 };
  const available = panels.reduce((a, p) => a + panelH[p], 0);
  const scale = 1 / available;
  let bottom = 0;
  const domains = {};
  for (const p of [...panels].reverse()) {
    const h2 = panelH[p] * scale;
    domains[p] = [bottom, bottom + h2 - 0.02];
    bottom += h2;
  }

  const yRange = priceYRange(data);

  const layout = {
    paper_bgcolor: "transparent",
    plot_bgcolor:  "transparent",
    font: { family: "'SF Pro Display','Segoe UI',system-ui,sans-serif", color: "#6b82a8", size: 11 },
    margin: { t: 8, r: 16, b: 40, l: 60 },
    height: h,
    hovermode:  "x unified",
    hoverlabel: { bgcolor: "#0f1520", bordercolor: "#1e2d45", font: { color: "#e0eaff", size: 11 } },
    legend:     { bgcolor: "transparent", font: { color: "#6b82a8", size: 10 }, x: 0, y: 1 },
    showlegend: true,
    xaxis: {
      gridcolor: "#1e2d45", zeroline: false, showline: false,
      tickfont: { color: "#6b82a8", size: 10 },
      domain: [0, 1],
      rangebreaks: chartRangeBreaks(data.interval || ""),
    },
    yaxis: {
      gridcolor: "#1e2d45", zeroline: false, showline: false,
      tickfont: { color: "#6b82a8", size: 10 }, side: "right",
      domain: domains.price || [0.4, 1],
      ...(yRange ? { range: yRange, autorange: false } : {}),
    },
  };

  if (hasVol) {
    layout.yaxis2 = {
      gridcolor: "#1e2d45", zeroline: false, showline: false,
      tickfont: { color: "#6b82a8", size: 9 }, side: "right",
      domain: domains.vol, title: { text: "Vol", font: { size: 9 } },
    };
  }
  if (hasRSI) {
    layout.yaxis3 = {
      gridcolor: "#1e2d45", zeroline: false, showline: false,
      tickfont: { color: "#6b82a8", size: 9 }, side: "right",
      domain: domains.rsi, range: [0, 100],
      title: { text: "RSI", font: { size: 9 } },
    };
  }
  if (hasMACD) {
    layout.yaxis4 = {
      gridcolor: "#1e2d45", zeroline: false, showline: false,
      tickfont: { color: "#6b82a8", size: 9 }, side: "right",
      domain: domains.macd,
      title: { text: "MACD", font: { size: 9 } },
    };
  }

  //Subplots share xaxis
  const axes = ["yaxis2","yaxis3","yaxis4"].filter(a => layout[a]);
  for (const a of axes) layout[a].anchor = "x";

  return layout;
}

function buildChartTraces(data) {
  const ts    = data.timestamps || [];
  const close = data.close  || [];
  const open  = data.open   || [];
  const high  = data.high   || [];
  const low   = data.low    || [];
  const vol   = data.volume || [];
  const isUp  = close.length >= 2 ? close[close.length - 1] >= close[0] : true;
  const color = isUp ? "#00e676" : "#ff1744";

  const traces = [];

  //Price trace
  if (STATE.chartType === "candlestick") {
    traces.push({
      type: "candlestick", x: ts,
      open, high, low, close,
      increasing: { line: { color: "#00e676" } },
      decreasing: { line: { color: "#ff1744" } },
      name: STATE.ticker, yaxis: "y",
    });
  } else {
    //Invisible baseline at data min so fill hugs the price range, not y=0
    const validClose = close.filter(v => v != null);
    const baseY = validClose.length ? Math.min(...validClose) : 0;
    traces.push({
      type: "scatter", mode: "lines", x: ts, y: ts.map(() => baseY),
      yaxis: "y", showlegend: false, hoverinfo: "skip",
      line: { width: 0 }, name: "_baseline",
    });
    traces.push({
      type: "scatter", mode: "lines", x: ts, y: close,
      name: STATE.ticker, yaxis: "y",
      line:      { color, width: 2, shape: "spline", smoothing: 0.4 },
      fill:      "tonexty",
      fillcolor: isUp ? "rgba(0,230,118,0.06)" : "rgba(255,23,68,0.06)",
    });
  }

  //Overlay indicators (all on yaxis "y")
  if (CHART_STATE.sma20) {
    const sma20 = computeSMA(close, 20);
    traces.push({ type: "scatter", mode: "lines", x: ts, y: sma20, name: "SMA 20", yaxis: "y",
      line: { color: "#f0b429", width: 1.5, dash: "dot" }, showlegend: true });
  }
  if (CHART_STATE.sma50) {
    const sma50 = computeSMA(close, 50);
    traces.push({ type: "scatter", mode: "lines", x: ts, y: sma50, name: "SMA 50", yaxis: "y",
      line: { color: "#7b61ff", width: 1.5, dash: "dot" }, showlegend: true });
  }
  if (CHART_STATE.ema12) {
    const ema12 = computeEMA(close, 12);
    traces.push({ type: "scatter", mode: "lines", x: ts, y: ema12, name: "EMA 12", yaxis: "y",
      line: { color: "#00e5b0", width: 1.5 }, showlegend: true });
  }
  if (CHART_STATE.ema26) {
    const ema26 = computeEMA(close, 26);
    traces.push({ type: "scatter", mode: "lines", x: ts, y: ema26, name: "EMA 26", yaxis: "y",
      line: { color: "#ff9100", width: 1.5 }, showlegend: true });
  }
  if (CHART_STATE.bb) {
    const bb = computeBollinger(close);
    traces.push({ type: "scatter", mode: "lines", x: ts, y: bb.upper, name: "BB Upper", yaxis: "y",
      line: { color: "rgba(123,97,255,.5)", width: 1, dash: "dash" }, showlegend: false });
    traces.push({ type: "scatter", mode: "lines", x: ts, y: bb.mid, name: "BB Mid", yaxis: "y",
      line: { color: "rgba(123,97,255,.3)", width: 1 }, showlegend: false });
    traces.push({ type: "scatter", mode: "lines", x: ts, y: bb.lower, name: "BB Lower", yaxis: "y",
      line: { color: "rgba(123,97,255,.5)", width: 1, dash: "dash" }, showlegend: false,
      fill: "tonexty", fillcolor: "rgba(123,97,255,0.04)" });
  }

  //Volume subpanel
  if (CHART_STATE.vol && vol.length) {
    const barColors = close.map((c, i) =>
      i === 0 ? "rgba(0,196,255,0.5)" : c >= (close[i - 1] ?? c) ? "rgba(0,230,118,0.5)" : "rgba(255,23,68,0.5)");
    traces.push({
      type: "bar", x: ts, y: vol, name: "Volume", yaxis: "y2",
      marker: { color: barColors }, showlegend: false,
    });
  }

  //RSI subpanel
  if (CHART_STATE.rsi) {
    const rsi = computeRSI(close);
    traces.push({ type: "scatter", mode: "lines", x: ts, y: rsi, name: "RSI 14", yaxis: "y3",
      line: { color: "#00c4ff", width: 1.5 }, showlegend: false });
    //Overbought / oversold reference lines via shapes (done in layout shapes)
  }

  //MACD subpanel
  if (CHART_STATE.macd) {
    const { macd, signal, hist } = computeMACD(close);
    const histColors = hist.map(v => v == null ? "transparent" : v >= 0 ? "rgba(0,230,118,0.6)" : "rgba(255,23,68,0.6)");
    traces.push({ type: "bar", x: ts, y: hist, name: "MACD Hist", yaxis: "y4",
      marker: { color: histColors }, showlegend: false });
    traces.push({ type: "scatter", mode: "lines", x: ts, y: macd, name: "MACD", yaxis: "y4",
      line: { color: "#00c4ff", width: 1.5 }, showlegend: false });
    traces.push({ type: "scatter", mode: "lines", x: ts, y: signal, name: "Signal", yaxis: "y4",
      line: { color: "#ff9100", width: 1.5, dash: "dot" }, showlegend: false });
  }

  return traces;
}

async function refreshChart() {
  spinner("chart-spinner", true);
  try {
    const d = await apiFetch(`/api/history?ticker=${STATE.ticker}&range=${STATE.range}`);
    STATE._lastChartData = d;
    renderChart(d);
  } catch (e) {
    console.warn("Chart:", e);
    toast("Chart data unavailable", "#ff1744");
  } finally { spinner("chart-spinner", false); }
}

function renderChart(data) {
  const el = document.getElementById("main-chart");
  const h  = Math.max(el.parentElement.clientHeight - 50, 220);
  el.style.height = h + "px";

  const traces = buildChartTraces(data);
  const layout = buildChartLayout(h, data);

  if (_chartInit) {
    Plotly.react("main-chart", traces, layout, PLOT_CONFIG);
  } else {
    Plotly.newPlot("main-chart", traces, layout, PLOT_CONFIG);
    _chartInit = true;
  }

  const labels = {
    "1D":"Today","5D":"5 Days","1M":"1 Month","3M":"3 Months",
    "6M":"6 Months","1Y":"1 Year","MAX":"All Time"
  };
  setText("chart-range-label", labels[STATE.range] || STATE.range);
}


//Sparklines (SVG)

function renderSparkline(containerEl, prices) {
  if (!containerEl || !prices || prices.length < 2) {
    containerEl.innerHTML = "";
    return;
  }
  const W = containerEl.clientWidth || 70;
  const H = 28;
  const min = Math.min(...prices);
  const max = Math.max(...prices);
  const range = max - min || 1;
  const pts = prices.map((p, i) => {
    const x = (i / (prices.length - 1)) * W;
    const y = H - ((p - min) / range) * H;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  const isUp = prices[prices.length - 1] >= prices[0];
  const color = isUp ? "#00e676" : "#ff1744";
  containerEl.innerHTML = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
    <polyline points="${pts}" fill="none" stroke="${color}" stroke-width="1.5" stroke-linejoin="round"/>
  </svg>`;
}


//Price flash animation

function flashPrice(el, newPrice, oldPrice) {
  if (!el || oldPrice == null) return;
  el.classList.remove("flash-up", "flash-down");
  void el.offsetWidth; //reflow
  if (newPrice > oldPrice) el.classList.add("flash-up");
  else if (newPrice < oldPrice) el.classList.add("flash-down");
}


//Plotly dark theme utils

function fmt(n, d = 2) {
  if (n == null || isNaN(n)) return "—";
  return Number(n).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
}
function fmtCompact(n) {
  if (n == null || isNaN(n)) return "—";
  if (Math.abs(n) >= 1e12) return (n / 1e12).toFixed(2) + "T";
  if (Math.abs(n) >= 1e9)  return (n / 1e9).toFixed(2) + "B";
  if (Math.abs(n) >= 1e6)  return (n / 1e6).toFixed(2) + "M";
  if (Math.abs(n) >= 1e3)  return (n / 1e3).toFixed(1) + "K";
  return String(n);
}
function fmtSign(n, d = 2) {
  if (n == null || isNaN(n)) return "—";
  return (Number(n) >= 0 ? "+" : "") + fmt(n, d);
}
function setText(id, val) { const el = document.getElementById(id); if (el) el.textContent = val; }
function spinner(id, on)  { const el = document.getElementById(id); if (el) el.classList.toggle("hidden", !on); }
function escHtml(s) {
  return String(s || "").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}
function toast(msg, color = "#00c4ff") {
  const c = document.getElementById("toast-container");
  const el = document.createElement("div");
  el.className = "toast"; el.style.borderColor = color; el.textContent = msg;
  c.appendChild(el); setTimeout(() => el.remove(), 3000);
}
async function apiFetch(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}
function sentPill(label, score) {
  if (!label) return "";
  const sign = Number(score) >= 0 ? "+" : "";
  return `<span class="sent-pill ${label}">${label} ${sign}${fmt(score, 2)}</span>`;
}


//Clock

function startClock() {
  const tick = () => setText("clock", new Date().toLocaleTimeString("en-US", { hour12: false }));
  tick(); setInterval(tick, 1000);
}


//Market Status

async function refreshMarketStatus() {
  try {
    const d = await apiFetch("/api/market_status");
    const pill = document.getElementById("market-pill");
    pill.style.color = d.color;
    setText("market-label", d.status);
  } catch (_) {}
}


//Market Index Bar

const MB_MAP = {
  "sp500":   { price: "mb-sp500",  chg: "mb-sp500-chg"  },
  "nasdaq":  { price: "mb-nasdaq", chg: "mb-nasdaq-chg" },
  "dow":     { price: "mb-dow",    chg: "mb-dow-chg"    },
  "vix":     { price: "mb-vix",    chg: "mb-vix-chg"    },
  "russell": { price: "mb-russell",chg: "mb-russell-chg" },
};

async function refreshMarketBar() {
  try {
    const d = await apiFetch("/api/market_overview");
    for (const [key, ids] of Object.entries(MB_MAP)) {
      const item = d[key];
      if (!item) continue;
      setText(ids.price, `$${fmt(item.price)}`);
      const chgEl = document.getElementById(ids.chg);
      if (chgEl) {
        chgEl.textContent = `${fmtSign(item.change_pct, 2)}%`;
        chgEl.className = "mbar-chg " + (item.change_pct >= 0 ? "up" : "down");
      }
    }
  } catch (_) {}
}


//Ticker Tape

function buildTapeHTML(tickers) {
  return tickers.map(t => {
    const dir  = t.is_up ? "up" : "down";
    const sign = t.is_up ? "+" : "";
    return `<span class="tape-item" onclick="loadTicker('${escHtml(t.symbol)}')">
      <span class="tape-sym">${escHtml(t.symbol)}</span>
      <span class="tape-price">$${fmt(t.price)}</span>
      <span class="tape-chg ${dir}">${sign}${fmt(t.change_pct)}%</span>
    </span>`;
  }).join("");
}

function setTapeHTML(html) {
  document.getElementById("tape-inner").innerHTML = html;
  document.getElementById("tape-inner-clone").innerHTML = html;
}

async function refreshTape() {
  try {
    if (STATE.tapeMode === "gainers" || STATE.tapeMode === "losers") {
      const d = await apiFetch("/api/top_movers");
      const list = STATE.tapeMode === "gainers" ? (d.gainers || []) : (d.losers || []);
      setTapeHTML(buildTapeHTML(list));
    } else {
      const d = await apiFetch("/api/ticker_tape");
      if (d.tickers?.length) setTapeHTML(buildTapeHTML(d.tickers));
    }
  } catch (_) {}
}

function initTapeControls() {
  document.getElementById("tape-ctrl").addEventListener("click", e => {
    const btn = e.target.closest("[data-mode]");
    if (!btn) return;
    STATE.tapeMode = btn.dataset.mode;
    document.querySelectorAll(".tape-mode-btn").forEach(b =>
      b.classList.toggle("active", b.dataset.mode === STATE.tapeMode));
    refreshTape();
  });
}


//Autocomplete

function initAutocomplete() {
  const input = document.getElementById("ticker-input");
  const dd    = document.getElementById("ac-dropdown");

  function showDropdown(query) {
    const q = query.toUpperCase();
    const matches = TICKER_DATABASE.filter(t =>
      t.sym.startsWith(q) || t.name.toUpperCase().includes(q)
    ).slice(0, 8);
    if (!matches.length || !q) { dd.classList.remove("open"); return; }
    dd.innerHTML = matches.map(t =>
      `<div class="ac-item" data-sym="${t.sym}">
        <span class="ac-sym">${escHtml(t.sym)}</span>
        <span class="ac-name">${escHtml(t.name)}</span>
      </div>`
    ).join("");
    dd.classList.add("open");
  }

  input.addEventListener("input", e => showDropdown(e.target.value.trim()));
  input.addEventListener("keydown", e => {
    if (e.key === "Escape") dd.classList.remove("open");
    if (e.key === "ArrowDown") {
      const first = dd.querySelector(".ac-item");
      if (first) first.focus();
    }
  });

  dd.addEventListener("click", e => {
    const item = e.target.closest(".ac-item");
    if (!item) return;
    const sym = item.dataset.sym;
    input.value = sym;
    dd.classList.remove("open");
    loadTicker(sym);
  });

  dd.addEventListener("keydown", e => {
    if (e.key === "Enter") e.target.click();
    if (e.key === "ArrowDown") e.target.nextElementSibling?.focus();
    if (e.key === "ArrowUp")   e.target.previousElementSibling?.focus();
    if (e.key === "Escape")    dd.classList.remove("open");
  });

  document.addEventListener("click", e => {
    if (!e.target.closest("#search-input-wrap")) dd.classList.remove("open");
  });
}


//Live Quote

async function refreshQuote() {
  spinner("quote-spinner", true);
  try {
    const d = await apiFetch(`/api/price?ticker=${STATE.ticker}`);
    STATE.quoteData = d;
    renderQuote(d);
  } catch (e) { console.warn("Quote:", e); }
  finally     { spinner("quote-spinner", false); }
}
function renderQuote(d) {
  const isUp = d.is_up;
  const cls  = isUp ? "up" : "down";
  const sign = isUp ? "+" : "";
  setText("quote-ticker", d.symbol);
  const priceEl = document.getElementById("quote-price");
  const oldPrice = STATE._lastPrice;
  priceEl.textContent = `$${fmt(d.price)}`;
  priceEl.className   = cls;
  flashPrice(priceEl, d.price, oldPrice);
  STATE._lastPrice = d.price;

  const chgEl = document.getElementById("quote-change");
  chgEl.textContent = `${sign}$${fmt(Math.abs(d.change))}  (${sign}${fmt(d.change_pct)}%)`;
  chgEl.className   = cls;
  setText("s-vol",  fmtCompact(d.volume));
  setText("s-mcap", fmtCompact(d.market_cap));
  setText("s-high", d.day_high    ? `$${fmt(d.day_high)}`    : "—");
  setText("s-low",  d.day_low     ? `$${fmt(d.day_low)}`     : "—");
  setText("s-52h",  d["52w_high"] ? `$${fmt(d["52w_high"])}` : "—");
  setText("s-52l",  d["52w_low"]  ? `$${fmt(d["52w_low"])}`  : "—");
  setText("s-beta", d.beta        ? fmt(d.beta, 2)           : "—");
  setText("s-pe",   d.pe_ratio    ? fmt(d.pe_ratio, 1)       : "—");
}


//Company Info

async function refreshCompanyInfo() {
  try {
    const d = await apiFetch(`/api/company_info?ticker=${STATE.ticker}`);
    setText("quote-name",  d.name || STATE.ticker);
    setText("chart-title", `${d.name || STATE.ticker} (${STATE.ticker})`);
  } catch (_) { setText("quote-name", STATE.ticker); }
}


//News Feed (with impact, breaking, highlighted keywords)

async function refreshNews() {
  spinner("right-spinner", true);
  try {
    const d = await apiFetch(`/api/news?ticker=${STATE.ticker}&limit=25`);
    renderNews(d.articles || [], d.distribution || null);
  } catch (_) { renderNews([]); }
  finally { spinner("right-spinner", false); }
}

function renderDistBar(dist) {
  if (!dist) return;
  const posEl = document.getElementById("dist-pos");
  const neuEl = document.getElementById("dist-neu");
  const negEl = document.getElementById("dist-neg");
  if (posEl) posEl.style.width = (dist.positive || 0) + "%";
  if (neuEl) neuEl.style.width = (dist.neutral  || 0) + "%";
  if (negEl) negEl.style.width = (dist.negative || 0) + "%";
  setText("dist-pos-lbl", `Pos ${fmt(dist.positive || 0, 0)}%`);
  setText("dist-neu-lbl", `Neu ${fmt(dist.neutral  || 0, 0)}%`);
  setText("dist-neg-lbl", `Neg ${fmt(dist.negative || 0, 0)}%`);
}

function renderNews(articles, dist) {
  renderDistBar(dist);
  const list = document.getElementById("news-list");
  if (!articles.length) {
    list.innerHTML = `<div style="padding:20px;text-align:center;font-size:12px;color:var(--text2);">No news available</div>`;
    return;
  }
  list.innerHTML = articles.map(a => {
    const url     = a.link ? `href="${escHtml(a.link)}" target="_blank" rel="noopener"` : "";
    const pill    = sentPill(a.sentiment, a.sentiment_score);
    //Use server-side highlighted title (already HTML-escaped + keywords wrapped)
    const title   = a.highlighted_title || escHtml(a.title || "");
    const impact  = a.impact_score != null ? a.impact_score : null;
    const breakBadge = a.is_breaking
      ? `<span class="breaking-badge">● BREAKING</span>` : "";
    const impactBar = impact != null ? `
      <div class="impact-wrap">
        <span class="impact-label">Impact</span>
        <div class="impact-track"><div class="impact-fill" style="width:${(impact / 10) * 100}%"></div></div>
        <span class="impact-score">${fmt(impact, 1)}</span>
      </div>` : "";
    return `<a ${url} style="text-decoration:none;display:block;">
      <div class="news-item">
        <div class="news-title">${title}</div>
        ${impactBar}
        <div class="news-footer">
          <div class="news-footer-left">
            <div class="news-meta">
              <span class="news-pub">${escHtml(a.publisher || "")}</span>
              <span> · ${escHtml(a.time_str || "")}</span>
            </div>
            ${breakBadge}
          </div>
          ${pill}
        </div>
      </div>
    </a>`;
  }).join("");
}


//Sentiment Feed Tab

async function refreshSentimentFeed() {
  spinner("right-spinner", true);
  try {
    const d = await apiFetch(`/api/sentiment_feed?ticker=${STATE.ticker}&limit=25`);
    renderSentimentFeed(d);
  } catch (_) {}
  finally { spinner("right-spinner", false); }
}
function renderSentimentFeed(data) {
  const agg = data.aggregate || {};
  const sigEl = document.getElementById("agg-signal");
  if (sigEl) { sigEl.textContent = (agg.overall || "—").toUpperCase(); sigEl.style.color = agg.overall_color || "var(--text2)"; }
  setText("agg-score", agg.avg_score != null ? fmtSign(agg.avg_score, 3) : "—");
  setText("agg-pos",   agg.positive ?? 0);
  setText("agg-neu",   agg.neutral  ?? 0);
  setText("agg-neg",   agg.negative ?? 0);

  const list = document.getElementById("sentiment-list");
  const articles = data.articles || [];
  if (!articles.length) {
    list.innerHTML = `<div style="padding:20px;text-align:center;font-size:12px;color:var(--text2);">No sentiment data</div>`;
    return;
  }
  list.innerHTML = articles.map(a => {
    const url    = a.link ? `href="${escHtml(a.link)}" target="_blank" rel="noopener"` : "";
    const pill   = sentPill(a.sentiment, a.sentiment_score);
    const pct    = Math.min(Math.abs((a.sentiment_score || 0)) * 100, 100);
    const barClr = a.sentiment === "positive" ? "var(--up)" : a.sentiment === "negative" ? "var(--down)" : "var(--neutral)";
    return `<a ${url} style="text-decoration:none;display:block;">
      <div class="news-item">
        <div class="news-title">${escHtml(a.title)}</div>
        <div style="margin:6px 0 4px;height:3px;background:var(--bg3);border-radius:2px;overflow:hidden;">
          <div style="width:${pct}%;height:100%;background:${barClr};border-radius:2px;transition:width .5s;"></div>
        </div>
        <div class="news-footer">
          <div class="news-meta">
            <span class="news-pub">${escHtml(a.publisher || "")}</span>
            <span> · ${escHtml(a.time_str || "")}</span>
          </div>
          ${pill}
        </div>
      </div>
    </a>`;
  }).join("");
}


//AI Prediction (with horizon + breakdown + explain)

async function refreshPrediction() {
  try {
    const d = await apiFetch(`/api/predict?ticker=${STATE.ticker}&horizon=${STATE.horizon}`);
    renderPrediction(d);
  } catch (_) {}
}
function renderPrediction(d) {
  const isUp = d.direction === "UP";
  document.getElementById("pred-direction").textContent = d.direction || "—";
  document.getElementById("pred-direction").className   = isUp ? "up" : "down";
  setText("pred-confidence", `Confidence: ${fmt(d.confidence, 1)}%`);

  const upPct   = d.direction_probs?.UP   ?? 50;
  const downPct = d.direction_probs?.DOWN ?? 50;
  document.getElementById("bar-up").style.width   = upPct + "%";
  document.getElementById("bar-down").style.width = downPct + "%";
  setText("pct-up",   fmt(upPct, 1) + "%");
  setText("pct-down", fmt(downPct, 1) + "%");

  //Confidence breakdown
  const sentContrib = d.sentiment_contrib ?? null;
  const techContrib = d.technical_contrib ?? null;
  if (sentContrib != null) {
    document.getElementById("conf-sent").style.width = Math.min(sentContrib, 100) + "%";
    setText("conf-sent-pct", fmt(sentContrib, 0) + "%");
  }
  if (techContrib != null) {
    document.getElementById("conf-tech").style.width = Math.min(techContrib, 100) + "%";
    setText("conf-tech-pct", fmt(techContrib, 0) + "%");
  }

  setText("pred-sentiment", d.sentiment || "—");
  setText("pred-vol",       d.volatility ? fmt(d.volatility, 4) : "—");

  const badge = document.getElementById("pred-mode-badge");
  if (badge) {
    badge.textContent = d.mode === "model" ? "LIVE" : "DEMO";
    badge.style.color = d.mode === "model" ? "var(--up)" : "var(--text2)";
  }

  //Top news drivers for explain panel
  const topNews = d.top_news || [];
  const panel = document.getElementById("explain-panel");
  if (panel) {
    panel.innerHTML = topNews.length
      ? topNews.map(n => `<div class="explain-item">${n.highlighted_title || escHtml(n.title || "")}</div>`).join("")
      : `<div class="explain-item">No top drivers available.</div>`;
  }
}

function initHorizonButtons() {
  document.querySelectorAll(".hz-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      STATE.horizon = btn.dataset.hz;
      document.querySelectorAll(".hz-btn").forEach(b =>
        b.classList.toggle("active", b.dataset.hz === STATE.horizon));
      refreshPrediction();
    });
  });
}

function toggleExplain() {
  const panel  = document.getElementById("explain-panel");
  const toggle = document.getElementById("explain-toggle");
  if (!panel || !toggle) return;
  const open = panel.classList.toggle("open");
  toggle.textContent = open ? "▾ Hide Top Drivers" : "▸ Show Top Drivers";
}


//Watchlist (with sparklines)

async function refreshWatchlist() {
  const container = document.getElementById("watchlist-items");
  //Fetch sparklines in batch
  let sparklines = {};
  try {
    const d = await apiFetch(`/api/sparklines?tickers=${STATE.watchlist.join(",")}`);
    sparklines = d.sparklines || d || {};
  } catch (_) {}

  for (const sym of STATE.watchlist) {
    try {
      const d   = await apiFetch(`/api/price?ticker=${sym}`);
      const isUp = d.is_up;
      const sign = isUp ? "+" : "";

      let el = container.querySelector(`[data-sym="${sym}"]`);
      if (!el) {
        el = document.createElement("div");
        el.className   = "wl-item" + (sym === STATE.ticker ? " active" : "");
        el.dataset.sym = sym;
        el.innerHTML   = `
          <div class="wl-sym">${escHtml(sym)}</div>
          <div class="wl-spark" id="spark-${sym}"></div>
          <div class="wl-right">
            <div class="wl-price wl-p"></div>
            <div class="wl-chg wl-c"></div>
          </div>`;
        el.addEventListener("click", () => loadTicker(sym));
        container.appendChild(el);
      }

      el.querySelector(".wl-p").textContent = `$${fmt(d.price)}`;
      const chgEl = el.querySelector(".wl-c");
      chgEl.textContent = `${sign}${fmt(d.change_pct)}%`;
      chgEl.className   = `wl-chg ${isUp ? "up" : "down"}`;
      el.className = "wl-item" + (sym === STATE.ticker ? " active" : "");

      //Render sparkline
      const sparkEl = document.getElementById(`spark-${sym}`);
      if (sparkEl && sparklines[sym]) renderSparkline(sparkEl, sparklines[sym]);
    } catch (_) {}
  }
}
function updateWatchlistActive() {
  document.querySelectorAll(".wl-item").forEach(el =>
    el.classList.toggle("active", el.dataset.sym === STATE.ticker));
}


//Backtest / Performance Panel

let _perfChartInit = false;
const _btState = { strategy: "rsi", days: 252 };

async function togglePerf() {
  STATE._perfOpen = !STATE._perfOpen;
  const body = document.getElementById("perf-body");
  const icon = document.getElementById("perf-toggle-icon");
  body.classList.toggle("open", STATE._perfOpen);
  icon.classList.toggle("open", STATE._perfOpen);
  if (STATE._perfOpen && !_perfChartInit) await loadBacktest();
}

async function loadBacktest() {
  const { strategy, days } = _btState;
  try {
    const d = await apiFetch(
      `/api/backtest?ticker=${STATE.ticker}&strategy=${strategy}&days=${days}`
    );
    renderBacktest(d);
    _perfChartInit = true;
  } catch (e) { toast("Backtest unavailable", "#ff1744"); }
}

function _btFillMetrics(m) {
  const pct = (v, d = 2) => fmtSign(v ?? 0, d) + "%";
  const num = (v, d = 2) => fmt(v ?? 0, d);
  const setV = (id, val, cls) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = val;
    if (cls !== undefined) el.className = `perf-stat-val ${cls}`;
  };
  const sign = v => (v ?? 0) >= 0 ? "up" : "down";

  setV("bt-strat-ret",  pct(m.total_return),       sign(m.total_return));
  setV("bt-bh-ret",     pct(m.bh_return),           sign(m.bh_return));
  setV("bt-ann-return", pct(m.annualized_return),   sign(m.annualized_return));
  setV("bt-sharpe",     num(m.sharpe),              (m.sharpe ?? 0) >= 1 ? "up" : (m.sharpe ?? 0) >= 0 ? "" : "down");
  setV("bt-sortino",    num(m.sortino),             (m.sortino ?? 0) >= 1 ? "up" : (m.sortino ?? 0) >= 0 ? "" : "down");
  setV("bt-calmar",     num(m.calmar),              (m.calmar ?? 0) >= 1 ? "up" : "");
  setV("bt-maxdd",      pct(m.max_drawdown),        "down");
  setV("bt-winrate",    num(m.win_rate, 1) + "%",   (m.win_rate ?? 0) >= 50 ? "up" : "down");
  setV("bt-wl",         `${m.trades_won ?? 0}W / ${m.trades_lost ?? 0}L`);
  setV("bt-exposure",   num(m.exposure_pct, 1) + "%");
  setV("bt-alpha",      num(m.alpha),               sign(m.alpha));
  setV("bt-beta",       num(m.beta, 3));
}

function _btShowWarning(text) {
  const el = document.getElementById("bt-warning");
  if (!el) return;
  if (text) { el.textContent = "⚠ " + text; el.style.display = "block"; }
  else      { el.style.display = "none"; }
}

function _btEquityLayout(title) {
  return {
    paper_bgcolor: "transparent", plot_bgcolor: "transparent", height: 180,
    margin: { t: 22, r: 16, b: 28, l: 16 },
    title: { text: title, font: { color: "#6b82a8", size: 9 }, x: 0.01, xanchor: "left" },
    font: { family: "'SF Pro Display','Segoe UI',system-ui,sans-serif", color: "#6b82a8", size: 10 },
    xaxis: { gridcolor: "#1e2d45", zeroline: false, tickfont: { color: "#6b82a8", size: 9 } },
    yaxis: { gridcolor: "#1e2d45", zeroline: false, tickfont: { color: "#6b82a8", size: 9 }, side: "right", tickprefix: "$" },
    legend: { bgcolor: "transparent", font: { color: "#6b82a8", size: 10 }, x: 0, y: 1 },
    hovermode: "x unified",
    hoverlabel: { bgcolor: "#0f1520", bordercolor: "#1e2d45", font: { color: "#e0eaff", size: 10 } },
  };
}

function _btDrawdownLayout() {
  return {
    paper_bgcolor: "transparent", plot_bgcolor: "transparent", height: 80,
    margin: { t: 4, r: 16, b: 24, l: 16 },
    font: { family: "'SF Pro Display','Segoe UI',system-ui,sans-serif", color: "#6b82a8", size: 10 },
    xaxis: { gridcolor: "#1e2d45", zeroline: false, tickfont: { color: "#6b82a8", size: 9 } },
    yaxis: {
      gridcolor: "#1e2d45", zeroline: false, tickfont: { color: "#6b82a8", size: 9 },
      side: "right", ticksuffix: "%",
      title: { text: "Drawdown", font: { size: 8, color: "#6b82a8" } },
    },
    hovermode: "x unified",
    hoverlabel: { bgcolor: "#0f1520", bordercolor: "#1e2d45", font: { color: "#e0eaff", size: 10 } },
  };
}

const _BT_COLORS = ["#00c4ff", "#00e5b0", "#f0b429", "#ff6b6b", "#a78bfa"];

function renderBacktest(d) {
  const ts = d.timestamps || [];

  if (d.mode === "comparison") {
    const strategies = d.strategies || {};
    const bh = d.buy_hold || [];

    //Pick best-sharpe strategy for scalar metrics
    let bestKey = null, bestSharpe = -Infinity;
    for (const [k, v] of Object.entries(strategies)) {
      const sh = v.metrics?.sharpe ?? -Infinity;
      if (sh > bestSharpe) { bestSharpe = sh; bestKey = k; }
    }
    if (bestKey) {
      _btFillMetrics(strategies[bestKey].metrics);
      _btShowWarning(strategies[bestKey].metrics?.statistical_warning ?? null);
    }

    const equityTraces = [
      { type: "scatter", mode: "lines", x: ts, y: bh, name: "Buy & Hold",
        line: { color: "#7b61ff", width: 2, dash: "dot" } },
      ...Object.entries(strategies).map(([k, v], i) => ({
        type: "scatter", mode: "lines",
        x: ts, y: v.equity || [],
        name: v.label || k,
        line: { color: _BT_COLORS[i % _BT_COLORS.length], width: 2 },
      })),
    ];
    Plotly.newPlot("backtest-chart", equityTraces,
      _btEquityLayout("Strategy Comparison — Portfolio Value ($10k)"),
      { ...PLOT_CONFIG, displayModeBar: false });

    const ddTraces = Object.entries(strategies).map(([k, v], i) => ({
      type: "scatter", mode: "lines", fill: "tozeroy",
      x: ts, y: v.drawdown || [],
      name: v.label || k,
      line: { color: _BT_COLORS[i % _BT_COLORS.length], width: 1 },
      fillcolor: _BT_COLORS[i % _BT_COLORS.length] + "20",
      showlegend: false,
    }));
    Plotly.newPlot("drawdown-chart", ddTraces, _btDrawdownLayout(),
      { ...PLOT_CONFIG, displayModeBar: false });
    return;
  }

  //Single-strategy mode
  const m  = d.metrics || {};
  const eq = d.equity   || [];
  const bh = d.buy_hold || [];
  const dd = d.drawdown || [];

  _btFillMetrics(m);
  _btShowWarning(m.statistical_warning ?? null);

  Plotly.newPlot("backtest-chart", [
    { type: "scatter", mode: "lines", x: ts, y: eq,
      name: d.strategy_label || "Strategy",
      line: { color: "#00c4ff", width: 2 } },
    { type: "scatter", mode: "lines", x: ts, y: bh,
      name: "Buy & Hold",
      line: { color: "#7b61ff", width: 2, dash: "dot" } },
  ], _btEquityLayout(`${d.strategy_label || "Strategy"} vs Buy & Hold ($10k)`),
    { ...PLOT_CONFIG, displayModeBar: false });

  Plotly.newPlot("drawdown-chart", [
    { type: "scatter", mode: "lines", fill: "tozeroy",
      x: ts, y: dd, name: "Drawdown",
      line: { color: "#ff1744", width: 1 },
      fillcolor: "rgba(255,23,68,.15)",
      showlegend: false },
  ], _btDrawdownLayout(), { ...PLOT_CONFIG, displayModeBar: false });
}

function clearBtMetrics() {
  [
    "bt-strat-ret","bt-bh-ret","bt-ann-return","bt-sharpe","bt-sortino",
    "bt-calmar","bt-maxdd","bt-winrate","bt-wl","bt-exposure","bt-alpha","bt-beta",
  ].forEach(id => {
    const el = document.getElementById(id);
    if (el) { el.textContent = "—"; el.className = "perf-stat-val"; }
  });
  _btShowWarning(null);
}

function initBtControls() {
  document.querySelectorAll("#bt-strategy-btns .bt-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      _btState.strategy = btn.dataset.strategy;
      document.querySelectorAll("#bt-strategy-btns .bt-btn")
        .forEach(b => b.classList.toggle("active", b === btn));
      _perfChartInit = false;
      loadBacktest();
    });
  });
  document.querySelectorAll("#bt-period-btns .bt-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      _btState.days = parseInt(btn.dataset.days, 10);
      document.querySelectorAll("#bt-period-btns .bt-btn")
        .forEach(b => b.classList.toggle("active", b === btn));
      _perfChartInit = false;
      loadBacktest();
    });
  });
}


//Load ticker

async function loadTicker(sym) {
  sym = sym.toUpperCase().trim();
  if (!sym) return;
  STATE.ticker    = sym;
  STATE._lastPrice = null;
  document.getElementById("ticker-input").value = sym;
  document.getElementById("analyzer-ticker").value = sym;
  setText("quote-ticker", sym);
  setText("chart-title",  `${sym} — Price`);
  updateWatchlistActive();

  if (!STATE.watchlist.includes(sym)) {
    STATE.watchlist.unshift(sym);
    if (STATE.watchlist.length > 12) STATE.watchlist.pop();
    document.getElementById("watchlist-items").innerHTML = "";
  }

  _chartInit = false;
  _perfChartInit = false;
  clearBtMetrics();

  await Promise.allSettled([
    refreshCompanyInfo(),
    refreshQuote(),
    refreshChart(),
    refreshNews(),
    refreshSentimentFeed(),
    refreshPrediction(),
    refreshWatchlist(),
  ]);
  if (STATE._perfOpen) loadBacktest();
}


//Tabs

function initTabs() {
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const tab = btn.dataset.tab;
      STATE.activeTab = tab;
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.toggle("active", b.dataset.tab === tab));
      document.querySelectorAll(".tab-panel").forEach(p => p.classList.toggle("active", p.id === `tab-${tab}`));
      if (tab === "sentiment") refreshSentimentFeed();
      else                     refreshNews();
    });
  });
}


//Chart controls

function initChartControls() {
  document.getElementById("range-btns").addEventListener("click", e => {
    const btn = e.target.closest("[data-range]");
    if (!btn) return;
    document.querySelectorAll("[data-range]").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    STATE.range = btn.dataset.range;
    _chartInit = false;
    refreshChart();
  });
  document.getElementById("type-btns").addEventListener("click", e => {
    const btn = e.target.closest("[data-type]");
    if (!btn) return;
    document.querySelectorAll("[data-type]").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    STATE.chartType = btn.dataset.type;
    if (STATE._lastChartData) renderChart(STATE._lastChartData);
  });

  //Indicator overlay toggles
  document.querySelectorAll("[data-ind]").forEach(btn => {
    btn.addEventListener("click", () => {
      const ind = btn.dataset.ind;
      CHART_STATE[ind] = !CHART_STATE[ind];
      btn.classList.toggle("active", CHART_STATE[ind]);
      if (STATE._lastChartData) renderChart(STATE._lastChartData);
    });
  });

  //Sub-panel toggles
  document.querySelectorAll("[data-sub]").forEach(btn => {
    btn.addEventListener("click", () => {
      const sub = btn.dataset.sub;
      CHART_STATE[sub] = !CHART_STATE[sub];
      btn.classList.toggle("active", CHART_STATE[sub]);
      _chartInit = false;
      if (STATE._lastChartData) renderChart(STATE._lastChartData);
    });
  });
}


//Search

function initSearch() {
  const input = document.getElementById("ticker-input");
  const go    = () => {
    const s = input.value.trim().toUpperCase();
    document.getElementById("ac-dropdown").classList.remove("open");
    if (s) loadTicker(s);
  };
  document.getElementById("search-btn").addEventListener("click", go);
  input.addEventListener("keydown", e => { if (e.key === "Enter") go(); });
}


//Explain toggle (global for inline onclick)

window.togglePerf    = togglePerf;
window.toggleExplain = toggleExplain;

//Wire explain button
function initExplainToggle() {
  const btn = document.getElementById("explain-toggle");
  if (btn) btn.addEventListener("click", toggleExplain);
}


//Text Analyzer Modal

function initAnalyzerModal() {
  const modal    = document.getElementById("analyzer-modal");
  const openBtn  = document.getElementById("analyze-btn");
  const closeBtn = document.getElementById("close-analyzer");
  const runBtn   = document.getElementById("run-analysis-btn");
  const textarea = document.getElementById("news-textarea");
  const results  = document.getElementById("analysis-results");
  const errEl    = document.getElementById("result-error");

  openBtn.addEventListener("click", () => {
    modal.classList.add("open");
    document.getElementById("analyzer-ticker").value = STATE.ticker;
    textarea.focus();
  });
  closeBtn.addEventListener("click", () => modal.classList.remove("open"));
  modal.addEventListener("click", e => { if (e.target === modal) modal.classList.remove("open"); });
  document.addEventListener("keydown", e => {
    if (e.key === "Escape" && modal.classList.contains("open")) modal.classList.remove("open");
  });

  runBtn.addEventListener("click", async () => {
    const text   = textarea.value.trim();
    const ticker = (document.getElementById("analyzer-ticker").value.trim().toUpperCase()) || STATE.ticker;
    if (text.length < 5) { toast("Please enter at least a few words", "#ff1744"); return; }
    runBtn.disabled    = true;
    runBtn.textContent = "Analyzing...";
    results.classList.remove("visible");
    errEl.style.display = "none";
    try {
      const res = await fetch("/api/analyze_text", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ text, ticker }),
      });
      const d = await res.json();
      if (!res.ok || d.error) throw new Error(d.error || "Analysis failed");
      renderAnalysisResult(d);
      results.classList.add("visible");
    } catch (e) {
      errEl.textContent   = `Error: ${e.message}`;
      errEl.style.display = "block";
      results.classList.add("visible");
    } finally {
      runBtn.disabled    = false;
      runBtn.textContent = "Analyze Impact";
    }
  });
}
function renderAnalysisResult(d) {
  const isUp  = d.direction === "UP";
  const dirEl = document.getElementById("result-direction");
  dirEl.textContent = d.direction || "—";
  dirEl.className   = "result-direction " + (isUp ? "up" : "down");
  setText("result-conf",   `Confidence: ${fmt(d.confidence, 1)}%`);
  setText("result-ticker", d.ticker || "—");
  const sentPillEl = document.getElementById("result-sent-pill");
  sentPillEl.innerHTML = sentPill(d.finbert_sentiment || d.sentiment || "neutral", d.finbert_score ?? 0);
  const upPct   = d.direction_probs?.UP   ?? 50;
  const downPct = d.direction_probs?.DOWN ?? 50;
  document.getElementById("result-bar-up").style.width   = upPct + "%";
  document.getElementById("result-bar-down").style.width = downPct + "%";
  setText("result-pct-up",   fmt(upPct, 1) + "%");
  setText("result-pct-down", fmt(downPct, 1) + "%");
  const move = d.price_change_pct;
  setText("result-move",  move != null ? fmtSign(move, 2) + "%" : "—");
  setText("result-vol",   d.volatility    != null ? fmt(d.volatility, 4)    : "—");
  //Use FinBERT signed score (pos_prob - neg_prob) — financial-domain aware
  const finbertSigned = d.finbert_probs
    ? d.finbert_probs.positive - d.finbert_probs.negative
    : d.finbert_score ?? null;
  setText("result-vader", finbertSigned != null ? fmtSign(finbertSigned, 3) : "—");
  const moveEl = document.getElementById("result-move");
  if (moveEl && move != null) moveEl.className = "result-meta-val " + (move >= 0 ? "up" : "down");
  document.getElementById("result-error").style.display = "none";
}


//WebSocket (SocketIO)

function initSocketIO() {
  try {
    if (typeof io === "undefined") return;
    const sock = io({ transports: ["websocket", "polling"], reconnectionAttempts: 3 });
    sock.on("connect", () => { sock.emit("subscribe_ticker", { ticker: STATE.ticker }); });
    sock.on("ticker_tape", data => {
      if (data.tickers && STATE.tapeMode === "tape") {
        setTapeHTML(buildTapeHTML(data.tickers));
      }
    });
    sock.on("market_status", data => {
      document.getElementById("market-pill").style.color = data.color;
      setText("market-label", data.status);
    });
    sock.on("quote_update", data => { if (data.symbol === STATE.ticker) renderQuote(data); });
    sock.on("connect_error", () => {});
  } catch (_) {}
}


//Polling

function startPolling() {
  setInterval(() => refreshQuote(),        8_000);
  setInterval(() => refreshTape(),        12_000);
  setInterval(() => refreshMarketStatus(),60_000);
  setInterval(() => refreshMarketBar(),   60_000);
  setInterval(() => refreshWatchlist(),   30_000);
  setInterval(() => refreshPrediction(),  60_000);
  setInterval(() => {
    if (STATE.activeTab === "news") refreshNews();
    else                            refreshSentimentFeed();
  }, 300_000);
  setInterval(() => { if (STATE.range === "1D") refreshChart(); }, 30_000);
  setInterval(() => { if (STATE.range !== "1D") refreshChart(); }, 300_000);
}


//Boot

async function boot() {
  startClock();
  initSearch();
  initAutocomplete();
  initChartControls();
  initTabs();
  initHorizonButtons();
  initExplainToggle();
  initTapeControls();
  initAnalyzerModal();
  initBtControls();

  //Initialize volume sub-panel button as active
  document.querySelector("[data-sub='vol']")?.classList.add("active");

  await Promise.allSettled([refreshMarketStatus(), refreshTape(), refreshMarketBar()]);
  await loadTicker("AAPL");
  startPolling();
  initSocketIO();
}

document.addEventListener("DOMContentLoaded", boot);
