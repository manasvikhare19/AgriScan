'use client'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Activity, ArrowUpRight, BarChart3, CheckCircle2, CloudRain, Download, FileImage,
  History, Languages, Leaf, Loader2, MapPin, RefreshCw, ScanLine, ShieldCheck,
  Sparkles, UploadCloud, X, Zap, AlertTriangle, Check, Wind, Droplets, Thermometer, Trash2
} from 'lucide-react'

type TopPrediction = {
  class_id?: number
  label?: string
  disease?: string
  display_name?: string
  confidence: number
  confidence_pct?: string
  severity?: string
  is_healthy?: boolean
  recommendations?: string[]
}

type PredictionResponse = {
  success?: boolean
  disease?: string
  prediction?: string
  confidence?: number
  confidence_pct?: string
  severity?: string
  is_healthy?: boolean
  recommendations?: string[]
  top_prediction?: TopPrediction
  top5?: Array<{
    class_id?: number
    label?: string
    disease?: string
    display_name?: string
    confidence: number
    confidence_pct?: string
  }>
  error?: string
  [key: string]: unknown
}

type HistoryItem = {
  id: string
  name: string
  disease: string
  confidence: number
  time: string
}

type WeatherRisk = {
  disease: string
  risk: 'High' | 'Medium' | 'Low'
  reason: string
}

type WeatherData = {
  city: string
  temp: number
  feels_like: number
  humidity: number
  wind_kph: number
  condition: string
  spray_ok: boolean
  spray_advice: string
  disease_risk: WeatherRisk[]
}

const copy = {
  en: {
    eyebrow: 'FIELD INTELLIGENCE / 01',
    title: 'See what your crops are saying.',
    sub: 'A calm, precise second opinion for every leaf, powered by your existing AgriScan model.',
    scan: 'New scan',
    history: 'Scan history',
    weather: 'Weather risk',
    select: 'Select a crop image',
    drop: 'or drop a JPG, PNG, or WEBP here',
    analyze: 'Analyze leaf',
    analyzing: 'Reading leaf signals…',
    idle: 'Ready for your next field note',
    confidence: 'Model confidence',
    recommendations: 'Field notes & recommendations',
    clear: 'Clear',
    test: 'Test connection',
    connecting: 'Testing API...',
    online: 'MODEL ONLINE',
    offline: 'MODEL OFFLINE',
    export: 'Export report',
    noHistory: 'No scans yet. Your analyzed leaves will be saved here.',
    clearHistory: 'Clear history',
    primarySignal: 'PRIMARY SIGNAL',
    modelBreakdown: 'MODEL BREAKDOWN',
    inputSection: '01 / INPUT',
    outputSection: '02 / OUTPUT',
    actionSection: '03 / ACTION',
    weatherHeading: 'Field Weather & Crop Disease Risk',
    weatherSub: 'Analyze current environmental conditions and disease susceptibility for your crops.',
    getGps: 'Use Current Location',
    fetchWeather: 'Fetch Weather Risk',
    fetchingWeather: 'Fetching environmental data…',
    cropLabel: 'Select Crop',
    temp: 'Temperature',
    humidity: 'Humidity',
    wind: 'Wind Speed',
    sprayAdvisory: 'Spray Advisory',
    diseaseRisks: 'Identified Disease Risks',
    noRisks: 'No major disease risks detected for current conditions.',
    riskHigh: 'High Risk',
    riskMed: 'Medium Risk',
    riskLow: 'Low Risk',
  },
  hi: {
    eyebrow: 'फील्ड इंटेलिजेंस / ०१',
    title: 'अपनी फसल की आवाज़ समझें।',
    sub: 'हर पत्ते के लिए आपके AgriScan मॉडल की मदद से एक शांत और सटीक दूसरी राय।',
    scan: 'नया स्कैन',
    history: 'स्कैन इतिहास',
    weather: 'मौसम जोखिम',
    select: 'फसल की तस्वीर चुनें',
    drop: 'या JPG, PNG, WEBP यहां छोड़ें',
    analyze: 'पत्ते का विश्लेषण',
    analyzing: 'पत्ते के संकेत पढ़ रहे हैं…',
    idle: 'आपके अगले फील्ड नोट के लिए तैयार',
    confidence: 'मॉडल विश्वास',
    recommendations: 'फील्ड नोट्स और सुझाव',
    clear: 'साफ़ करें',
    test: 'कनेक्शन जांचें',
    connecting: 'जांच हो रही है...',
    online: 'मॉडल सक्रिय',
    offline: 'मॉडल अनुपलब्ध',
    export: 'रिपोर्ट डाउनलोड करें',
    noHistory: 'कोई स्कैन नहीं मिला। आपके विश्लेषित पत्ते यहां सहेजे जाएंगे।',
    clearHistory: 'इतिहास साफ़ करें',
    primarySignal: 'मुख्य परिणाम',
    modelBreakdown: 'मॉडल विभाजन',
    inputSection: '०१ / इनपुट',
    outputSection: '०२ / परिणाम',
    actionSection: '०३ / कार्यवाही',
    weatherHeading: 'मौसम और फसल रोग जोखिम',
    weatherSub: 'वर्तमान मौसम और फसल के लिए रोग जोखिम का विश्लेषण करें।',
    getGps: 'वर्तमान स्थान प्राप्त करें',
    fetchWeather: 'मौसम जोखिम देखें',
    fetchingWeather: 'मौसम डेटा प्राप्त हो रहा है…',
    cropLabel: 'फसल चुनें',
    temp: 'तापमान',
    humidity: 'आर्द्रता',
    wind: 'हवा की गति',
    sprayAdvisory: 'छिड़काव सलाह',
    diseaseRisks: 'रोग जोखिम चेतावनी',
    noRisks: 'वर्तमान मौसम में कोई बड़ा रोग जोखिम नहीं है।',
    riskHigh: 'उच्च जोखिम',
    riskMed: 'मध्यम जोखिम',
    riskLow: 'कम जोखिम',
  },
}

const CROP_OPTIONS = [
  { value: 'Rice', labelEn: 'Rice (चावल)', labelHi: 'चावल (Rice)' },
  { value: 'Wheat', labelEn: 'Wheat (गेहूं)', labelHi: 'गेहूं (Wheat)' },
  { value: 'Tomato', labelEn: 'Tomato (टमाटर)', labelHi: 'टमाटर (Tomato)' },
  { value: 'Potato', labelEn: 'Potato (आलू)', labelHi: 'आलू (Potato)' },
  { value: 'Corn', labelEn: 'Corn / Maize (मक्का)', labelHi: 'मक्का (Corn)' },
  { value: 'Grape', labelEn: 'Grape (अंगूर)', labelHi: 'अंगूर (Grape)' },
  { value: 'Apple', labelEn: 'Apple (सेब)', labelHi: 'सेब (Apple)' },
  { value: 'Peach', labelEn: 'Peach (आड़ू)', labelHi: 'आड़ू (Peach)' },
  { value: 'Pepper', labelEn: 'Pepper / Capsicum (शिमला मिर्च)', labelHi: 'शिमला मिर्च (Pepper)' },
]

export default function Page() {
  const [lang, setLang] = useState<'en' | 'hi'>('en')
  const t = copy[lang]
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState('')
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState('')
  const [tab, setTab] = useState<'scan' | 'history' | 'weather'>('scan')
  const [isOnline, setIsOnline] = useState<boolean | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    try {
      setHistory(JSON.parse(localStorage.getItem('agriscan-history') || '[]'))
    } catch {}
    // Initial health ping
    fetch('/health')
      .then((res) => setIsOnline(res.ok))
      .catch(() => setIsOnline(false))
  }, [])

  const choose = useCallback((next: File | undefined) => {
    if (!next || !next.type.startsWith('image/')) return
    setFile(next)
    setPreview(URL.createObjectURL(next))
    setResult(null)
    setStatus('')
  }, [])

  const analyze = async () => {
    if (!file) return
    setLoading(true)
    setStatus('')
    try {
      const body = new FormData()
      body.append('image', file)
      const response = await fetch('/predict', { method: 'POST', body })
      if (!response.ok) throw new Error('The model could not read this image. Check the server.')
      const data: PredictionResponse = await response.json()
      setResult(data)

      const diseaseName =
        data.top_prediction?.display_name ||
        data.disease ||
        data.prediction ||
        data.top_prediction?.label?.replace(/___/g, ' — ').replace(/_/g, ' ') ||
        'Unknown'

      const confValue = Number(data.top_prediction?.confidence ?? data.confidence ?? 0)

      const item: HistoryItem = {
        id: crypto.randomUUID(),
        name: file.name,
        disease: String(diseaseName),
        confidence: confValue,
        time: new Date().toLocaleString(),
      }
      const next = [item, ...history].slice(0, 20)
      setHistory(next)
      localStorage.setItem('agriscan-history', JSON.stringify(next))
      setIsOnline(true)
    } catch (error) {
      setStatus(error instanceof Error ? error.message : 'Something went wrong. Check Flask connection.')
    } finally {
      setLoading(false)
    }
  }

  const primaryDisease = useMemo(() => {
    if (!result) return ''
    return (
      result.top_prediction?.display_name ||
      result.disease ||
      result.prediction ||
      result.top_prediction?.label?.replace(/___/g, ' — ').replace(/_/g, ' ') ||
      'Analysis complete'
    )
  }, [result])

  const primaryConfidence = useMemo(() => {
    if (!result) return 0
    const val = Number(result.top_prediction?.confidence ?? result.confidence ?? 0)
    return val <= 1 ? val * 100 : val
  }, [result])

  const top5 = useMemo(() => {
    if (!result?.top5) return []
    return result.top5.map((item) => {
      const dName =
        item.disease ||
        item.display_name ||
        item.label?.replace(/___/g, ' — ').replace(/_/g, ' ') ||
        'Unknown class'
      const rawConf = Number(item.confidence || 0)
      const confPct = rawConf <= 1 ? rawConf * 100 : rawConf
      return {
        disease: dName,
        confidence: confPct,
      }
    })
  }, [result])

  const recommendations = useMemo(() => {
    if (!result) return []
    return result.top_prediction?.recommendations || result.recommendations || []
  }, [result])

  const testConnection = async () => {
    setStatus(t.connecting)
    try {
      const response = await fetch('/health')
      if (response.ok) {
        setIsOnline(true)
        const info = await response.json()
        setStatus(`Connection healthy. Loaded ${info.classes ?? 57} disease classes.`)
      } else {
        setIsOnline(false)
        setStatus('Connection unavailable.')
      }
    } catch {
      setIsOnline(false)
      setStatus('Connection unavailable.')
    }
  }

  const clearHistory = () => {
    setHistory([])
    localStorage.removeItem('agriscan-history')
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <Leaf />
          </div>
          <div>
            <strong>
              Agri<span>Scan</span>
            </strong>
            <small>field intelligence</small>
          </div>
        </div>
        <div className="top-actions">
          <span className={`live ${isOnline === false ? 'offline' : ''}`}>
            <i style={{ backgroundColor: isOnline === false ? '#e25050' : '#9bd47f' }} />{' '}
            {isOnline === false ? t.offline : t.online}
          </span>
          <button
            className="language"
            onClick={() => setLang(lang === 'en' ? 'hi' : 'en')}
            aria-label="Switch language"
          >
            <Languages /> {lang === 'en' ? 'हिन्दी' : 'English'}
          </button>
          <div className="avatar">AS</div>
        </div>
      </header>

      <div className="content">
        <section className="hero">
          <div>
            <p className="eyebrow">{t.eyebrow}</p>
            <h1>{t.title}</h1>
            <p className="hero-sub">{t.sub}</p>
          </div>
          <div className="hero-signal">
            <Activity />
            <span>
              <b>57 Classes</b>
              <small>MobileNetV2 engine</small>
            </span>
          </div>
        </section>

        <nav className="tabs" aria-label="Dashboard sections">
          <button className={tab === 'scan' ? 'active' : ''} onClick={() => setTab('scan')}>
            <ScanLine /> {t.scan}
          </button>
          <button className={tab === 'history' ? 'active' : ''} onClick={() => setTab('history')}>
            <History /> {t.history}
            <em>{history.length}</em>
          </button>
          <button className={tab === 'weather' ? 'active' : ''} onClick={() => setTab('weather')}>
            <CloudRain /> {t.weather}
          </button>
        </nav>

        {tab === 'history' ? (
          <HistoryPanel history={history} onClear={clearHistory} t={t} />
        ) : tab === 'weather' ? (
          <WeatherPanel lang={lang} t={t} />
        ) : (
          <section className="dashboard-grid">
            <div className="panel upload-panel">
              <div className="panel-heading">
                <div>
                  <span className="kicker">{t.inputSection}</span>
                  <h2>Leaf image</h2>
                </div>
                <span className="format">JPG · PNG · WEBP</span>
              </div>
              <div
                className={`dropzone ${preview ? 'has-image' : ''}`}
                onClick={() => inputRef.current?.click()}
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault()
                  choose(e.dataTransfer.files[0])
                }}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
              >
                {preview ? (
                  <>
                    <img src={preview} alt="Selected crop leaf" />
                    <button
                      className="remove"
                      onClick={(e) => {
                        e.stopPropagation()
                        setFile(null)
                        setPreview('')
                        setResult(null)
                      }}
                      aria-label="Remove image"
                    >
                      <X />
                    </button>
                    <div className="image-overlay">
                      <FileImage /> {file?.name}
                    </div>
                  </>
                ) : (
                  <div className="drop-copy">
                    <div className="upload-icon">
                      <UploadCloud />
                    </div>
                    <h3>{t.select}</h3>
                    <p>{t.drop}</p>
                    <span className="browse">
                      Browse files <ArrowUpRight />
                    </span>
                  </div>
                )}
              </div>
              <input
                ref={inputRef}
                type="file"
                accept="image/*"
                hidden
                onChange={(e) => choose(e.target.files?.[0])}
              />
              <button
                className="primary-button"
                disabled={!file || loading}
                onClick={analyze}
              >
                {loading ? (
                  <>
                    <Loader2 className="spin" /> {t.analyzing}
                  </>
                ) : (
                  <>
                    <Sparkles /> {t.analyze}
                    <span>⌘ ↵</span>
                  </>
                )}
              </button>
              <div className="privacy">
                <ShieldCheck /> Processed with high accuracy MobileNetV2 transfer learning
              </div>
            </div>

            <div className="panel result-panel">
              <div className="panel-heading">
                <div>
                  <span className="kicker">{t.outputSection}</span>
                  <h2>Diagnosis</h2>
                </div>
                {result && <CheckCircle2 className="success" />}
              </div>
              {result ? (
                <div className="result-content">
                  <p className="diagnosis-label">{t.primarySignal}</p>
                  <h3>{primaryDisease}</h3>
                  <div className="confidence-row">
                    <span>{t.confidence}</span>
                    <strong>{Math.round(primaryConfidence)}%</strong>
                  </div>
                  <div className="meter">
                    <i style={{ width: `${Math.min(100, Math.max(0, primaryConfidence))}%` }} />
                  </div>

                  {top5.length > 0 && (
                    <div className="breakdown">
                      <p className="diagnosis-label">{t.modelBreakdown}</p>
                      {top5.slice(0, 5).map((item) => (
                        <div className="bar-row" key={item.disease}>
                          <span title={item.disease}>{item.disease}</span>
                          <div>
                            <i style={{ width: `${Math.min(100, item.confidence)}%` }} />
                          </div>
                          <b>{Math.round(item.confidence)}%</b>
                        </div>
                      ))}
                    </div>
                  )}

                  <div className="result-actions">
                    <button className="secondary-button" onClick={() => window.print()}>
                      <Download /> {t.export}
                    </button>
                    <button
                      className="text-button"
                      onClick={() => {
                        setResult(null)
                        setFile(null)
                        setPreview('')
                      }}
                    >
                      {t.clear}
                    </button>
                  </div>
                </div>
              ) : (
                <div className="empty-result">
                  <div className="empty-orbit">
                    <Zap />
                  </div>
                  <h3>{t.idle}</h3>
                  <p>Upload a leaf image to reveal the diagnosis and model breakdown.</p>
                </div>
              )}
              {status && (
                <p className="error" style={{ marginTop: '14px' }}>
                  <RefreshCw /> {status}
                </p>
              )}
            </div>

            <div className="panel notes-panel">
              <div className="panel-heading">
                <div>
                  <span className="kicker">{t.actionSection}</span>
                  <h2>{t.recommendations}</h2>
                </div>
                <span className="note-count">{recommendations.length} notes</span>
              </div>
              {recommendations.length > 0 ? (
                <ul>
                  {recommendations.map((note, i) => (
                    <li key={i}>
                      <span>{String(i + 1).padStart(2, '0')}</span>
                      <p>{note}</p>
                    </li>
                  ))}
                </ul>
              ) : (
                <div className="notes-empty">
                  <Leaf />
                  <p>Recommendations will appear here after analysis.</p>
                </div>
              )}
            </div>
          </section>
        )}
      </div>

      <footer>
        <span>AGRISCAN / v2.4.0</span>
        <button onClick={testConnection}>
          {t.test} <ArrowUpRight />
        </button>
        <span>57 CLASSES · FLASK + NEXT.JS</span>
      </footer>
    </main>
  )
}

function HistoryPanel({
  history,
  onClear,
  t,
}: {
  history: HistoryItem[]
  onClear: () => void
  t: typeof copy['en']
}) {
  return (
    <section className="panel history-panel">
      <div className="panel-heading">
        <div>
          <span className="kicker">ARCHIVE / RECENT</span>
          <h2>{t.history}</h2>
        </div>
        {history.length > 0 && (
          <button className="text-button" onClick={onClear} style={{ color: '#e25050' }}>
            <Trash2 style={{ width: 14, display: 'inline', marginRight: 4 }} />
            {t.clearHistory}
          </button>
        )}
      </div>
      {history.length ? (
        <div className="history-list">
          {history.map((item) => (
            <div className="history-item" key={item.id}>
              <div className="history-thumb">
                <FileImage />
              </div>
              <div>
                <strong>{item.disease}</strong>
                <span>{item.name}</span>
              </div>
              <b>{Math.round(item.confidence <= 1 ? item.confidence * 100 : item.confidence)}%</b>
              <small>{item.time}</small>
            </div>
          ))}
        </div>
      ) : (
        <div className="empty-history">
          <History />
          <p>{t.noHistory}</p>
        </div>
      )}
    </section>
  )
}

// Weather Engine & Risk Rules
const DISEASE_RULES: Record<
  string,
  Array<{ disease: string; diseaseHi: string; tempMin: number; tempMax: number; humMin: number; risk: 'High' | 'Medium' }>
> = {
  Rice: [
    { disease: 'Rice Blast', diseaseHi: 'धान का ब्लास्ट रोग', tempMin: 20, tempMax: 30, humMin: 80, risk: 'High' },
    { disease: 'Bacterial Blight', diseaseHi: 'जीवाणु झुलसा रोग', tempMin: 25, tempMax: 35, humMin: 70, risk: 'High' },
    { disease: 'Brown Spot', diseaseHi: 'भूरा धब्बा रोग', tempMin: 25, tempMax: 35, humMin: 65, risk: 'Medium' },
  ],
  Wheat: [
    { disease: 'Yellow Rust', diseaseHi: 'पीला रतुआ', tempMin: 8, tempMax: 15, humMin: 70, risk: 'High' },
    { disease: 'Brown Rust', diseaseHi: 'भूरा रतुआ', tempMin: 15, tempMax: 22, humMin: 70, risk: 'High' },
    { disease: 'Powdery Mildew', diseaseHi: 'चूर्णिल आसिता', tempMin: 15, tempMax: 20, humMin: 60, risk: 'Medium' },
    { disease: 'Fusarium Head Blight', diseaseHi: 'हेड ब्लाइट', tempMin: 20, tempMax: 30, humMin: 80, risk: 'High' },
  ],
  Tomato: [
    { disease: 'Late Blight', diseaseHi: 'पछेती झुलसा', tempMin: 10, tempMax: 25, humMin: 85, risk: 'High' },
    { disease: 'Early Blight', diseaseHi: 'अगेती झुलसा', tempMin: 24, tempMax: 35, humMin: 70, risk: 'Medium' },
    { disease: 'Septoria Leaf Spot', diseaseHi: 'सेप्टोरिया लीफ स्पॉट', tempMin: 20, tempMax: 25, humMin: 75, risk: 'Medium' },
    { disease: 'Leaf Mold', diseaseHi: 'लीफ मोल्ड', tempMin: 20, tempMax: 30, humMin: 85, risk: 'High' },
  ],
  Potato: [
    { disease: 'Late Blight', diseaseHi: 'पछेती झुलसा', tempMin: 10, tempMax: 25, humMin: 85, risk: 'High' },
    { disease: 'Early Blight', diseaseHi: 'अगेती झुलसा', tempMin: 24, tempMax: 32, humMin: 70, risk: 'Medium' },
  ],
  Corn: [
    { disease: 'Gray Leaf Spot', diseaseHi: 'ग्रे लीफ स्पॉट', tempMin: 22, tempMax: 32, humMin: 80, risk: 'High' },
    { disease: 'Northern Leaf Blight', diseaseHi: 'उत्तरी पत्ती झुलसा', tempMin: 18, tempMax: 27, humMin: 75, risk: 'Medium' },
    { disease: 'Common Rust', diseaseHi: 'कॉमन रस्ट', tempMin: 15, tempMax: 25, humMin: 70, risk: 'Medium' },
  ],
  Grape: [
    { disease: 'Black Rot', diseaseHi: 'ब्लैक रॉट', tempMin: 20, tempMax: 30, humMin: 75, risk: 'High' },
    { disease: 'Powdery Mildew', diseaseHi: 'चूर्णिल फफूंदी', tempMin: 20, tempMax: 30, humMin: 50, risk: 'Medium' },
  ],
  Apple: [
    { disease: 'Apple Scab', diseaseHi: 'सेब का स्कैब', tempMin: 10, tempMax: 24, humMin: 75, risk: 'High' },
    { disease: 'Cedar Apple Rust', diseaseHi: 'देवदार सेब जंग', tempMin: 15, tempMax: 25, humMin: 70, risk: 'Medium' },
  ],
}

function WeatherPanel({ lang, t }: { lang: 'en' | 'hi'; t: typeof copy['en'] }) {
  const [lat, setLat] = useState('23.2599')
  const [lon, setLon] = useState('77.4126')
  const [crop, setCrop] = useState('Rice')
  const [loading, setLoading] = useState(false)
  const [weather, setWeather] = useState<WeatherData | null>(null)
  const [error, setError] = useState('')

  const getGps = () => {
    if (!navigator.geolocation) {
      setError(lang === 'hi' ? 'जीपीएस समर्थित नहीं है।' : 'Geolocation not supported by browser.')
      return
    }
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setLat(pos.coords.latitude.toFixed(4))
        setLon(pos.coords.longitude.toFixed(4))
        setError('')
      },
      () => {
        setError(lang === 'hi' ? 'स्थान प्राप्त करने में विफल।' : 'Failed to retrieve GPS location.')
      }
    )
  }

  const fetchWeather = async () => {
    const latNum = parseFloat(lat)
    const lonNum = parseFloat(lon)
    if (isNaN(latNum) || isNaN(lonNum)) {
      setError(lang === 'hi' ? 'कृपया मान्य अक्षांश और देशांतर दर्ज करें।' : 'Please enter valid latitude and longitude coordinates.')
      return
    }

    setLoading(true)
    setError('')
    try {
      const meteoUrl = `https://api.open-meteo.com/v1/forecast?latitude=${latNum}&longitude=${lonNum}&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,wind_speed_10m,weather_code&wind_speed_unit=kmh&timezone=auto`
      const geoUrl = `https://nominatim.openstreetmap.org/reverse?lat=${latNum}&lon=${lonNum}&format=json`

      const [meteoRes, geoRes] = await Promise.all([
        fetch(meteoUrl),
        fetch(geoUrl).catch(() => null),
      ])

      if (!meteoRes.ok) throw new Error('Could not fetch weather data from service.')
      const meteoData = await meteoRes.json()
      let cityName = 'Current Location'
      if (geoRes && geoRes.ok) {
        try {
          const geoData = await geoRes.json()
          const a = geoData.address || {}
          cityName = a.city || a.town || a.village || a.county || a.state || 'Field Region'
        } catch {}
      }

      const c = meteoData.current
      const temp = parseFloat(c.temperature_2m.toFixed(1))
      const humidity = c.relative_humidity_2m
      const windKph = parseFloat(c.wind_speed_10m.toFixed(1))
      const rain = c.precipitation || 0
      const feelsLike = parseFloat(c.apparent_temperature.toFixed(1))

      const sprayOk = windKph < 20 && rain === 0
      const sprayAdvice = sprayOk
        ? (lang === 'hi' ? 'छिड़काव के लिए अनुकूल मौसम — हवा शांत और बारिश नहीं है' : 'Good conditions for spraying — calm winds & no rain')
        : (lang === 'hi' ? 'छिड़काव से बचें — तेज हवा या वर्षा की संभावना' : 'Avoid spraying — winds are strong or rain is present')

      // Calculate disease risk
      const rules = DISEASE_RULES[crop] || []
      const risks: WeatherRisk[] = []
      for (const r of rules) {
        if (temp >= r.tempMin && temp <= r.tempMax && humidity >= r.humMin) {
          risks.push({
            disease: lang === 'hi' ? r.diseaseHi : r.disease,
            risk: r.risk,
            reason: lang === 'hi'
              ? `तापमान ${temp}°C और आर्द्रता ${humidity}% इस रोग को बढ़ावा देते हैं`
              : `Temp ${temp}°C & Humidity ${humidity}% favour this fungal/bacterial threat`,
          })
        }
      }

      setWeather({
        city: cityName,
        temp,
        feels_like: feelsLike,
        humidity,
        wind_kph: windKph,
        condition: decodeWeatherCode(c.weather_code, lang),
        spray_ok: sprayOk,
        spray_advice: sprayAdvice,
        disease_risk: risks,
      })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to retrieve weather risk.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="panel weather-panel">
      <div className="panel-heading">
        <div>
          <span className="kicker">FIELD CONDITIONS / LIVE</span>
          <h2>{t.weatherHeading}</h2>
        </div>
        <CloudRain style={{ color: '#d5f36d' }} />
      </div>

      <p style={{ color: 'var(--muted-foreground)', fontSize: 13, marginBottom: 20 }}>
        {t.weatherSub}
      </p>

      {/* Inputs */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: 12,
          marginBottom: 18,
        }}
      >
        <div>
          <label style={{ display: 'block', fontSize: 11, color: 'var(--muted-foreground)', marginBottom: 6 }}>
            Latitude
          </label>
          <input
            type="text"
            value={lat}
            onChange={(e) => setLat(e.target.value)}
            style={{
              width: '100%',
              padding: '10px 12px',
              borderRadius: 8,
              background: '#0b110d',
              border: '1px solid var(--border)',
              color: 'var(--foreground)',
              fontSize: 13,
            }}
          />
        </div>

        <div>
          <label style={{ display: 'block', fontSize: 11, color: 'var(--muted-foreground)', marginBottom: 6 }}>
            Longitude
          </label>
          <input
            type="text"
            value={lon}
            onChange={(e) => setLon(e.target.value)}
            style={{
              width: '100%',
              padding: '10px 12px',
              borderRadius: 8,
              background: '#0b110d',
              border: '1px solid var(--border)',
              color: 'var(--foreground)',
              fontSize: 13,
            }}
          />
        </div>

        <div>
          <label style={{ display: 'block', fontSize: 11, color: 'var(--muted-foreground)', marginBottom: 6 }}>
            {t.cropLabel}
          </label>
          <select
            value={crop}
            onChange={(e) => setCrop(e.target.value)}
            style={{
              width: '100%',
              padding: '10px 12px',
              borderRadius: 8,
              background: '#0b110d',
              border: '1px solid var(--border)',
              color: 'var(--foreground)',
              fontSize: 13,
            }}
          >
            {CROP_OPTIONS.map((c) => (
              <option key={c.value} value={c.value}>
                {lang === 'hi' ? c.labelHi : c.labelEn}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 24 }}>
        <button
          className="secondary-button"
          onClick={getGps}
          type="button"
          style={{ margin: 0, padding: '10px 14px' }}
        >
          <MapPin /> {t.getGps}
        </button>
        <button
          className="primary-button"
          onClick={fetchWeather}
          disabled={loading}
          type="button"
          style={{ margin: 0, width: 'auto', padding: '10px 20px' }}
        >
          {loading ? (
            <>
              <Loader2 className="spin" /> {t.fetchingWeather}
            </>
          ) : (
            <>
              <RefreshCw /> {t.fetchWeather}
            </>
          )}
        </button>
      </div>

      {error && (
        <p className="error" style={{ marginBottom: 18 }}>
          <AlertTriangle /> {error}
        </p>
      )}

      {weather ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {/* Weather summary hero */}
          <div className="weather-hero" style={{ minHeight: 'auto', padding: 22 }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
              <div>
                <span style={{ fontSize: 11, color: 'var(--primary)', letterSpacing: '0.1em' }}>
                  📍 {weather.city}
                </span>
                <strong style={{ fontSize: 28, marginTop: 4 }}>{weather.temp}°C</strong>
                <p style={{ color: 'var(--muted-foreground)', fontSize: 13, margin: '2px 0 0' }}>
                  {weather.condition} · Feels like {weather.feels_like}°C
                </p>
              </div>
              <CloudRain style={{ width: 44, height: 44, color: 'var(--primary)' }} />
            </div>
          </div>

          {/* Quick Metrics Grid */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
              gap: 12,
            }}
          >
            <div style={{ padding: 14, borderRadius: 10, background: '#ffffff04', border: '1px solid var(--border)' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: 'var(--muted-foreground)' }}>
                <Thermometer style={{ width: 14 }} /> {t.temp}
              </span>
              <b style={{ fontSize: 18, display: 'block', marginTop: 4 }}>{weather.temp}°C</b>
            </div>
            <div style={{ padding: 14, borderRadius: 10, background: '#ffffff04', border: '1px solid var(--border)' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: 'var(--muted-foreground)' }}>
                <Droplets style={{ width: 14 }} /> {t.humidity}
              </span>
              <b style={{ fontSize: 18, display: 'block', marginTop: 4 }}>{weather.humidity}%</b>
            </div>
            <div style={{ padding: 14, borderRadius: 10, background: '#ffffff04', border: '1px solid var(--border)' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: 'var(--muted-foreground)' }}>
                <Wind style={{ width: 14 }} /> {t.wind}
              </span>
              <b style={{ fontSize: 18, display: 'block', marginTop: 4 }}>{weather.wind_kph} km/h</b>
            </div>
          </div>

          {/* Spray Advisory */}
          <div
            style={{
              padding: 16,
              borderRadius: 10,
              border: `1px solid ${weather.spray_ok ? '#9bd47f44' : '#e2505044'}`,
              background: weather.spray_ok ? 'rgba(74,200,110,0.06)' : 'rgba(226,80,80,0.06)',
              display: 'flex',
              alignItems: 'center',
              gap: 12,
            }}
          >
            {weather.spray_ok ? (
              <Check style={{ color: '#9bd47f', width: 22, height: 22, flexShrink: 0 }} />
            ) : (
              <AlertTriangle style={{ color: '#e25050', width: 22, height: 22, flexShrink: 0 }} />
            )}
            <div>
              <strong style={{ display: 'block', fontSize: 14, color: weather.spray_ok ? '#9bd47f' : '#e25050' }}>
                {t.sprayAdvisory}: {weather.spray_ok ? (lang === 'hi' ? 'अनुकूल' : 'Recommended') : (lang === 'hi' ? 'प्रतिकूल' : 'Not Recommended')}
              </strong>
              <p style={{ margin: 0, fontSize: 12, color: 'var(--foreground)' }}>
                {weather.spray_advice}
              </p>
            </div>
          </div>

          {/* Disease Risks */}
          <div style={{ marginTop: 8 }}>
            <span className="kicker" style={{ display: 'block', marginBottom: 10 }}>
              {t.diseaseRisks} ({crop})
            </span>
            {weather.disease_risk.length > 0 ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {weather.disease_risk.map((r, idx) => (
                  <div
                    key={idx}
                    style={{
                      padding: 14,
                      borderRadius: 10,
                      border: '1px solid var(--border)',
                      background: '#0b110d',
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                    }}
                  >
                    <div>
                      <strong style={{ fontSize: 14 }}>{r.disease}</strong>
                      <p style={{ margin: '3px 0 0', fontSize: 12, color: 'var(--muted-foreground)' }}>
                        {r.reason}
                      </p>
                    </div>
                    <span
                      style={{
                        padding: '4px 10px',
                        borderRadius: 100,
                        fontSize: 11,
                        fontWeight: 600,
                        backgroundColor: r.risk === 'High' ? 'rgba(226,80,80,0.15)' : 'rgba(226,160,48,0.15)',
                        color: r.risk === 'High' ? '#e25050' : '#e2a030',
                        border: `1px solid ${r.risk === 'High' ? 'rgba(226,80,80,0.4)' : 'rgba(226,160,48,0.4)'}`,
                      }}
                    >
                      {r.risk === 'High' ? t.riskHigh : t.riskMed}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <div style={{ padding: 18, textAlign: 'center', borderRadius: 10, background: '#ffffff04', color: 'var(--muted-foreground)', fontSize: 13 }}>
                <CheckCircle2 style={{ color: '#9bd47f', margin: '0 auto 6px', display: 'block' }} />
                {t.noRisks}
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="weather-hero">
          <CloudRain />
          <div>
            <strong>Ready to monitor field conditions</strong>
            <p>Select your crop and click &quot;Fetch Weather Risk&quot; to calculate disease susceptibility in real-time.</p>
          </div>
        </div>
      )}
    </section>
  )
}

function decodeWeatherCode(code: number, lang: 'en' | 'hi'): string {
  if (code === 0) return lang === 'hi' ? 'साफ़ आसमान' : 'Clear sky'
  if (code === 1 || code === 2) return lang === 'hi' ? 'आंशिक रूप से बादल' : 'Partly cloudy'
  if (code === 3) return lang === 'hi' ? 'बादल छाए रहेंगे' : 'Overcast'
  if (code >= 45 && code <= 48) return lang === 'hi' ? 'कोहरा' : 'Foggy'
  if (code >= 51 && code <= 55) return lang === 'hi' ? 'हल्की बूंदाबांदी' : 'Drizzle'
  if (code >= 61 && code <= 65) return lang === 'hi' ? 'बारिश' : 'Rain'
  if (code >= 71 && code <= 77) return lang === 'hi' ? 'बर्फबारी' : 'Snowfall'
  if (code >= 80 && code <= 82) return lang === 'hi' ? 'तेज बारिश' : 'Rain showers'
  if (code >= 95) return lang === 'hi' ? 'गरज के साथ बारिश' : 'Thunderstorm'
  return lang === 'hi' ? 'सामान्य' : 'Fair conditions'
}
