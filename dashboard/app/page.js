"use client";
import { useState, useEffect, useRef, useCallback } from "react";
import Link from "next/link";
import {
  Shield, BarChart3, DollarSign, GitBranch, Brain, TrendingUp,
  ChevronRight, Zap, Target, ArrowRight, Code2, Timer,
  Cpu, Database, Sparkles, ArrowDown,
} from "lucide-react";

/* ═══════════════════════════════════════════════════════════
   CAPTIVE SCROLL — Bidirectional fade + word-by-word reveal
   ═══════════════════════════════════════════════════════════ */

function useCaptiveScroll() {
  const ref = useRef(null);

  useEffect(() => {
    const container = ref.current;
    if (!container) return;

    const sections = Array.from(container.querySelectorAll("[data-captive]") || []);

    // Precompute word ordering per section (left→right, top→bottom)
    const computeWordOrder = () => {
      sections.forEach((sec) => {
        const words = Array.from(sec.querySelectorAll(".scroll-word") || []);
        if (!words.length) return;
        const rects = words.map((w) => ({ el: w, r: w.getBoundingClientRect() }));

        // Group by row with small tolerance, then sort rows top->bottom and within row left->right
        const rows = [];
        rects.forEach(({ el, r }) => {
          const top = Math.round(r.top);
          let row = rows.find((rg) => Math.abs(rg.top - top) <= 8);
          if (!row) {
            row = { top, items: [] };
            rows.push(row);
          }
          row.items.push({ el, left: r.left, top: r.top });
        });
        rows.sort((a, b) => a.top - b.top);
        let order = 0;
        rows.forEach((row) => {
          row.items.sort((a, b) => a.left - b.left);
          row.items.forEach((it) => {
            it.el.dataset.order = String(order++);
          });
        });
        sec.dataset.words = String(order);
      });
    };

    const update = () => {
      const vh = window.innerHeight;

      sections.forEach((el, idx) => {
        const rect = el.getBoundingClientRect();

        // visibility mapping: plateau (fully visible) + hold (stays fully visible) + fadeRange
        const plateau = vh * 0.18; // fully visible zone
        const hold = vh * 0.12; // stay fully visible for a short scroll
        const fadeRange = vh * 0.6; // range over which it fades out

        const center = rect.top + rect.height / 2;
        const viewCenter = vh / 2;
        const dist = Math.abs(center - viewCenter);

        let sectionVisibility = 0;
        if (dist <= plateau + hold) {
          sectionVisibility = 1;
        } else if (dist <= plateau + hold + fadeRange) {
          const t = (dist - (plateau + hold)) / fadeRange; // 0..1
          const eased = 1 - (t * t * (3 - 2 * t));
          sectionVisibility = Math.max(0, Math.min(1, eased));
        } else {
          sectionVisibility = 0;
        }

        // If at bottom of document, ensure last section is visible
        if (idx === sections.length - 1) {
          if (window.innerHeight + window.scrollY >= document.body.scrollHeight - 2) {
            sectionVisibility = 1;
          }
        }

        el.style.opacity = String(sectionVisibility);
        el.style.transform = `translateY(${(1 - sectionVisibility) * 30}px) scale(${0.97 + sectionVisibility * 0.03})`;

        // Word reveal inside this section — left→right, top→bottom order
        const totalWords = parseInt(el.dataset.words || "0", 10);
        const words = Array.from(el.querySelectorAll(".scroll-word") || []);
        words.forEach((w) => {
          const order = parseInt(w.dataset.order || "0", 10);

          // Spread reveal across words with small per-word delay; map sectionVisibility to word final
          const per = 0.04; // fraction per word for stagger
          const revealStart = order * per;
          const revealSpan = Math.max(0.16, 1 - totalWords * per); // ensure reasonable span
          const local = (sectionVisibility - revealStart) / revealSpan;
          const final = Math.max(0, Math.min(1, local));

          w.style.opacity = String(final);
          w.style.transform = `translateY(${(1 - final) * 8}px)`;
        });
      });
    };

    // initial compute once images/fonts/layout settled
    computeWordOrder();
    // recompute on resize since layout may change
    window.addEventListener("resize", computeWordOrder);

    window.addEventListener("scroll", update, { passive: true });
    window.addEventListener("resize", update);
    update(); // initial
    return () => {
      window.removeEventListener("scroll", update);
      window.removeEventListener("resize", update);
      window.removeEventListener("resize", computeWordOrder);
    };
  }, []);

  return ref;
}

/* ── Word Reveal Text ─────────────────────────────────────── */
function ScrollText({ text, className = "" }) {
  const wordsArr = text.split(" ");
  return (
    <span className={className}>
      {wordsArr.map((word, i) => (
        <span key={i} className="scroll-word inline-block mr-[0.3em]" data-idx={i}>
          {word}
        </span>
      ))}
    </span>
  );
}

/* ── Animated Counter ─────────────────────────────────────── */
function Counter({ end, prefix = "", suffix = "", decimals = 0 }) {
  const [val, setVal] = useState(0);
  const [started, setStarted] = useState(false);
  const ref = useRef(null);
  useEffect(() => {
    const obs = new IntersectionObserver(([e]) => { if (e.isIntersecting) setStarted(true); }, { threshold: 0.5 });
    if (ref.current) obs.observe(ref.current);
    return () => obs.disconnect();
  }, []);
  useEffect(() => {
    if (!started) return;
    let t0;
    const step = (ts) => {
      if (!t0) t0 = ts;
      const p = Math.min((ts - t0) / 2000, 1);
      setVal((1 - Math.pow(1 - p, 3)) * end);
      if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }, [started, end]);
  return <span ref={ref}>{prefix}{val.toFixed(decimals).replace(/\B(?=(\d{3})+(?!\d))/g, ",")}{suffix}</span>;
}

/* ── Floating Particle ────────────────────────────────────── */
function Particle({ size, color, top, left, delay, dur }) {
  return <div className="absolute rounded-full opacity-20 animate-float" style={{ width: size, height: size, background: color, top, left, animationDelay: `${delay}s`, animationDuration: `${dur}s`, filter: size > 8 ? "blur(2px)" : "none" }} />;
}

function LatchedHover({ children, className = "", activeClassName = "", holdMs = 520, style = {} }) {
  const [active, setActive] = useState(false);
  const timerRef = useRef(null);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  const begin = () => {
    if (timerRef.current) clearTimeout(timerRef.current);
    setActive(true);
  };

  const end = () => {
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => setActive(false), holdMs);
  };

  return (
    <div
      className={`${className} ${active ? activeClassName : ""}`}
      style={style}
      onMouseEnter={begin}
      onMouseLeave={end}
      onFocus={begin}
      onBlur={end}
    >
      {children(active)}
    </div>
  );
}

/* ── Tech Data ────────────────────────────────────────────── */
const TECH = [
  { name: "Python", icon: Code2, color: "#3B82F6", desc: "Core Pipeline" },
  { name: "lifelines", icon: BarChart3, color: "#6366F1", desc: "Survival Models" },
  { name: "scikit-learn", icon: Brain, color: "#F59E0B", desc: "ML Models" },
  { name: "Tree-sitter", icon: GitBranch, color: "#8B5CF6", desc: "AST Parser" },
  { name: "SonarQube", icon: Shield, color: "#EF4444", desc: "Code Quality" },
  { name: "Next.js", icon: Zap, color: "#10B981", desc: "Frontend" },
  { name: "FastAPI", icon: Cpu, color: "#EC4899", desc: "API Server" },
  { name: "Recharts", icon: TrendingUp, color: "#F59E0B", desc: "Data Viz" },
  { name: "Tailwind", icon: Sparkles, color: "#06B6D4", desc: "Styling" },
  { name: "SQLite", icon: Database, color: "#8B5CF6", desc: "Storage" },
];

/* ══════════════════════════════════════════════════════════
   Landing Page — Captive Scroll Experience
   ══════════════════════════════════════════════════════════ */
export default function LandingPage() {
  const containerRef = useCaptiveScroll();
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const h = (e) => setMousePos({ x: (e.clientX / window.innerWidth - 0.5) * 20, y: (e.clientY / window.innerHeight - 0.5) * 20 });
    window.addEventListener("mousemove", h);
    return () => window.removeEventListener("mousemove", h);
  }, []);

  return (
    <div ref={containerRef} className="min-h-screen relative overflow-x-hidden">
      {/* BG */}
      <div className="fixed inset-0 hero-gradient pointer-events-none" />
      <div className="fixed inset-0 grid-pattern pointer-events-none opacity-30" />
      {/* Top/bottom viewport fade masks */}
      <div className="fixed top-0 left-0 right-0 h-32 z-[60] pointer-events-none" style={{ background: "linear-gradient(to bottom, var(--background), transparent)" }} />
      <div className="fixed bottom-0 left-0 right-0 h-32 z-[60] pointer-events-none" style={{ background: "linear-gradient(to top, var(--background), transparent)" }} />
      {/* Particles */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <Particle size={6} color="#6366F1" top="15%" left="10%" delay={0} dur={7} />
        <Particle size={4} color="#EC4899" top="25%" left="80%" delay={1} dur={8} />
        <Particle size={8} color="#8B5CF6" top="60%" left="15%" delay={2} dur={6} />
        <Particle size={5} color="#10B981" top="70%" left="75%" delay={0.5} dur={9} />
        <Particle size={10} color="#6366F1" top="85%" left="50%" delay={1.5} dur={10} />
      </div>

      {/* ── Navbar ─────────────────────────────────────────── */}
      <nav className="relative z-[70] flex items-center justify-between px-8 py-5 max-w-7xl mx-auto">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[var(--accent)] to-[var(--gradient-end)] flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <Shield size={20} className="text-white" />
          </div>
          <span className="text-xl font-bold text-[var(--text-primary)]">Code<span className="gradient-text">Survival</span></span>
        </div>
        <div className="flex items-center gap-4">
          <Link href="/login" className="text-sm font-medium text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors">Log In</Link>
          <Link href="/login" className="btn-primary text-sm py-2.5 px-6">Get Started</Link>
        </div>
      </nav>

      {/* ── S1: Hero ───────────────────────────────────────── */}
      <section data-captive className="relative z-10 pt-24 pb-40 px-8 max-w-7xl mx-auto text-center" style={{ transition: "opacity 0.1s, transform 0.1s" }}>
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-[var(--card-border)] bg-[var(--card-bg)] text-sm text-[var(--text-secondary)] mb-8 animate-shimmer">
          <Zap size={14} className="text-[var(--warning)]" /> Built on 154K+ commits from 31 Apache Java projects
        </div>
        <div style={{ transform: `translate(${mousePos.x * 0.3}px, ${mousePos.y * 0.3}px)` }}>
          <h1 className="text-5xl md:text-7xl font-extrabold leading-tight mb-6">
            <span className="text-[var(--text-primary)]">Predict When Your Code</span>
            <br />
            <span className="gradient-text text-glow">Will Fail</span>
            <span className="text-[var(--text-primary)]"> — Before It Does</span>
          </h1>
        </div>
        <p className="text-lg md:text-xl max-w-3xl mx-auto mb-12 leading-relaxed">
          <ScrollText text="Combines survival analysis, machine learning, and AST parsing to transform static code metrics into financially actionable refactoring recommendations." className="text-[var(--text-secondary)]" />
        </p>
        <div className="flex items-center justify-center gap-4 mb-8">
          <Link href="/login" className="btn-primary inline-flex items-center gap-2 text-base py-3.5 px-8 glow-accent">Explore Dashboard <ArrowRight size={18} /></Link>
          <a href="https://github.com/Chiron-R/code-survival-intelligence" target="_blank" rel="noopener noreferrer" className="btn-secondary inline-flex items-center gap-2 text-base py-3.5 px-8"><GitBranch size={18} /> View Source</a>
        </div>
        <div className="mt-16">
          <div className="flex flex-col items-center gap-2 text-[var(--text-muted)] animate-float">
            <span className="text-xs uppercase tracking-widest">Scroll to explore</span>
            <ArrowDown size={16} />
          </div>
        </div>
      </section>

      {/* ── S2: KPIs ───────────────────────────────────────── */}
      <section data-captive className="relative z-10 py-20 px-8 max-w-5xl mx-auto" style={{ transition: "opacity 0.1s, transform 0.1s, filter 0.1s" }}>
        <div className="text-center mb-12">
          <ScrollText text="The Numbers That Matter" className="text-3xl md:text-4xl font-bold text-[var(--text-primary)]" />
        </div>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-5">
          {[
            { label: "C-Index", value: <Counter end={0.80} decimals={2} />, icon: Target, color: "var(--accent)", sub: "Concordance" },
            { label: "Net Savings", value: <Counter end={3251231} prefix="$" />, icon: DollarSign, color: "var(--success)", sub: "1-Year" },
            { label: "Files", value: <Counter end={37102} />, icon: Code2, color: "var(--info)", sub: "Analyzed" },
            { label: "Positive ROI", value: <Counter end={47.4} suffix="%" decimals={1} />, icon: TrendingUp, color: "var(--warning)", sub: "17,597 Files" },
          ].map((kpi) => (
            <div key={kpi.label} className="kpi-card text-left">
              <div className="flex items-center gap-3 mb-3">
                <div className="w-9 h-9 rounded-lg flex items-center justify-center" style={{ background: `color-mix(in srgb, ${kpi.color} 12%, transparent)` }}>
                  <kpi.icon size={18} style={{ color: kpi.color }} />
                </div>
                <span className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">{kpi.label}</span>
              </div>
              <div className="text-3xl font-bold text-[var(--text-primary)] mb-1">{kpi.value}</div>
              <span className="text-xs text-[var(--text-muted)]">{kpi.sub}</span>
            </div>
          ))}
        </div>
      </section>

      {/* ── S3: How It Works ───────────────────────────────── */}
      <section data-captive className="relative z-10 py-28 px-8 max-w-6xl mx-auto" style={{ transition: "opacity 0.1s, transform 0.1s, filter 0.1s" }}>
        <div className="text-center mb-16 section-header-shell">
          <span className="section-chip section-chip-accent">Pipeline</span>
          <h2 className="mt-4 mb-4"><ScrollText text="How It Works" className="text-3xl md:text-5xl font-bold text-[var(--text-primary)]" /></h2>
          <p className="max-w-2xl mx-auto"><ScrollText text="A multi-phase pipeline from raw code to dollar-value refactoring priorities." className="text-[var(--text-secondary)]" /></p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 relative">
          <div className="hidden md:block absolute top-[40px] left-[20%] right-[20%] h-[2px] z-0" style={{ background: "linear-gradient(90deg, transparent, rgba(255,255,255,0.15), rgba(255,255,255,0.15), transparent)" }} />
          {[
            { n: 1, icon: Code2, title: "Extract Features", desc: "16 database features + 20 AST structural features via Tree-sitter parsing at historical commits.", color: "#6366F1" },
            { n: 2, icon: Brain, title: "Predict Failure", desc: "Cox PH model predicts when code will fail. Survival curves at 90, 180, 365, 730 day horizons.", color: "#EC4899" },
            { n: 3, icon: DollarSign, title: "Calculate ROI", desc: "Convert failure probabilities into dollar-value Expected Loss and ROI for every file.", color: "#10B981" },
          ].map((s) => (
            <LatchedHover
              key={s.n}
              className="text-center relative z-10 group cursor-pointer"
              activeClassName="is-hover-latched"
              holdMs={520}
            >
              {(active) => (
                <>
                  <div className="relative mx-auto mb-6 transition-all duration-300">
                    <div 
                      className={`w-20 h-20 rounded-2xl flex items-center justify-center mx-auto shadow-xl relative z-10 hover-wobble-glow icon-glow-card ${active ? "latched-icon-active" : ""}`} 
                      style={{ 
                        background: `linear-gradient(135deg, ${s.color}, ${s.color}cc)`, 
                        boxShadow: `0 12px 40px ${s.color}30`,
                        '--glow-color': `${s.color}80` 
                      }}
                    >
                      <s.icon size={32} className="text-white" />
                    </div>
                    <div className={`absolute -top-2 -right-2 w-8 h-8 rounded-full bg-[var(--background)] border-2 flex items-center justify-center text-sm font-bold z-20 transition-transform duration-300 ${active ? "scale-110" : ""}`} style={{ borderColor: s.color, color: s.color }}>{s.n}</div>
                  </div>
                  <h3 className="text-lg font-semibold text-[var(--text-primary)] mb-2 transition-colors duration-300" style={{ textShadow: `0 0 10px ${s.color}00` }}>{s.title}</h3>
                  <p className="text-sm text-[var(--text-secondary)] leading-relaxed max-w-xs mx-auto">{s.desc}</p>
                </>
              )}
            </LatchedHover>
          ))}
        </div>
      </section>

      {/* ── S4: Features ───────────────────────────────────── */}
      <section data-captive className="relative z-10 py-28 px-8 max-w-6xl mx-auto" style={{ transition: "opacity 0.1s, transform 0.1s, filter 0.1s" }}>
        <div className="text-center mb-16 section-header-shell">
          <span className="section-chip section-chip-mid">Capabilities</span>
          <h2 className="mt-4 mb-4"><ScrollText text="Powerful Analytics" className="text-3xl md:text-5xl font-bold text-[var(--text-primary)]" /></h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          {[
            { icon: BarChart3, title: "Survival Analysis", desc: "Cox PH model with C-index 0.80. Interactive Kaplan-Meier curves.", color: "#6366F1" },
            { icon: DollarSign, title: "Financial ROI", desc: "$3.25M net savings identified across 37,102 files.", color: "#10B981" },
            { icon: Target, title: "Risk Tiering", desc: "Auto classification: Critical, High, Medium, Low tiers.", color: "#EF4444" },
            { icon: GitBranch, title: "AST Parsing", desc: "Tree-sitter extracts 20 structural features at fault-inducing commits.", color: "#8B5CF6" },
            { icon: Brain, title: "Model Comparison", desc: "Cox PH vs Random Forest vs Logistic Regression side-by-side.", color: "#F59E0B" },
            { icon: Timer, title: "Multi-Horizon", desc: "Predictions at 90, 180, 365, and 730 day failure horizons.", color: "#EC4899" },
          ].map((f) => (
            <LatchedHover key={f.title} className="feature-card-wrap" activeClassName="is-hover-latched" holdMs={520}>
              {(active) => (
                <div className={`glass-card glass-card-lift p-6 group cursor-pointer feature-card ${active ? "feature-card-active" : ""}`}>
                  <div className={`w-14 h-14 rounded-2xl flex items-center justify-center mb-5 transition-transform duration-300 hover-wobble-glow icon-glow-card ${active ? "latched-icon-active scale-110 rotate-3" : ""}`} style={{ background: `${f.color}15`, border: `1px solid ${f.color}30`, boxShadow: `inset 0 0 18px ${f.color}12`, '--glow-color': `${f.color}80` }}>
                    <f.icon size={26} style={{ color: f.color, filter: `drop-shadow(0 0 10px ${f.color}55)` }} />
                  </div>
                  <h3 className="text-lg font-semibold text-[var(--text-primary)] mb-2">{f.title}</h3>
                  <p className="text-sm text-[var(--text-secondary)] leading-relaxed">{f.desc}</p>
                </div>
              )}
            </LatchedHover>
          ))}
        </div>
      </section>

      {/* ── S5: Benchmarks ─────────────────────────────────── */}
      <section data-captive className="relative z-10 py-28 px-8 max-w-5xl mx-auto" style={{ transition: "opacity 0.1s, transform 0.1s, filter 0.1s" }}>
        <div className="text-center mb-16 section-header-shell">
          <span className="section-chip section-chip-success">Benchmarks</span>
          <h2 className="mt-4 mb-4"><ScrollText text="Model Performance" className="text-3xl md:text-5xl font-bold text-[var(--text-primary)]" /></h2>
        </div>
        <div className="glass-card overflow-hidden benchmark-shell">
          <table className="data-table">
            <thead><tr><th>Model</th><th>C-Index</th><th>AUC-ROC</th><th>Brier</th><th>Verdict</th></tr></thead>
            <tbody>
              <tr className="bg-[rgba(99,102,241,0.05)]"><td className="font-semibold text-[var(--text-primary)]">Cox PH</td><td className="font-mono text-[var(--accent-hover)]">0.80</td><td className="font-mono">0.660</td><td className="font-mono">0.083</td><td><span className="badge badge-low">★ Best</span></td></tr>
              <tr><td className="text-[var(--text-primary)]">Logistic Regression</td><td className="font-mono">—</td><td className="font-mono">0.643</td><td className="font-mono">0.122</td><td><span className="badge badge-medium">Good</span></td></tr>
              <tr><td className="text-[var(--text-primary)]">Random Forest</td><td className="font-mono">—</td><td className="font-mono">0.569</td><td className="font-mono">0.305</td><td><span className="badge badge-high">Weak</span></td></tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* ── S6: Built With ─────────────────────────────────── */}
      <section data-captive className="relative z-10 py-32 px-8 max-w-7xl mx-auto" style={{ transition: "opacity 0.1s, transform 0.1s" }}>
        <div className="text-center mb-16">
          <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-[var(--card-border)] bg-[var(--card-bg)] text-xs uppercase tracking-widest text-[var(--accent)] font-semibold mb-6 shadow-lg">
            <Sparkles size={14} className="text-[var(--accent)]" /> Technology Stack
          </span>
          <h2 className="mt-2 mb-4"><span className="text-4xl md:text-5xl font-bold text-[var(--text-primary)]">Powered by Modern Tech</span></h2>
          <p className="text-[var(--text-secondary)] max-w-2xl mx-auto text-lg">An enterprise-grade stack combining robust data pipelines, machine learning, and interactive data visualization.</p>
        </div>
        
        <div className="grid grid-cols-2 md:grid-cols-5 gap-6">
          {TECH.map((t, i) => (
            <LatchedHover key={t.name} className="relative group cursor-pointer tech-tile-wrap" activeClassName="is-hover-latched" holdMs={520} style={{ transitionDelay: `${i * 50}ms` }}>
              {(active) => (
                <div className={`relative z-10 glass-card p-6 flex flex-col items-center justify-center text-center h-full transform transition-all duration-500 tech-pill-card ${active ? "tech-pill-card-active -translate-y-2 border-opacity-50" : "group-hover:-translate-y-2 group-hover:border-opacity-50"}`}
                  style={{ borderColor: `color-mix(in srgb, ${t.color} 20%, var(--card-border))` }}>
                  {/* Animated Glow Background */}
                  <div className={`absolute inset-0 rounded-[24px] bg-gradient-to-br transition-opacity duration-500 blur-xl z-0 ${active ? "opacity-100" : "opacity-0 group-hover:opacity-100"}`} 
                    style={{ backgroundImage: `linear-gradient(to bottom right, ${t.color}40, transparent)` }} />
                  
                  <div className={`w-16 h-16 rounded-2xl mx-auto mb-4 flex items-center justify-center transition-all duration-500 hover-wobble-glow ${active ? "latched-icon-active scale-110" : ""}`} 
                    style={{ background: `linear-gradient(135deg, ${t.color}15, transparent)`, border: `1px solid ${t.color}40`, boxShadow: `inset 0 0 20px ${t.color}10`, '--glow-color': `${t.color}80` }}>
                    <t.icon size={28} className="transition-all duration-500" style={{ color: t.color, filter: `drop-shadow(0 0 8px ${t.color}60)` }} />
                  </div>
                  
                  <h4 className="text-base font-bold mb-1 relative">
                    <span className={`text-[var(--text-primary)] transition-opacity duration-300 ${active ? "opacity-0" : "group-hover:opacity-0"}`}>{t.name}</span>
                    <span className={`absolute left-0 right-0 top-0 transition-opacity duration-300 bg-clip-text text-transparent ${active ? "opacity-100" : "opacity-0 group-hover:opacity-100"}`}
                      style={{ backgroundImage: `linear-gradient(to right, #fff, ${t.color})` }}>{t.name}</span>
                  </h4>
                  <p className="text-xs text-[var(--text-secondary)] font-medium">{t.desc}</p>
                  
                  {/* Decorative dot */}
                  <div className={`absolute top-3 right-3 w-1.5 h-1.5 rounded-full transition-opacity duration-300 ${active ? "opacity-100" : "opacity-0 group-hover:opacity-100"}`} style={{ background: t.color, boxShadow: `0 0 10px ${t.color}` }} />
                </div>
              )}
            </LatchedHover>
          ))}
        </div>
      </section>

      {/* ── S7: CTA ────────────────────────────────────────── */}
      <section data-captive className="relative z-10 py-28 px-8 max-w-4xl mx-auto text-center" style={{ transition: "opacity 0.1s, transform 0.1s, filter 0.1s" }}>
        <div className="glass-card p-14 glow-accent">
          <Sparkles size={32} className="text-[var(--accent)] mx-auto mb-4" />
          <h2 className="mb-4"><ScrollText text="Ready to Explore?" className="text-3xl md:text-4xl font-bold text-[var(--text-primary)]" /></h2>
          <p className="mb-8"><ScrollText text="Dive into the interactive dashboard to see survival curves, ROI scores, and refactoring priorities for 37,000+ Java files." className="text-[var(--text-secondary)]" /></p>
          <Link href="/login" className="btn-primary inline-flex items-center gap-2 text-lg py-4 px-10">Launch Dashboard <ChevronRight size={20} /></Link>
        </div>
      </section>

      {/* ── Footer ─────────────────────────────────────────── */}
      <footer className="relative z-10 py-12 px-8 border-t border-[var(--card-border)]">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-[var(--accent)] to-[var(--gradient-end)] flex items-center justify-center"><Shield size={16} className="text-white" /></div>
            <span className="text-sm font-semibold text-[var(--text-primary)]">Code Survival Intelligence</span>
          </div>
          <p className="text-xs text-[var(--text-muted)]">Final Year Project • Lenarduzzi et al. Technical Debt Dataset V2 • MIT</p>
          <a href="https://github.com/Chiron-R/code-survival-intelligence" target="_blank" rel="noopener noreferrer" className="text-sm text-[var(--text-secondary)] hover:text-[var(--accent)] transition-colors">GitHub →</a>
        </div>
      </footer>
    </div>
  );
}
