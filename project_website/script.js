// ROV Seabed Inspection — site interactions

// 1 · sticky nav background on scroll
const nav = document.getElementById('nav');
const onScroll = () => nav.classList.toggle('scrolled', window.scrollY > 24);
onScroll();
window.addEventListener('scroll', onScroll, { passive: true });

// 2 · reveal-on-scroll
const io = new IntersectionObserver(
  (entries) => {
    for (const e of entries) {
      if (e.isIntersecting) {
        e.target.classList.add('in');
        io.unobserve(e.target);
      }
    }
  },
  { threshold: 0.12, rootMargin: '0px 0px -8% 0px' }
);
document.querySelectorAll('.reveal').forEach((el) => io.observe(el));

// 3 · animated stat counters
const animateCount = (el) => {
  const target = parseInt(el.dataset.count, 10);
  if (target === 0) { el.textContent = '0'; return; }
  const duration = 1400;
  const start = performance.now();
  const tick = (now) => {
    const p = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - p, 3); // easeOutCubic
    el.textContent = Math.round(eased * target).toString();
    if (p < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
};
const countIO = new IntersectionObserver(
  (entries) => {
    for (const e of entries) {
      if (e.isIntersecting) {
        animateCount(e.target);
        countIO.unobserve(e.target);
      }
    }
  },
  { threshold: 0.6 }
);
document.querySelectorAll('.stat__num').forEach((el) => countIO.observe(el));

// 4 · animated pipeline player
const player = document.getElementById('pipeline');
if (player) {
  const scenes = [...player.querySelectorAll('.scene')];
  const steps = [...player.querySelectorAll('.step')];
  const bars = steps.map((s) => s.querySelector('.step__bar'));
  const DURATIONS = [5400, 5600, 5000];
  const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  let current = -1;
  let timer = null;
  let startedAt = 0;
  let remaining = 0;
  let started = false;

  const setBar = (i, anim) => {
    const bar = bars[i];
    bar.style.animation = 'none';
    void bar.offsetWidth; // reflow → restart
    bar.style.animation = anim;
  };

  const schedule = (ms) => {
    clearTimeout(timer);
    if (prefersReduced) return;
    startedAt = performance.now();
    remaining = ms;
    timer = setTimeout(() => go((current + 1) % scenes.length), ms);
  };

  function go(i) {
    current = i;
    scenes.forEach((s, k) => s.classList.toggle('active', k === i));
    steps.forEach((s, k) => s.classList.toggle('current', k === i));
    bars.forEach((b, k) => { if (k !== i) b.style.animation = 'none'; });
    if (!prefersReduced) setBar(i, `fill ${DURATIONS[i]}ms linear forwards`);
    schedule(DURATIONS[i]);
  }

  // start once the player scrolls into view (keeps the first animation in sync)
  const startIO = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (e.isIntersecting && !started) {
          started = true;
          go(0);
          startIO.disconnect();
        }
      }
    },
    { threshold: 0.18 }
  );
  startIO.observe(player);

  // manual step selection
  steps.forEach((s, i) => s.addEventListener('click', () => { started = true; go(i); }));

  // pause while hovered/focused so the reader can take it in
  const pause = () => {
    if (prefersReduced || current < 0) return;
    clearTimeout(timer);
    remaining -= performance.now() - startedAt;
    bars[current].style.animationPlayState = 'paused';
  };
  const resume = () => {
    if (prefersReduced || current < 0) return;
    bars[current].style.animationPlayState = 'running';
    schedule(Math.max(600, remaining));
  };
  player.addEventListener('mouseenter', pause);
  player.addEventListener('mouseleave', resume);
  player.addEventListener('focusin', pause);
  player.addEventListener('focusout', resume);
}

// 5 · floating bioluminescent particles
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const particleHost = document.getElementById('particles');
if (particleHost && !reduceMotion) {
  const COUNT = 28;
  for (let i = 0; i < COUNT; i++) {
    const p = document.createElement('span');
    const size = 1 + Math.random() * 3;
    p.style.left = Math.random() * 100 + 'vw';
    p.style.width = p.style.height = size + 'px';
    p.style.opacity = (0.15 + Math.random() * 0.35).toFixed(2);
    p.style.animationDuration = 14 + Math.random() * 26 + 's';
    p.style.animationDelay = -Math.random() * 30 + 's';
    particleHost.appendChild(p);
  }
}
