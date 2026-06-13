/* ===== PlantGuard-AI — marketing site interactions ===== */
(function () {
  "use strict";

  const nav = document.getElementById("nav");
  const hero = document.getElementById("hero");
  const burger = document.getElementById("burger");
  const links = document.querySelector(".nav__links");

  /* Nav: solid background after scroll + "on hero" light styling */
  function onScroll() {
    const scrolled = window.scrollY > 20;
    nav.classList.toggle("scrolled", scrolled);

    if (hero) {
      const heroBottom = hero.offsetTop + hero.offsetHeight - 80;
      nav.classList.toggle("at-hero", window.scrollY < heroBottom && !scrolled);
    }
  }
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  /* Mobile menu toggle */
  if (burger && links) {
    burger.addEventListener("click", () => links.classList.toggle("open"));
    links.querySelectorAll("a").forEach((a) =>
      a.addEventListener("click", () => links.classList.remove("open"))
    );
  }

  /* Animated stat counters (run once when hero stats enter view) */
  function animateCount(el) {
    const target = parseFloat(el.dataset.count);
    const suffix = el.dataset.suffix || "";
    const isFloat = !Number.isInteger(target);
    const duration = 1400;
    const start = performance.now();

    function tick(now) {
      const p = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - p, 3); // easeOutCubic
      const val = target * eased;
      el.textContent = (isFloat ? val.toFixed(2) : Math.round(val)) + suffix;
      if (p < 1) requestAnimationFrame(tick);
      else el.textContent = (isFloat ? target.toFixed(2) : target) + suffix;
    }
    requestAnimationFrame(tick);
  }

  const counters = document.querySelectorAll(".stat__num");
  if ("IntersectionObserver" in window && counters.length) {
    const seen = new WeakSet();
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting && !seen.has(e.target)) {
            seen.add(e.target);
            animateCount(e.target);
          }
        });
      },
      { threshold: 0.6 }
    );
    counters.forEach((c) => io.observe(c));
  } else {
    counters.forEach((c) => (c.textContent = c.dataset.count + (c.dataset.suffix || "")));
  }

  /* Reveal-on-scroll for sections */
  const revealEls = document.querySelectorAll(
    ".step, .feature, .result-card, .agent, .how__visual, .demo__panel"
  );
  revealEls.forEach((el) => el.classList.add("reveal"));
  if ("IntersectionObserver" in window) {
    const ro = new IntersectionObserver(
      (entries, obs) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            e.target.classList.add("in");
            obs.unobserve(e.target);
          }
        });
      },
      { threshold: 0.15 }
    );
    revealEls.forEach((el) => ro.observe(el));
  } else {
    revealEls.forEach((el) => el.classList.add("in"));
  }

  /* Demo "Diagnose" button: small playful re-run animation */
  const demoRun = document.getElementById("demoRun");
  const demoOutput = document.getElementById("demoOutput");
  const placeholder = document.querySelector(".demo__placeholder");
  if (demoRun && demoOutput) {
    demoRun.addEventListener("click", () => {
      if (placeholder) placeholder.classList.add("is-active");
      demoRun.textContent = "🔬 Analyzing…";
      demoRun.disabled = true;
      demoOutput.style.opacity = "0.35";
      demoOutput.style.transition = "opacity .25s ease";
      setTimeout(() => {
        demoOutput.style.opacity = "1";
        demoRun.textContent = "🔍 Diagnose";
        demoRun.disabled = false;
        demoOutput.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }, 1100);
    });
  }

  /* "Try the demo" nav CTA -> jump to demo section */
  const navCta = document.getElementById("navCta");
  if (navCta) {
    navCta.addEventListener("click", (e) => {
      e.preventDefault();
      document.getElementById("demo").scrollIntoView({ behavior: "smooth" });
    });
  }
})();
