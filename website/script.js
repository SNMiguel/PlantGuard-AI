/* ===== PlantGuard AI landing interactions ===== */
(function () {
  "use strict";

  /* ---- mobile sidebar ---- */
  var sidebar  = document.getElementById("sidebar");
  var burger   = document.getElementById("burger");
  var backdrop = document.getElementById("backdrop");

  function closeNav() {
    sidebar && sidebar.classList.remove("is-open");
    backdrop && backdrop.classList.remove("is-open");
  }
  if (burger) {
    burger.addEventListener("click", function () {
      var open = sidebar.classList.toggle("is-open");
      backdrop.classList.toggle("is-open", open);
    });
  }
  if (backdrop) backdrop.addEventListener("click", closeNav);

  /* ---- active nav highlight + close on click ---- */
  var navItems = Array.prototype.slice.call(document.querySelectorAll(".snav__item"));
  navItems.forEach(function (a) {
    a.addEventListener("click", function () {
      navItems.forEach(function (i) { i.classList.remove("is-active"); });
      a.classList.add("is-active");
      closeNav();
    });
  });

  var sections = navItems
    .map(function (a) {
      var id = a.getAttribute("href");
      return id && id.charAt(0) === "#" ? document.querySelector(id) : null;
    })
    .filter(Boolean);

  // Scroll-position spy: pick the last section whose top has crossed 35% of the
  // viewport. Deterministic (no IntersectionObserver timing/thin-band quirks), so
  // the highlighted item always matches the section actually in view.
  function currentSectionId() {
    var line = window.innerHeight * 0.35;
    var atBottom = window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2;
    if (atBottom && sections.length) return sections[sections.length - 1].id; // ensure last item lights at page end
    var pick = sections.length ? sections[0].id : null;
    for (var i = 0; i < sections.length; i++) {
      if (sections[i].getBoundingClientRect().top <= line) pick = sections[i].id;
    }
    return pick;
  }
  function syncSpy() {
    var id = "#" + currentSectionId();
    navItems.forEach(function (a) {
      a.classList.toggle("is-active", a.getAttribute("href") === id);
    });
  }
  var spyTick = false;
  function onScroll() {
    if (spyTick) return;
    spyTick = true;
    requestAnimationFrame(function () { spyTick = false; syncSpy(); });
  }
  if (sections.length) {
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", syncSpy);
    syncSpy();
  }

  /* ---- reveal on scroll ---- */
  var reveals = document.querySelectorAll(".reveal");
  if ("IntersectionObserver" in window) {
    var ro = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) { e.target.classList.add("is-in"); ro.unobserve(e.target); }
      });
    }, { threshold: 0.15 });
    reveals.forEach(function (el) { ro.observe(el); });
  } else {
    reveals.forEach(function (el) { el.classList.add("is-in"); });
  }

  /* ---- count-up stats ---- */
  function animateCount(el) {
    var target = parseFloat(el.getAttribute("data-count"));
    var suffix = el.getAttribute("data-suffix") || "";
    var decimals = (String(target).split(".")[1] || "").length;
    var start = null, dur = 1400;
    function tick(ts) {
      if (start === null) start = ts;
      var p = Math.min((ts - start) / dur, 1);
      var eased = 1 - Math.pow(1 - p, 3);
      el.textContent = (target * eased).toFixed(decimals) + suffix;
      if (p < 1) requestAnimationFrame(tick);
      else el.textContent = target.toFixed(decimals) + suffix;
    }
    requestAnimationFrame(tick);
  }
  var nums = document.querySelectorAll(".hstat__num");
  if ("IntersectionObserver" in window) {
    var co = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) { animateCount(e.target); co.unobserve(e.target); }
      });
    }, { threshold: 0.6 });
    nums.forEach(function (n) { co.observe(n); });
  } else {
    nums.forEach(animateCount);
  }

  /* ---- live scan: upload a leaf, call /api/diagnose, render on-brand ---- */
  (function () {
    var scan      = document.getElementById("scan");
    var offline   = document.getElementById("scanOffline");
    var form      = document.getElementById("scanForm");
    if (!form || !scan) return;

    var fileInput = document.getElementById("scanFile");
    var dropzone  = document.getElementById("dropzone");
    var dzEmpty   = document.getElementById("dzEmpty");
    var dzPreview = document.getElementById("dzPreview");
    var acres     = document.getElementById("scanAcres");
    var btn       = document.getElementById("scanBtn");
    var output    = document.getElementById("scanOutput");
    var retry     = document.getElementById("scanRetry");

    var selectedFile = null;

    function esc(s) {
      return String(s == null ? "" : s).replace(/[&<>"]/g, function (c) {
        return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
      });
    }
    function pct(x) { return (Number(x) * 100).toFixed(2); }

    function setFile(file) {
      if (!file || !/^image\//.test(file.type)) return;
      selectedFile = file;
      dzPreview.src = URL.createObjectURL(file);
      dzPreview.hidden = false;
      dzEmpty.hidden = true;
      btn.disabled = false;
    }

    fileInput.addEventListener("change", function () {
      if (fileInput.files[0]) setFile(fileInput.files[0]);
    });
    ["dragenter", "dragover"].forEach(function (ev) {
      dropzone.addEventListener(ev, function (e) { e.preventDefault(); dropzone.classList.add("is-drag"); });
    });
    ["dragleave", "drop"].forEach(function (ev) {
      dropzone.addEventListener(ev, function (e) { e.preventDefault(); dropzone.classList.remove("is-drag"); });
    });
    dropzone.addEventListener("drop", function (e) {
      if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
    });

    form.addEventListener("submit", function (e) {
      e.preventDefault();
      if (!selectedFile) return;
      output.innerHTML = '<div class="scan__loading"><div class="spinner"></div>' +
        '<p class="muted-small">Running vision model and reasoning agent…</p></div>';
      btn.disabled = true;

      var fd = new FormData();
      fd.append("image", selectedFile);
      fd.append("acres", acres.value || "10");

      fetch("/api/diagnose", { method: "POST", body: fd })
        .then(function (r) {
          if (!r.ok) return r.json().then(function (j) { throw new Error(j.detail || ("HTTP " + r.status)); });
          return r.json();
        })
        .then(render)
        .catch(function (err) {
          output.innerHTML = '<div class="scan__placeholder"><span class="scan__ph-icon">⚠️</span>' +
            '<p class="muted-small">' + esc(err.message || "Something went wrong.") + "</p></div>";
        })
        .finally(function () { btn.disabled = false; });
    });

    function render(d) {
      var sev = (d.severity || "moderate").toLowerCase();
      var statusColor = d.is_healthy ? "#22c55e" : "#ef4444";
      var conf = pct(d.confidence);
      var a = d.agent || {};

      var others = (d.top_k || []).slice(1).map(function (p) {
        return '<div class="rc__other"><span>' + esc(p.species) + " · " + esc(p.condition) +
          "</span><span>" + pct(p.confidence) + "%</span></div>";
      }).join("");

      var html =
        '<div class="rcard">' +
          '<div class="rc__top"><div><div class="rc__label">Crop</div>' +
            '<div class="rc__crop">' + esc(d.species) + "</div></div>" +
            '<span class="badge badge--' + esc(sev) + '">Severity: ' + esc(sev.toUpperCase()) + "</span></div>" +
          '<div class="rc__status"><span class="status-dot" style="background:' + statusColor + '"></span>' +
            esc(d.status) + " · <b>" + esc(d.condition) + "</b></div>" +
          '<div class="meter"><div class="meter__fill" style="width:' + conf + '%"></div></div>' +
          '<div class="muted-small">Model confidence: <b style="color:var(--ink)">' + conf + "%</b></div>" +
          (others ? '<div class="rc__others"><div class="rc__label">Other possibilities</div>' + others + "</div>" : "") +
        "</div>";

      if (d.economics) {
        var e = d.economics;
        html +=
          '<div class="rcard econ">' +
            '<div class="econ__label">💰 Estimated economic exposure</div>' +
            '<div class="econ__head">' + esc(e.headline) + "</div>" +
            '<div class="econ__sub">' + esc(e.subline) + "</div>" +
            '<div class="econ__disc">' + esc(e.disclaimer) + "</div>" +
          "</div>";
      }

      html +=
        '<div class="rcard">' +
          '<div class="agent__head"><h4>🧠 Agent reasoning</h4>' +
            '<span class="agent__tag">via ' + esc(a.engine) + "</span></div>" +
          (a.headline ? '<p class="agent__lead">' + esc(a.headline) + "</p>" : "");

      if (a.differential) {
        var df = a.differential;
        var cands = (df.candidates || []).map(function (c) {
          return '<div class="diff__cand"><span>' + esc(c.condition) + "</span><span>" + pct(c.confidence) + "%</span></div>";
        }).join("");
        html +=
          '<div class="astep diff"><h5>🔎 Differential diagnosis ' +
            '<span class="agent__tag">· ' + esc(df.retrievals || 0) + " grounded retrievals</span></h5>" +
            '<div class="diff__cands">' + cands + "</div>" +
            '<div class="diff__verdict"><b>Verdict:</b> ' + esc(df.verdict) + ". " + esc(df.rationale) + "</div>" +
            '<div class="diff__confirm">✔️ <b>To confirm:</b> ' + esc(df.confirm_checks) + "</div></div>";
      }

      (a.steps || []).forEach(function (s) {
        html += '<div class="astep"><h5>' + esc(s.icon) + " " + esc(s.title) + "</h5><p>" + esc(s.content) + "</p></div>";
      });
      html += "</div>";

      var cites = (a.citations || []).map(function (c) {
        var label = esc(c.title) + (c.source ? " · " + esc(c.source) : "");
        return "<li>" + (c.url ? '<a href="' + esc(c.url) + '" target="_blank" rel="noopener">' + label + "</a>" : label) + "</li>";
      }).join("");
      if (cites) {
        html +=
          '<div class="rcard"><div class="agent__head"><h4>📚 Grounded sources</h4>' +
            '<span class="agent__tag">' + esc(a.grounding) + "</span></div>" +
            '<ul class="cites">' + cites + "</ul>" +
            (d.disclaimer ? '<div class="rc__disc">' + esc(d.disclaimer) + "</div>" : "") +
          "</div>";
      }

      output.innerHTML = html;
      output.scrollTop = 0;
    }

    // Only enable the scan UI when the diagnosis engine is reachable.
    function probe() {
      fetch("/api/health", { cache: "no-store" })
        .then(function (r) { if (!r.ok) throw new Error(); return r.json(); })
        .then(function () { scan.classList.remove("is-hidden"); offline.classList.add("is-hidden"); })
        .catch(function () { scan.classList.add("is-hidden"); offline.classList.remove("is-hidden"); });
    }
    probe();
    if (retry) retry.addEventListener("click", probe);
  })();

  /* ---- year ---- */
  var y = document.querySelector("[data-year]");
  if (y) y.textContent = new Date().getFullYear();
})();
