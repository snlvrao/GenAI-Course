/* =============================================================================
   Shared behaviour for every page: chrome, progress, video embeds, quizzes,
   notes, search, theme. Reads window.COURSE from data.js.

   Everything that touches localStorage goes through the store wrapper below,
   because file:// pages in some browsers throw on access rather than returning
   null. A thrown error there would take the whole page down.
   ============================================================================= */
(function () {
  "use strict";

  var KEY = "gaic:v1";
  var C = window.COURSE || { modules: [], phases: [] };

  /* --- storage ----------------------------------------------------------- */

  var storageOK = true;
  function read(name, fallback) {
    try {
      var raw = localStorage.getItem(KEY + ":" + name);
      return raw ? JSON.parse(raw) : fallback;
    } catch (e) {
      storageOK = false;
      return fallback;
    }
  }
  function write(name, value) {
    try {
      localStorage.setItem(KEY + ":" + name, JSON.stringify(value));
      return true;
    } catch (e) {
      storageOK = false;
      return false;
    }
  }

  var progress = read("progress", {});
  var notes = read("notes", {});
  var quizState = read("quiz", {});

  function moduleProgress(id) {
    return progress[id] || { read: false, lab: false, mini: false };
  }
  function setModuleProgress(id, patch) {
    var cur = moduleProgress(id);
    for (var k in patch) cur[k] = patch[k];
    progress[id] = cur;
    write("progress", progress);
    paintProgress();
  }

  /* A module counts as complete when all three boxes are ticked. Reading
     alone is not completion - that is the whole point of the mini-projects. */
  function isComplete(id) {
    var p = moduleProgress(id);
    return !!(p.read && p.lab && p.mini);
  }

  /* Setup is a prerequisite, not an optional page. It is tracked separately
     from module progress so the course can nag until it is done. */
  function setupDone() { return !!read("setup", false); }
  function setSetupDone(v) {
    write("setup", !!v);
    paintSetup();
    paintProgress();
  }
  function overallPercent() {
    if (!C.modules.length) return 0;
    var units = C.modules.length * 3, done = 0;
    C.modules.forEach(function (m) {
      var p = moduleProgress(m.id);
      if (p.read) done++;
      if (p.lab) done++;
      if (p.mini) done++;
    });
    return Math.round((done / units) * 100);
  }

  /* --- small helpers ----------------------------------------------------- */

  function el(tag, cls, html) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (html != null) n.innerHTML = html;
    return n;
  }
  function esc(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }
  function fmtViews(v) {
    if (v >= 1e6) return (v / 1e6).toFixed(1).replace(/\.0$/, "") + "M views";
    if (v >= 1e3) return Math.round(v / 1e3) + "k views";
    return v + " views";
  }
  function fmtMins(m) {
    if (m < 60) return m + " min";
    var h = Math.floor(m / 60), r = m % 60;
    return r ? h + "h " + r + "m" : h + "h";
  }
  function currentModule() {
    var b = document.body.getAttribute("data-module");
    if (!b) return null;
    for (var i = 0; i < C.modules.length; i++) if (C.modules[i].id === b) return C.modules[i];
    return null;
  }
  function base() {
    return document.body.getAttribute("data-root") || "";
  }

  var ICON_CHECK = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>';

  /* --- theme ------------------------------------------------------------- */

  function initTheme() {
    var saved = read("theme", null);
    if (saved === "dark" || saved === "light") {
      document.documentElement.setAttribute("data-theme", saved);
    }
    document.addEventListener("click", function (e) {
      var btn = e.target.closest("[data-act='theme']");
      if (!btn) return;
      var isDark =
        document.documentElement.getAttribute("data-theme") === "dark" ||
        (!document.documentElement.getAttribute("data-theme") &&
          window.matchMedia("(prefers-color-scheme: dark)").matches);
      var next = isDark ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      write("theme", next);
      btn.setAttribute("aria-label", next === "dark" ? "Switch to light theme" : "Switch to dark theme");
    });
  }

  /* --- masthead ---------------------------------------------------------- */

  function buildMasthead() {
    var host = document.querySelector("[data-chrome='masthead']");
    if (!host) return;
    var segs = "";
    for (var p = 1; p <= (C.phases || []).length; p++) {
      segs += '<span class="progressbar__seg" data-phase="' + p + '"><i class="progressbar__fill" data-pfill="' + p + '"></i></span>';
    }
    host.className = "masthead";
    host.innerHTML =
      '<button class="iconbtn navtoggle" data-act="nav" aria-label="Show module list" aria-expanded="false">' +
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M3 6h18M3 12h18M3 18h18"/></svg>' +
      "</button>" +
      '<a class="masthead__brand" href="' + base() + 'index.html">' + esc(C.title || "Course") + " <span>&middot; self-study</span></a>" +
      '<span class="masthead__spacer"></span>' +
      '<div class="progressbar" role="img" aria-label="Course progress" data-progressbar>' + segs + "</div>" +
      '<span class="progressbar__pct" data-pct>0%</span>' +
      '<a class="iconbtn" href="' + base() + 'setup.html" aria-label="Setup and installation guide" title="Setup and installation">' +
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.6 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.6a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9c.2.5.66.86 1.19 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1Z"/></svg>' +
      "</a>" +
      '<button class="iconbtn" data-act="theme" aria-label="Switch theme" title="Switch light / dark">' +
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/></svg>' +
      "</button>";
  }

  function paintProgress() {
    var byPhase = {};
    C.modules.forEach(function (m) {
      byPhase[m.phase] = byPhase[m.phase] || { t: 0, d: 0 };
      byPhase[m.phase].t += 3;
      var p = moduleProgress(m.id);
      byPhase[m.phase].d += (p.read ? 1 : 0) + (p.lab ? 1 : 0) + (p.mini ? 1 : 0);
    });
    document.querySelectorAll("[data-pfill]").forEach(function (f) {
      var ph = byPhase[+f.getAttribute("data-pfill")];
      f.style.width = ph && ph.t ? (ph.d / ph.t) * 100 + "%" : "0%";
    });
    var pct = overallPercent();
    document.querySelectorAll("[data-pct]").forEach(function (n) { n.textContent = pct + "%"; });
    var bar = document.querySelector("[data-progressbar]");
    if (bar) bar.setAttribute("aria-label", "Course progress: " + pct + " percent");

    document.querySelectorAll("[data-navdots]").forEach(function (host) {
      var p = moduleProgress(host.getAttribute("data-navdots"));
      var d = host.children;
      if (d[0]) d[0].classList.toggle("dot--on", !!p.read);
      if (d[1]) d[1].classList.toggle("dot--on", !!p.lab);
      if (d[2]) d[2].classList.toggle("dot--on", !!p.mini);
    });
    document.querySelectorAll("[data-mstate]").forEach(function (n) {
      n.classList.toggle("dot--on", isComplete(n.getAttribute("data-mstate")));
    });
    document.querySelectorAll("[data-stat='done']").forEach(function (n) {
      n.textContent = C.modules.filter(function (m) { return isComplete(m.id); }).length;
    });
  }

  /* --- sidebar ----------------------------------------------------------- */

  function buildSidebar() {
    var host = document.querySelector("[data-chrome='sidebar']");
    if (!host) return;
    var cur = document.body.getAttribute("data-module");
    var onSetup = /setup\.html$/i.test(location.pathname);
    var html = '<label class="sr" for="navsearch">Filter modules</label>' +
      '<input class="navsearch" id="navsearch" type="search" placeholder="Filter by topic…" autocomplete="off">' +
      /* Step 0 is pinned above the phases so it is never something you have to
         go looking for. */
      '<div class="phase" data-phase="1"><div class="phase__head">Start here</div>' +
        '<a class="navlink" href="' + base() + 'setup.html"' + (onSetup ? ' aria-current="page"' : "") +
        ' data-kw="setup install python venv environment api key packages pip prerequisite start">' +
        '<span class="navlink__n">0</span>' +
        '<span class="navlink__t">Setup: do this first</span>' +
        '<span class="navlink__dots"><i class="dot" data-setupdot></i></span></a></div>';

    C.phases.forEach(function (ph) {
      var mods = C.modules.filter(function (m) { return m.phase === ph.n; });
      if (!mods.length) return;
      html += '<div class="phase" data-phase="' + ph.n + '"><div class="phase__head">' + esc(ph.name) + "</div>";
      mods.forEach(function (m) {
        var kw = (m.title + " " + m.promise + " " + m.concepts.join(" ") + " " + m.lab.t + " " + m.mini.t).toLowerCase();
        html +=
          '<a class="navlink" href="' + base() + "modules/" + m.file + '"' +
          (m.id === cur ? ' aria-current="page"' : "") +
          ' data-kw="' + esc(kw) + '">' +
          '<span class="navlink__n">' + m.n + "</span>" +
          '<span class="navlink__t">' + esc(m.title) + "</span>" +
          '<span class="navlink__dots" data-navdots="' + m.id + '" title="read · lab · mini-project">' +
            '<i class="dot"></i><i class="dot"></i><i class="dot"></i>' +
          "</span></a>";
      });
      html += "</div>";
    });
    html += '<p class="small muted" style="padding:0 8px">Three dots per module: read, lab, mini-project.</p>';
    host.className = "sidebar";
    host.innerHTML = html;

    var input = host.querySelector("#navsearch");
    input.addEventListener("input", function () {
      var q = input.value.trim().toLowerCase();
      host.querySelectorAll(".phase").forEach(function (ph) {
        var any = false;
        ph.querySelectorAll(".navlink").forEach(function (a) {
          var hit = !q || a.getAttribute("data-kw").indexOf(q) !== -1;
          a.classList.toggle("is-hidden", !hit);
          if (hit) any = true;
        });
        ph.style.display = any ? "" : "none";
      });
    });

    document.addEventListener("click", function (e) {
      if (e.target.closest("[data-act='nav']")) {
        var open = host.classList.toggle("is-open");
        var b = document.querySelector("[data-act='nav']");
        if (b) b.setAttribute("aria-expanded", String(open));
      } else if (host.classList.contains("is-open") && !e.target.closest(".sidebar")) {
        host.classList.remove("is-open");
      }
    });
  }

  /* --- pager ------------------------------------------------------------- */

  function buildPager() {
    var host = document.querySelector("[data-chrome='pager']");
    var m = currentModule();
    if (!host || !m) return;
    var i = C.modules.indexOf(m);
    var prev = C.modules[i - 1], next = C.modules[i + 1];
    host.className = "pager";
    host.innerHTML =
      (prev
        ? '<a href="' + base() + "modules/" + prev.file + '"><small>&larr; Previous</small>' + esc(prev.n + ". " + prev.title) + "</a>"
        : '<a href="' + base() + 'index.html"><small>&larr; Back</small>Course home</a>') +
      (next
        ? '<a class="is-next" href="' + base() + "modules/" + next.file + '"><small>Next &rarr;</small>' + esc(next.n + ". " + next.title) + "</a>"
        : '<a class="is-next" href="' + base() + 'index.html"><small>Finished &rarr;</small>Back to course home</a>');
  }

  /* --- opened from disk -------------------------------------------------- */

  /* Everything on these pages works from disk except the videos. YouTube will
     not start its player in a page that has no address, so this says how to
     give it one. Shown once and dismissible, because a banner you cannot get
     rid of is worse than the problem. */
  function buildDiskNotice() {
    if (location.protocol !== "file:") return;
    if (read("diskNoticeHidden", false)) return;
    var main = document.querySelector(".main .wrap") || document.querySelector(".main");
    if (!main) return;

    var box = el("div", "disknote");
    var p = el("p", "");
    p.innerHTML =
      "<b>Videos will not play in this window.</b> You have opened the course "
      + "as a file, and YouTube refuses to run its player in a page with no "
      + "address. Everything else on the page works. To get the videos, close "
      + "this and run <code>Start course.cmd</code> in the course folder, or "
      + "<code>python start.py</code>. It serves the course on your own "
      + "machine and opens it. Nothing is uploaded.";
    var x = el("button", "disknote__x", "Dismiss");
    x.type = "button";
    x.addEventListener("click", function () {
      write("diskNoticeHidden", true);
      box.remove();
    });
    box.appendChild(p);
    box.appendChild(x);
    main.insertBefore(box, main.firstChild);
  }

  /* --- videos ------------------------------------------------------------ */

  /* Click-to-play. The poster is a plain <img>; only on click do we insert an
     iframe, so opening a page never loads six YouTube players. */
  function buildVideos() {
    var m = currentModule();

    document.querySelectorAll("[data-videos]").forEach(function (host) {
      var which = host.getAttribute("data-videos");
      var list = (m ? m.videos : []).filter(function (v) {
        return which === "all" ? true : which === "core" ? v.core : !v.core;
      });
      if (!list.length) { host.remove(); return; }
      host.className = "videos" + (list.length > 1 ? " videos--multi" : "");
      host.innerHTML = list.map(function (v) {
        return (
          '<div class="video">' +
            '<button class="video__frame" type="button" data-yt="' + v.id + '" ' +
              'aria-label="Play video: ' + esc(v.t) + '">' +
              '<img src="https://img.youtube.com/vi/' + v.id + '/hqdefault.jpg" alt="" loading="lazy" ' +
                'onerror="this.remove();this.parentNode.classList.add(\'video__frame--noposter\')">' +
              '<span class="video__play">' +
                '<svg viewBox="0 0 68 48" aria-hidden="true"><path fill="#f00" d="M66.5 7.7a8.6 8.6 0 0 0-6-6C55.2 0 34 0 34 0S12.8 0 7.5 1.7a8.6 8.6 0 0 0-6 6A89 89 0 0 0 0 24a89 89 0 0 0 1.5 16.3 8.6 8.6 0 0 0 6 6C12.8 48 34 48 34 48s21.2 0 26.5-1.7a8.6 8.6 0 0 0 6-6A89 89 0 0 0 68 24a89 89 0 0 0-1.5-16.3Z"/><path fill="#fff" d="M27 34 45 24 27 14Z"/></svg>' +
              "</span>" +
            "</button>" +
            '<div class="video__body">' +
              '<p class="video__t">' + esc(v.t) + "</p>" +
              '<p class="video__m"><b>' + esc(v.c) + "</b><span>" + fmtMins(v.m) + "</span><span>" +
                fmtViews(v.v) + "</span><span>" + esc(v.d) + "</span></p>" +
            "</div>" +
          "</div>"
        );
      }).join("");
    });

    document.addEventListener("click", function (e) {
      var btn = e.target.closest("[data-yt]");
      if (!btn) return;
      var id = btn.getAttribute("data-yt");

      /* A page opened straight from disk has no origin for YouTube to check,
         and the embedded player refuses to start: it shows "Error 153, video
         player configuration error" instead. Nothing on this side can satisfy
         that check, so rather than plant a broken player in the page we open
         the video in a new tab, which always works. Served over http the embed
         is fine, so inline playback is kept there. */
      if (location.protocol === "file:") {
        window.open("https://www.youtube.com/watch?v=" + id, "_blank", "noopener");
        return;
      }

      var f = document.createElement("iframe");
      f.src = "https://www.youtube-nocookie.com/embed/" + id + "?autoplay=1&rel=0&modestbranding=1";
      f.title = btn.getAttribute("aria-label") || "Video";
      f.allow = "accelerometer; autoplay; encrypted-media; picture-in-picture; fullscreen";
      f.allowFullscreen = true;
      var holder = document.createElement("div");
      holder.className = "video__frame";
      btn.replaceWith(holder);
      holder.appendChild(f);
    });
  }

  /* --- copy buttons ------------------------------------------------------ */

  function buildCopy() {
    document.querySelectorAll(".codeblock").forEach(function (block) {
      var bar = block.querySelector(".codeblock__bar");
      if (!bar || bar.querySelector("[data-copy]")) return;
      wireCopy(bar, el("button", "btn", "Copy"), block.querySelector("pre"));
    });
    // Worked examples get one too. These are the blocks the learner is most
    // likely to want in their own editor, and there is no bar to hang it on,
    // so the button floats in the corner of the block itself.
    document.querySelectorAll(".eg pre:not(.eg__out)").forEach(function (pre) {
      if (pre.parentNode.querySelector(".eg__copy")) return;
      var wrap = el("div", "eg__codewrap");
      pre.parentNode.insertBefore(wrap, pre);
      wrap.appendChild(pre);
      wireCopy(wrap, el("button", "eg__copy", "Copy"), pre);
    });
  }

  function wireCopy(host, b, code) {
    if (!code) return;
    b.type = "button";
    b.setAttribute("data-copy", "");
    host.appendChild(b);
    b.addEventListener("click", function () {
      // innerText, not textContent: the block is now painted with <span>
      // tokens and innerText is what preserves the line breaks between them.
      var text = code.innerText;
      var done = function () {
        b.textContent = "Copied";
        b.classList.add("btn--ok");
        setTimeout(function () { b.textContent = "Copy"; b.classList.remove("btn--ok"); }, 1600);
      };
      function fallback() {
        var ta = document.createElement("textarea");
        ta.value = text;
        ta.setAttribute("readonly", "");
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand("copy"); done(); } catch (err) { b.textContent = "Press Ctrl+C"; }
        ta.remove();
      }
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, fallback);
      } else fallback();
    });
  }

  /* --- quiz -------------------------------------------------------------- */

  function buildQuiz() {
    var m = currentModule();
    document.querySelectorAll("[data-quiz]").forEach(function (host, qi) {
      var answered = (m && quizState[m.id]) || {};
      host.querySelectorAll(".q").forEach(function (q, idx) {
        var correct = +q.getAttribute("data-answer");
        var why = q.querySelector(".q__why");
        var opts = Array.prototype.slice.call(q.querySelectorAll(".opt"));

        function reveal(chosen) {
          opts.forEach(function (o, i) {
            o.disabled = true;
            if (i === correct) o.classList.add("opt--right");
            else if (i === chosen) o.classList.add("opt--wrong");
          });
          if (why) {
            why.hidden = false;
            var verdict = chosen === correct
              ? "<b>Correct.</b> "
              : "<b>Not quite. The answer is " + String.fromCharCode(65 + correct) + ".</b> ";
            if (!why.getAttribute("data-orig")) why.setAttribute("data-orig", why.innerHTML);
            why.innerHTML = verdict + why.getAttribute("data-orig");
          }
        }

        opts.forEach(function (o, i) {
          o.addEventListener("click", function () {
            if (o.disabled) return;
            reveal(i);
            if (m) {
              quizState[m.id] = quizState[m.id] || {};
              quizState[m.id][idx] = i;
              write("quiz", quizState);
            }
          });
        });

        if (answered[idx] != null) reveal(answered[idx]);
      });
    });
  }

  /* --- notes ------------------------------------------------------------- */

  function buildNotes() {
    var m = currentModule();
    var box = document.querySelector("[data-notes]");
    if (!box || !m) return;
    var ta = box.querySelector("textarea");
    var status = box.querySelector(".notes__status");
    ta.value = notes[m.id] || "";
    var timer = null;
    ta.addEventListener("input", function () {
      if (status) status.textContent = "Saving…";
      clearTimeout(timer);
      timer = setTimeout(function () {
        notes[m.id] = ta.value;
        var ok = write("notes", notes);
        if (status) status.textContent = ok
          ? "Saved" + (ta.value.trim() ? " · " + ta.value.trim().split(/\s+/).length + " words" : "")
          : "Could not save. Storage is blocked in this browser.";
      }, 500);
    });
  }

  /* --- completion marks -------------------------------------------------- */

  function buildMarks() {
    var m = currentModule();
    if (!m) return;
    document.querySelectorAll("[data-mark]").forEach(function (btn) {
      var kind = btn.getAttribute("data-mark");
      var sync = function () {
        btn.setAttribute("aria-pressed", String(!!moduleProgress(m.id)[kind]));
      };
      if (!btn.querySelector(".mark__box")) {
        btn.insertBefore(el("span", "mark__box", ICON_CHECK), btn.firstChild);
      }
      btn.addEventListener("click", function () {
        var patch = {};
        patch[kind] = !moduleProgress(m.id)[kind];
        setModuleProgress(m.id, patch);
        sync();
      });
      sync();
    });
  }

  /* --- setup path -------------------------------------------------------- */

  /* There are three honest ways to get a model, and they need different
     instructions. Rather than printing all three and asking the learner to
     work out which lines apply to them, the page hides the ones that do not. */
  var PATHS = {
    offline: {
      name: "Fully offline",
      blurb: "A model on your own machine. No account, no key, no internet.",
      cost: "Free forever",
      needs: "About 3&nbsp;GB of disk",
      net: "Internet once, to download the model",
      icon: "M5 12h14M12 5v14",
    },
    free: {
      name: "Free hosted",
      blurb: "A free API key from Groq or Google AI Studio. No card needed.",
      cost: "Free, with daily limits",
      needs: "An email address",
      net: "Internet every time you run a lab",
      icon: "M12 3v18M3 12h18",
    },
    key: {
      name: "Your own key",
      blurb: "You already pay for OpenAI, Anthropic, Gemini or similar.",
      cost: "You pay per token",
      needs: "An existing API key",
      net: "Internet every time you run a lab",
      icon: "M4 12h16",
    },
  };

  function setupPath() {
    var p = read("path", null);
    return PATHS[p] ? p : null;
  }
  function setSetupPath(p) {
    write("path", p);
    paintPath();
    paintSetup();
  }

  function paintPath() {
    var cur = setupPath();

    /* the chooser itself */
    var host = document.querySelector("[data-pathpicker]");
    if (host) {
      host.innerHTML = Object.keys(PATHS).map(function (k) {
        var p = PATHS[k];
        var on = k === cur;
        return '<button class="pathcard' + (on ? " is-on" : "") + '" type="button" data-pick="' + k + '"' +
          ' aria-pressed="' + on + '">' +
          '<span class="pathcard__h">' + esc(p.name) + (on ? " &check;" : "") + "</span>" +
          '<span class="pathcard__b">' + esc(p.blurb) + "</span>" +
          '<span class="pathcard__m"><b>' + p.cost + "</b>" + p.needs + "</span>" +
          "</button>";
      }).join("");
    }

    /* show or hide every path-specific block */
    document.querySelectorAll("[data-path]").forEach(function (n) {
      var want = n.getAttribute("data-path").split(/\s+/);
      n.hidden = !cur || want.indexOf(cur) === -1;
    });
    document.querySelectorAll("[data-path-not]").forEach(function (n) {
      var no = n.getAttribute("data-path-not").split(/\s+/);
      n.hidden = !!cur && no.indexOf(cur) !== -1;
    });
    document.querySelectorAll("[data-needpath]").forEach(function (n) {
      n.hidden = !!cur;
    });
    document.querySelectorAll("[data-havepath]").forEach(function (n) {
      n.hidden = !cur;
    });

    /* renumber the steps that are actually on screen */
    var i = 0;
    document.querySelectorAll("[data-step]").forEach(function (h) {
      var sec = h.closest("section");
      if (sec && sec.hidden) return;
      if (h.hidden) return;
      i++;
      h.textContent = "Step " + i + ": " + h.getAttribute("data-step");
    });

    /* the summary line at the top */
    var sum = document.querySelector("[data-pathsummary]");
    if (sum) {
      if (!cur) {
        sum.innerHTML = "";
      } else {
        var p = PATHS[cur];
        sum.innerHTML =
          '<div class="note note--ok" style="margin:18px 0 0">' +
          '<div class="note__t">Your path: ' + esc(p.name) + "</div>" +
          "<p>" + esc(p.blurb) + " <b>" + p.cost + ".</b> " + p.needs + ". " + p.net + ".</p>" +
          '<p style="margin-bottom:0"><button class="btn" data-pick="" type="button">' +
          "Choose a different path</button></p></div>";
      }
    }

    /* anywhere else that names the current path */
    document.querySelectorAll("[data-pathname]").forEach(function (n) {
      n.textContent = cur ? PATHS[cur].name.toLowerCase() : "not chosen yet";
    });
  }

  function buildPathControls() {
    document.addEventListener("click", function (e) {
      var b = e.target.closest("[data-pick]");
      if (!b) return;
      var v = b.getAttribute("data-pick");
      setSetupPath(v || null);
      if (!v) {
        var picker = document.querySelector("[data-pathpicker]");
        if (picker) picker.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    });
  }

  /* --- setup gate -------------------------------------------------------- */

  /* Every lab depends on the setup page, so rather than hoping the learner
     finds it, the course says so on every module page until it is ticked off. */
  function paintSetup() {
    var done = setupDone();

    document.querySelectorAll("[data-setupdot]").forEach(function (d) {
      d.classList.toggle("dot--on", done);
    });

    document.querySelectorAll("[data-setupmark]").forEach(function (btn) {
      btn.setAttribute("aria-pressed", String(done));
      var label = btn.querySelector("[data-setuplabel]");
      if (label) label.textContent = done
        ? "Setup is done"
        : "I have finished the setup";
    });

    var card = document.querySelector("[data-setupcard]");
    if (card) {
      var cur = setupPath();
      card.innerHTML = done
        ? '<div class="note note--ok" style="margin:0">' +
            '<div class="note__t">Step 0 complete</div>' +
            "<p>Your workspace is ready" +
            (cur ? ", set up for the <b>" + PATHS[cur].name.toLowerCase() + "</b> path" : "") +
            ". Go to <a href=\"" + base() + "modules/m01-ml-refresher.html\">Module 1</a>, or back to " +
            '<a href="' + base() + 'setup.html">setup</a> for the troubleshooting notes.</p></div>'
        : '<div class="note note--warn" style="margin:0">' +
            '<div class="note__t">Step 0: do this before Module 1</div>' +
            "<p>Every lab needs a Python workspace and one model you can call. The " +
            '<a href="' + base() + 'setup.html"><b>setup page</b></a> asks which of three paths you ' +
            "want (fully offline, a free key, or your own key) and then shows only the steps for " +
            "that path. About twenty minutes.</p>" +
            '<p style="margin-bottom:0"><a class="btn" style="padding:9px 14px;font-size:.92rem" href="' +
            base() + 'setup.html">Open the setup page &rarr;</a></p></div>';
    }

    var banner = document.querySelector("[data-setupbanner]");
    if (banner) {
      if (done) { banner.innerHTML = ""; return; }
      banner.innerHTML =
        '<div class="note note--warn">' +
        '<div class="note__t">Before the lab in this module</div>' +
        "<p style=\"margin-bottom:0\">You have not marked the setup as done yet. The lab below needs " +
        "a working Python workspace and a model you can call. " +
        '<a href="' + base() + 'setup.html"><b>Do the setup first</b></a>, ' +
        "then come back. Reading the notes and watching the videos is fine without it.</p></div>";
    }
  }

  function buildSetupControls() {
    document.addEventListener("click", function (e) {
      var btn = e.target.closest("[data-setupmark]");
      if (!btn) return;
      setSetupDone(!setupDone());
    });
    document.querySelectorAll("[data-setupmark]").forEach(function (btn) {
      if (!btn.querySelector(".mark__box")) {
        btn.insertBefore(el("span", "mark__box", ICON_CHECK), btn.firstChild);
      }
    });
  }

  /* --- storage warning --------------------------------------------------- */

  function storageWarning() {
    try {
      localStorage.setItem(KEY + ":probe", "1");
      localStorage.removeItem(KEY + ":probe");
    } catch (e) { storageOK = false; }
    if (storageOK) return;
    var main = document.querySelector(".main .wrap") || document.querySelector(".main");
    if (!main) return;
    var n = el("div", "note note--warn",
      '<div class="note__t">Progress will not be saved</div>' +
      "<p>This browser is blocking local storage for files opened from disk, so ticking a module " +
      "complete will not survive a reload. Everything else on the page works normally. " +
      "Opening the course in Chrome or Edge usually fixes it.</p>");
    main.insertBefore(n, main.firstChild);
  }

  /* --- export / import / reset ------------------------------------------- */

  function buildDataTools() {
    var host = document.querySelector("[data-tools]");
    if (!host) return;

    host.addEventListener("click", function (e) {
      var act = e.target.getAttribute && e.target.getAttribute("data-act");

      if (act === "export") {
        var blob = new Blob([JSON.stringify({
          exported: new Date().toISOString(),
          version: 1,
          setup: setupDone(),
          path: setupPath(),
          progress: progress, notes: notes, quiz: quizState
        }, null, 2)], { type: "application/json" });
        var a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = "genai-course-progress.json";
        a.click();
        setTimeout(function () { URL.revokeObjectURL(a.href); }, 2000);
      }

      if (act === "import") host.querySelector("[data-import]").click();

      if (act === "reset") {
        var msg = "This erases all progress, notes and quiz answers on this device.\n\n" +
                  "Export first if you want a copy. Type nothing, just confirm.\n\nReset everything?";
        if (!confirm(msg)) return;
        if (!confirm("Last check: this cannot be undone. Reset?")) return;
        progress = {}; notes = {}; quizState = {};
        write("progress", progress); write("notes", notes); write("quiz", quizState);
        write("setup", false);
        write("path", null);
        paintProgress();
        alert("Progress cleared.");
        location.reload();
      }
    });

    var file = host.querySelector("[data-import]");
    if (file) {
      file.addEventListener("change", function () {
        var f = file.files && file.files[0];
        if (!f) return;
        var r = new FileReader();
        r.onload = function () {
          try {
            var d = JSON.parse(r.result);
            if (!d || typeof d !== "object") throw new Error("shape");
            progress = d.progress || {};
            notes = d.notes || {};
            quizState = d.quiz || {};
            write("progress", progress); write("notes", notes); write("quiz", quizState);
            write("setup", !!d.setup);
            write("path", d.path || null);
            paintProgress();
            alert("Progress restored.");
            location.reload();
          } catch (err) {
            alert("That file could not be read. It should be a JSON file exported from this course.");
          }
        };
        r.readAsText(f);
        file.value = "";
      });
    }
  }

  /* --- index page -------------------------------------------------------- */

  function buildIndex() {
    var host = document.querySelector("[data-chrome='phasegrid']");
    if (!host) return;
    host.className = "phasegrid";
    host.innerHTML = C.phases.map(function (ph) {
      var mods = C.modules.filter(function (m) { return m.phase === ph.n; });
      return '<section class="pcard" data-phase="' + ph.n + '">' +
        "<h3>" + esc(ph.name) + "</h3><p>" + esc(ph.tag) + "</p>" +
        '<div class="mlist">' + mods.map(function (m) {
          var kw = (m.title + " " + m.promise + " " + m.concepts.join(" ") + " " + m.lab.t + " " + m.mini.t).toLowerCase();
          return '<a class="mlink" href="modules/' + m.file + '" data-kw="' + esc(kw) + '">' +
            '<span class="mlink__n">' + (m.n < 10 ? "0" : "") + m.n + "</span>" +
            "<span>" + esc(m.title) + "</span>" +
            '<i class="dot mlink__s" data-mstate="' + m.id + '" title="complete"></i></a>';
        }).join("") + "</div></section>";
    }).join("");

    var search = document.querySelector("[data-sitesearch]");
    if (search) {
      var empty = document.querySelector("[data-noresults]");
      search.addEventListener("input", function () {
        var q = search.value.trim().toLowerCase();
        var hits = 0;
        host.querySelectorAll(".pcard").forEach(function (card) {
          var any = false;
          card.querySelectorAll(".mlink").forEach(function (a) {
            var hit = !q || a.getAttribute("data-kw").indexOf(q) !== -1;
            a.classList.toggle("is-hidden", !hit);
            if (hit) { any = true; hits++; }
          });
          card.classList.toggle("is-hidden", !any);
        });
        if (empty) empty.hidden = !(q && hits === 0);
      });
    }

    var vt = document.querySelector("[data-stat='videos']");
    if (vt) vt.textContent = C.modules.reduce(function (a, m) { return a + m.videos.length; }, 0);
    var ht = document.querySelector("[data-stat='hours']");
    if (ht) ht.textContent = C.modules.reduce(function (a, m) { return a + m.hours; }, 0);
    var mt = document.querySelector("[data-stat='modules']");
    if (mt) mt.textContent = C.modules.length;
    var bt = document.querySelector("[data-stat='builds']");
    if (bt) bt.textContent = C.modules.reduce(function (a, m) {
      return a + (m.lab ? 1 : 0) + (m.mini ? 1 : 0);
    }, 0);
  }

  /* --- module header ----------------------------------------------------- */

  function buildModuleHeader() {
    var m = currentModule();
    var host = document.querySelector("[data-chrome='modhead']");
    if (!m || !host) return;
    var ph = C.phases.filter(function (p) { return p.n === m.phase; })[0] || { name: "" };
    document.title = m.n + ". " + m.title + " · " + (C.title || "Course");
    host.innerHTML =
      '<div class="eyebrow">Module ' + m.n + " · " + esc(ph.name) + "</div>" +
      "<h1>" + esc(m.title) + "</h1>" +
      '<p class="promise">' + esc(m.promise) + "</p>" +
      '<div class="meta-row">' +
        '<span class="chip chip--phase">Phase ' + m.phase + "</span>" +
        '<span class="chip">' + m.hours + " hours</span>" +
        '<span class="chip">' + m.videos.length + " videos</span>" +
        '<span class="chip">' + m.concepts.length + " concepts</span>" +
        '<span class="chip">Lab + mini-project</span>' +
      "</div>";
  }

  /* --- boot -------------------------------------------------------------- */

  function boot() {
    var m = currentModule();
    if (m) {
      var root = document.querySelector(".shell") || document.body;
      root.setAttribute("data-phase", m.phase);
    }
    initTheme();
    buildMasthead();
    buildSidebar();
    buildModuleHeader();
    buildVideos();
    buildQuiz();
    buildNotes();
    buildMarks();
    buildPager();
    buildIndex();
    buildDataTools();
    buildSetupControls();
    buildPathControls();
    buildDiskNotice();
    buildCopy();
    paintProgress();
    paintPath();
    paintSetup();
    storageWarning();
    if (window.WIDGETS && typeof window.WIDGETS.init === "function") window.WIDGETS.init();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();

  window.COURSE_API = { read: read, write: write, progress: function () { return progress; }, repaint: paintProgress };
})();
