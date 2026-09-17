// Shared helpers for every CreditSense demo page. Vanilla JS, no framework, no build
// step (FRONTEND_REQUIREMENTS.md §6) -- every page includes this file directly.

const CS = (() => {
  async function fetchJSON(url, options) {
    let res;
    try {
      res = await fetch(url, options);
    } catch (err) {
      throw new Error("Network error: could not reach the server.");
    }
    let body = null;
    try {
      body = await res.json();
    } catch (err) {
      // no/invalid JSON body -- fall through, body stays null
    }
    if (!res.ok) {
      const detail = (body && body.detail) || `${res.status} ${res.statusText}`;
      const err = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      err.status = res.status;
      throw err;
    }
    return body;
  }

  function el(tag, attrs, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === "class") node.className = v;
      else if (k === "html") node.innerHTML = v;
      else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
      else if (v !== null && v !== undefined) node.setAttribute(k, v);
    }
    for (const child of children.flat()) {
      if (child === null || child === undefined) continue;
      node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
    }
    return node;
  }

  function skeleton(lines = 3) {
    const wrap = el("div", { class: "animate-pulse space-y-2" });
    for (let i = 0; i < lines; i++) {
      wrap.appendChild(el("div", { class: "h-4 bg-slate-200 rounded w-full" }));
    }
    return wrap;
  }

  function emptyState(message) {
    return el(
      "div",
      { class: "text-center text-slate-500 py-12 border border-dashed border-slate-300 rounded-lg" },
      message
    );
  }

  function errorState(message) {
    return el(
      "div",
      { class: "text-center text-red-700 bg-red-50 border border-red-200 rounded-lg py-8 px-4" },
      message
    );
  }

  // Never rely on color alone (FRONTEND_REQUIREMENTS.md §3.1) -- every badge pairs an
  // icon, text, and color.
  const VERDICTS = {
    APPROVE: { label: "Approve", icon: "✓", classes: "bg-emerald-100 text-emerald-800 border-emerald-300" },
    DECLINE: { label: "Decline", icon: "✕", classes: "bg-red-100 text-red-800 border-red-300" },
    ESCALATE_TO_HUMAN: { label: "Needs you", icon: "✋", classes: "bg-amber-100 text-amber-900 border-amber-400" },
  };

  function verdictBadge(decision, opts) {
    const v = VERDICTS[decision] || { label: decision, icon: "?", classes: "bg-slate-100 text-slate-700 border-slate-300" };
    const big = opts && opts.big;
    return el(
      "span",
      { class: `inline-flex items-center gap-1.5 border rounded-full font-semibold ${v.classes} ${big ? "text-lg px-4 py-2" : "text-xs px-2.5 py-1"}` },
      el("span", {}, v.icon),
      el("span", {}, v.label)
    );
  }

  const SOURCE_BADGES = {
    document: { label: "From document", classes: "bg-blue-100 text-blue-800" },
    applicant_record: { label: "Bank record", classes: "bg-slate-100 text-slate-700" },
    derived: { label: "Derived", classes: "bg-purple-100 text-purple-800" },
  };

  function sourceBadge(source) {
    const b = SOURCE_BADGES[source] || { label: source, classes: "bg-slate-100 text-slate-700" };
    return el("span", { class: `text-xs px-2 py-0.5 rounded ${b.classes}` }, b.label);
  }

  function unresolvedBadge() {
    return el(
      "span",
      { class: "text-xs px-2 py-0.5 rounded bg-amber-100 text-amber-900 font-medium" },
      "Unresolved"
    );
  }

  function originBadge(origin) {
    return origin === "deterministic"
      ? el("span", { class: "text-xs px-2 py-0.5 rounded bg-slate-800 text-white" }, "Rule check")
      : el("span", { class: "text-xs px-2 py-0.5 rounded bg-indigo-100 text-indigo-800" }, "AI-assisted");
  }

  function provenanceBadge(provenance) {
    return provenance === "real_sbp_regulation"
      ? el("span", { class: "text-xs px-2 py-0.5 rounded bg-teal-100 text-teal-900 font-medium" }, "Real SBP regulation")
      : el("span", { class: "text-xs px-2 py-0.5 rounded bg-slate-200 text-slate-800 font-medium" }, "Simulated internal policy");
  }

  function fmtPKR(n) {
    if (n === null || n === undefined) return "—";
    return "PKR " + Number(n).toLocaleString("en-PK", { maximumFractionDigits: 0 });
  }

  function fmtDate(iso) {
    if (!iso) return "—";
    try {
      return new Date(iso).toLocaleString();
    } catch (e) {
      return iso;
    }
  }

  return {
    fetchJSON, el, skeleton, emptyState, errorState,
    verdictBadge, sourceBadge, unresolvedBadge, originBadge, provenanceBadge,
    fmtPKR, fmtDate,
  };
})();
