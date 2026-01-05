// static/js/app.js

function getPlotContext() {
  const ctxEl = document.getElementById("plot-ctx");
  if (!ctxEl) return null;
  try {
    return JSON.parse(ctxEl.textContent);
  } catch (e) {
    console.error("Invalid plot context JSON", e);
    return null;
  }
}

function buildApiUrl(ctx) {
  const { approach_id, subject_id, hand, file_name } = ctx;

  // Base unified API route
  const base =
    `/api/approach/${encodeURIComponent(approach_id)}` +
    `/${encodeURIComponent(subject_id)}` +
    `/${encodeURIComponent(hand)}` +
    `/${encodeURIComponent(file_name)}`;

  // ✅ If Approach 2: include activity as query param (if available)
  // (Approach 1 ignores it safely even if present)
  const activity = (ctx.activity || "").trim();
  if (activity) {
    return `${base}?activity=${encodeURIComponent(activity)}`;
  }

  return base;
}

async function loadAndRenderPlot() {
  const ctx = getPlotContext();
  if (!ctx) return;

  const apiUrl = buildApiUrl(ctx);

  const statusEl = document.getElementById("plot-status");
  const plotDiv = document.getElementById("main-plot");

  try {
    if (statusEl) statusEl.textContent = "Loading plot…";

    const res = await fetch(apiUrl, { cache: "no-store" });
    let data = null;

    try {
      data = await res.json();
    } catch {
      const txt = await res.text();
      if (statusEl) statusEl.textContent = `API returned non-JSON: ${txt.slice(0, 180)}`;
      return;
    }

    if (!res.ok || (data && data.error)) {
      if (statusEl) statusEl.textContent = (data && data.error) ? data.error : `API failed (${res.status})`;
      return;
    }

    // If backend hasn't built the figure yet
    if (!data.fig_json) {
      if (statusEl) statusEl.textContent = data.note || "No plot data yet for this file.";
      return;
    }

    const figObj = JSON.parse(data.fig_json);

    // Optional: ensure subtle gridlines (in case backend doesn’t)
    figObj.layout = figObj.layout || {};
    figObj.layout.xaxis = figObj.layout.xaxis || {};
    figObj.layout.yaxis = figObj.layout.yaxis || {};
    figObj.layout.xaxis.gridcolor = figObj.layout.xaxis.gridcolor || "rgba(255,255,255,0.08)";
    figObj.layout.yaxis.gridcolor = figObj.layout.yaxis.gridcolor || "rgba(255,255,255,0.08)";
    figObj.layout.paper_bgcolor = figObj.layout.paper_bgcolor || "rgba(0,0,0,0)";
    figObj.layout.plot_bgcolor = figObj.layout.plot_bgcolor || "rgba(0,0,0,0)";

    Plotly.newPlot(plotDiv, figObj.data, figObj.layout, {
      responsive: true,
      displaylogo: false,
      scrollZoom: true,
    });

    if (statusEl) statusEl.textContent = "";
  } catch (e) {
    console.error(e);
    if (statusEl) statusEl.textContent = "Failed to load plot.";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  if (typeof Plotly === "undefined") {
    console.error("Plotly not loaded");
    return;
  }
  loadAndRenderPlot();
});
