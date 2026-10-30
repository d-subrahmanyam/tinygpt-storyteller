// Story Studio front end: plain JavaScript, no build step.
const $ = (sel, root = document) => root.querySelector(sel);
const fmt = (n) => n.toLocaleString();
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

let contextSize = 512;

// ---------- tabs ----------
document.querySelectorAll(".tab").forEach((tab) =>
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab, .panel").forEach((el) => el.classList.remove("active"));
    tab.classList.add("active");
    $(`#${tab.dataset.tab}`).classList.add("active");
  }));

fetch("/api/info").then((r) => r.json()).then((info) => {
  const m = info.model;
  contextSize = m.context;
  const stories = Object.values(info.library).reduce((a, b) => a + b, 0);
  $("#model-info").textContent =
    `Your ${m.params_m}M-parameter storyteller · trained ${fmt(m.step)} steps · ` +
    `val loss ${m.val_loss} · ${fmt(stories)} stories · ${m.gpu}`;
  $("#stat-context").textContent = `🪟 0 / ${contextSize} context`;
});

// ---------- library ----------
let query = "", offset = 0;

function storyInfo(s) {
  return `#${fmt(s.id)} · ${s.split} · ${fmt(s.n_words)} words · ${fmt(s.n_tokens)} tokens`;
}

function resultCard(s, snippetHtml) {
  const card = document.createElement("div");
  card.className = "result";
  card.innerHTML = `<div class="info">${storyInfo(s)}</div><p class="snippet">${snippetHtml}</p>`;
  card.addEventListener("click", () => openStory(card, s.id), { once: true });
  return card;
}

async function openStory(card, id) {
  const s = await fetch(`/api/story/${id}`).then((r) => r.json());
  card.classList.add("open");
  $(".snippet", card).remove();
  const full = document.createElement("div");
  full.className = "full";
  full.innerHTML = `
    <div class="story-text">${esc(s.text)}</div>
    <div class="token-view" hidden></div>
    <div class="actions">
      <button class="btn tok-btn">🔤 Show tokens</button>
      <span class="muted">This is exactly what the model saw during training.</span>
    </div>`;
  card.appendChild(full);
  $(".tok-btn", full).addEventListener("click", (e) => toggleTokens(e.target, full, id));
}

async function toggleTokens(btn, full, id) {
  const view = $(".token-view", full);
  const text = $(".story-text", full);
  if (!view.hidden) {
    view.hidden = true; text.hidden = false; btn.textContent = "🔤 Show tokens";
    return;
  }
  if (!view.dataset.loaded) {
    const toks = await fetch(`/api/tokens/${id}`).then((r) => r.json());
    const spans = toks.map((t) => {
      const special = t.id === 50256;
      const shown = special ? "<|endoftext|>" : t.text.replace(/\n/g, "↵\n");
      return `<span class="tok${special ? " special" : ""}" title="token id ${t.id}">${esc(shown)}</span>`;
    }).join("");
    view.innerHTML = `<p class="muted">${fmt(toks.length)} tokens · hover a token to see its id</p><div class="tokens">${spans}</div>`;
    view.dataset.loaded = "1";
  }
  view.hidden = false; text.hidden = true; btn.textContent = "📖 Show text";
}

async function runSearch(more = false) {
  if (!more) { offset = 0; $("#results").innerHTML = ""; }
  $("#search-summary").textContent = "Searching…";
  const res = await fetch(`/api/search?q=${encodeURIComponent(query)}&offset=${offset}`).then((r) => r.json());
  $("#search-summary").textContent = res.total
    ? `${fmt(res.total)} stories match “${query}”`
    : `No stories match “${query}”`;
  res.results.forEach((s) => $("#results").appendChild(resultCard(s, s.snippet)));
  offset += res.results.length;
  $("#more-btn").hidden = offset >= res.total;
}

$("#search-form").addEventListener("submit", (e) => {
  e.preventDefault();
  query = $("#search-input").value.trim();
  if (query) runSearch();
});
$("#more-btn").addEventListener("click", () => runSearch(true));
$("#random-btn").addEventListener("click", async () => {
  const s = await fetch("/api/random").then((r) => r.json());
  $("#search-summary").textContent = "🎲 A random story from the training data";
  $("#more-btn").hidden = true;
  $("#results").innerHTML = "";
  const card = resultCard(s, esc(s.text.slice(0, 220)) + "…");
  $("#results").appendChild(card);
  card.click();
});

// ---------- storyteller ----------
let source = null, lastResult = null;

document.querySelectorAll(".chip").forEach((chip) =>
  chip.addEventListener("click", () => { $("#request").value = chip.textContent; }));

function setStep(id, state) {
  const el = $(`#step-${id}`);
  el.classList.remove("active", "done", "failed");
  if (state) el.classList.add(state);
}

function resetStory() {
  ["opening", "story", "check"].forEach((s) => setStep(s, null));
  $("#step-opening .opening-text").textContent = "";
  $("#step-opening .meta").textContent = "";
  $(".from-qwen").textContent = "";
  $(".from-gpt").textContent = "";
  $(".checks").innerHTML = "";
  $("#step-check .summary").textContent = "";
  $("#stat-tokens").textContent = "🔤 0 tokens";
  $("#stat-speed").textContent = "⚡ – tokens/sec";
  $("#stat-context").textContent = `🪟 0 / ${contextSize} context`;
  $("#context-bar").style.width = "0";
}

function startStory() {
  const request = $("#request").value.trim() || $("#request").placeholder;
  $("#request").value = request;
  if (source) source.close();
  resetStory();
  lastResult = null;
  $("#pipeline").hidden = false;
  $("#final").hidden = true;
  $("#tell-error").hidden = true;
  $("#save-note").textContent = "";
  $("#tell-btn").disabled = true;

  let checks = [];
  source = new EventSource(`/api/tell?request=${encodeURIComponent(request)}`);
  source.onmessage = (msg) => {
    const ev = JSON.parse(msg.data);
    switch (ev.type) {
      case "attempt":
        resetStory();
        $("#attempt").textContent = ev.attempt === 1 ? "" : `Attempt ${ev.attempt} of ${ev.max}: trying again with a fresh opening…`;
        setStep("opening", "active");
        break;
      case "opening":
        $("#step-opening .opening-text").textContent = ev.text;
        $("#step-opening .meta").textContent = `🔤 ${ev.tokens} Qwen tokens · ${ev.seconds}s`;
        setStep("opening", "done");
        setStep("story", "active");
        $(".from-qwen").textContent = ev.text;
        break;
      case "token": {
        $(".from-gpt").textContent += ev.text;
        $("#stat-tokens").textContent = `🔤 ${fmt(ev.count)} tokens`;
        $("#stat-speed").textContent = `⚡ ${fmt(ev.tok_s)} tokens/sec`;
        $("#stat-context").textContent = `🪟 ${ev.context} / ${contextSize} context`;
        $("#context-bar").style.width = `${(100 * ev.context) / contextSize}%`;
        break;
      }
      case "story_done":
        $("#stat-tokens").textContent = `🔤 ${fmt(ev.tokens)} tokens (+${ev.prompt_tokens} from the opening)`;
        $("#stat-speed").textContent = `⚡ ${fmt(ev.tok_s)} tokens/sec · ${ev.seconds}s`;
        setStep("story", "done");
        setStep("check", "active");
        lastResult = { tokens: ev.tokens, seconds: ev.seconds };
        break;
      case "check":
        checks = ev.checks;
        $(".checks").innerHTML = ev.checks.map((c) =>
          `<span class="badge ${c.pass ? "pass" : "fail"}">${c.pass ? "✅" : "❌"} ${esc(c.label)}</span>`).join("");
        $("#step-check .summary").textContent = ev.summary;
        setStep("check", ev.passed ? "done" : "failed");
        break;
      case "retry":
        break;  // the next "attempt" event resets the view
      case "done":
        source.close();
        $("#tell-btn").disabled = false;
        lastResult = { ...lastResult, request, opening: ev.opening, story: ev.story, checks, passed: ev.passed };
        $("#final-summary").textContent = ev.passed
          ? `✅ ${fmt(lastResult.tokens)} tokens · ${lastResult.seconds}s · written by your model` +
            (ev.attempts > 1 ? ` · took ${ev.attempts} tries` : "")
          : `⚠️ Best effort after ${ev.attempts} tries; not every check passed.`;
        $("#final").hidden = false;
        break;
      case "error":
        source.close();
        $("#tell-btn").disabled = false;
        $("#tell-error").textContent = `Something went wrong: ${ev.message}`;
        $("#tell-error").hidden = false;
        break;
    }
  };
  source.onerror = () => {
    source.close();
    $("#tell-btn").disabled = false;
    if ($("#final").hidden) {
      $("#tell-error").textContent = "Lost connection to the Story Studio server.";
      $("#tell-error").hidden = false;
    }
  };
}

$("#tell-form").addEventListener("submit", (e) => { e.preventDefault(); startStory(); });
$("#again-btn").addEventListener("click", startStory);
$("#save-btn").addEventListener("click", async () => {
  if (!lastResult) return;
  const { request, opening, story, checks, passed } = lastResult;
  const res = await fetch("/api/save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ request, opening, story, checks, passed }),
  }).then((r) => r.json());
  $("#save-note").textContent = `Saved to ${res.saved_to}`;
});
