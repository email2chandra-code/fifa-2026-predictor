/* FIFA 2026 cheat sheet — vanilla JS, Dixon-Coles match probs from iter 8 model */

const STORAGE_KEY = "fifa2026_cheat_sheet_v2";
const TZ_STORAGE_KEY = "fifa2026_viewer_timezone";
const BUNDLE_URL = "data/iter8_bundle.json";

let bundle = null;
let teamByName = {};
let teamGroupMap = {};

function lgamma(z) {
  const g = 7;
  const coef = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.984369578019571e-4, 1.5056327351493116e-7,
  ];
  if (z < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * z)) - lgamma(1 - z);
  }
  z -= 1;
  let x = coef[0];
  for (let i = 1; i < g + 2; i += 1) {
    x += coef[i] / (z + i);
  }
  const t = z + g + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}

function poissonPmf(k, lambda) {
  return Math.exp(k * Math.log(lambda) - lambda - lgamma(k + 1));
}

function tauFactor(h, a, lambdaHome, lambdaAway, rho) {
  if (h === 0 && a === 0) return 1.0 - lambdaHome * lambdaAway * rho;
  if (h === 0 && a === 1) return 1.0 + lambdaHome * rho;
  if (h === 1 && a === 0) return 1.0 + lambdaAway * rho;
  if (h === 1 && a === 1) return 1.0 - rho;
  return 1.0;
}

function dixonColesOutcomeProbs(lambdaHome, lambdaAway, rho, maxGoals = 10) {
  const probs = Array.from({ length: maxGoals + 1 }, () => new Array(maxGoals + 1).fill(0));
  for (let h = 0; h <= maxGoals; h += 1) {
    for (let a = 0; a <= maxGoals; a += 1) {
      const tau = tauFactor(h, a, lambdaHome, lambdaAway, rho);
      probs[h][a] = tau * poissonPmf(h, lambdaHome) * poissonPmf(a, lambdaAway);
    }
  }
  let total = 0;
  for (let h = 0; h <= maxGoals; h += 1) {
    for (let a = 0; a <= maxGoals; a += 1) {
      total += probs[h][a];
    }
  }
  if (total > 0) {
    for (let h = 0; h <= maxGoals; h += 1) {
      for (let a = 0; a <= maxGoals; a += 1) {
        probs[h][a] /= total;
      }
    }
  }
  let pHome = 0;
  let pDraw = 0;
  let pAway = 0;
  for (let h = 0; h <= maxGoals; h += 1) {
    for (let a = 0; a <= maxGoals; a += 1) {
      if (h > a) pHome += probs[h][a];
      else if (h === a) pDraw += probs[h][a];
      else pAway += probs[h][a];
    }
  }
  return { pHome, pDraw, pAway };
}

function computeLambdas(teamA, teamB) {
  const a = teamByName[teamA];
  const b = teamByName[teamB];
  if (!a || !b) return null;
  const lambdaHome = Math.exp(a.att - b.def);
  const lambdaAway = Math.exp(b.att - a.def);
  return { lambdaHome, lambdaAway };
}

function pct(x) {
  return `${(x * 100).toFixed(1)}%`;
}

function loadState() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
  } catch {
    return {};
  }
}

function saveState(state) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

function escHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/</g, "&lt;");
}

function allTeamNames() {
  return bundle.teams.map((t) => t.name).sort();
}

function buildTeamGroupMap() {
  const map = {};
  Object.entries(bundle.groups || {}).forEach(([grp, teams]) => {
    teams.forEach((name) => {
      map[name] = grp;
    });
  });
  return map;
}

function stageTeamLabel(team) {
  const grp = teamGroupMap[team];
  const safeTeam = escHtml(team);
  if (!grp) return safeTeam;
  return `${safeTeam} <span class="stage-team-group">(Group ${escHtml(grp)})</span>`;
}

function datalistOptionsHtml(names) {
  return names.map((n) => `<option value="${escHtml(n)}">`).join("");
}

function ensureDatalists() {
  if (!document.getElementById("teams-list-all")) {
    const dlAll = document.createElement("datalist");
    dlAll.id = "teams-list-all";
    dlAll.innerHTML = datalistOptionsHtml(allTeamNames());
    document.body.appendChild(dlAll);
  }
  Object.keys(bundle.groups)
    .sort()
    .forEach((grp) => {
      const id = `teams-list-group-${grp}`;
      if (document.getElementById(id)) return;
      const dl = document.createElement("datalist");
      dl.id = id;
      dl.innerHTML = datalistOptionsHtml([...(bundle.groups[grp] || [])].sort());
      document.body.appendChild(dl);
    });
}

/** Resolve typed text to a canonical team name (exact, case-insensitive, or unique prefix). */
function resolveTeamName(typed) {
  const t = typed.trim();
  if (!t) return { name: "", error: "empty" };
  if (teamByName[t]) return { name: t, error: null };

  const lower = t.toLowerCase();
  const exactCi = allTeamNames().find((n) => n.toLowerCase() === lower);
  if (exactCi) return { name: exactCi, error: null };

  const prefixMatches = allTeamNames().filter((n) => n.toLowerCase().startsWith(lower));
  if (prefixMatches.length === 1) return { name: prefixMatches[0], error: null };
  if (prefixMatches.length > 1) {
    return {
      name: "",
      error: `"${t}" matches several teams — pick one: ${prefixMatches.slice(0, 5).join(", ")}${prefixMatches.length > 5 ? "…" : ""}`,
    };
  }
  return { name: "", error: `No team found for "${t}". Type a few letters and pick from the list.` };
}

function inferredFourthPlace(grp, state) {
  const teams = bundle.groups[grp] || [];
  const picked = new Set(
    ["1st", "2nd", "3rd"]
      .map((p) => (state[`group_${grp}_${p}_team`] || "").trim())
      .filter(Boolean)
  );
  const remaining = teams.filter((t) => !picked.has(t));
  if (remaining.length === 1) return remaining[0];
  if (remaining.length === 0) return "—";
  return remaining.join(" · ");
}

function updateGroupFourthLabels(state) {
  document.querySelectorAll(".group-card").forEach((card) => {
    const grp = card.dataset.group;
    const el = card.querySelector(".group-fourth");
    if (el) el.textContent = inferredFourthPlace(grp, state);
  });
}

function teamInputHtml(listId, dataKey, value, placeholder) {
  const val = escHtml(value || "");
  const ph = escHtml(placeholder || "Type to search…");
  return `<input type="text" class="team-picker team-input" list="${listId}" data-key="${dataKey}" value="${val}" placeholder="${ph}" autocomplete="off" spellcheck="false">`;
}

function renderGroups(state) {
  const grid = document.getElementById("groups-grid");
  grid.innerHTML = "";
  const letters = Object.keys(bundle.groups).sort();

  letters.forEach((grp) => {
    const card = document.createElement("div");
    card.className = "group-card";
    card.dataset.group = grp;

    const teams = bundle.groups[grp] || [];
    const picks = bundle.model_picks[grp] || {};
    card.innerHTML = `
      <h3>Group ${grp}</h3>
      <div class="group-field group-header" aria-hidden="true">
        <span></span><span>Team</span><span class="col-for" title="Goals for">For</span><span class="col-against" title="Goals against">Against</span>
      </div>`;

    [
      { place: "1st", label: "1st", modelTeam: picks.first || "" },
      { place: "2nd", label: "2nd", modelTeam: picks.second || "" },
      {
        place: "3rd",
        label: "Best 3rd",
        modelTeam: picks.third_qualifier || "",
      },
    ].forEach(({ place, label, modelTeam }) => {
      const key = `group_${grp}_${place}`;
      const teamVal = state[`${key}_team`] || "";
      const field = document.createElement("div");
      field.className = "group-field";
      field.innerHTML = `
        <label title="${place === "3rd" ? "Model pick if this group supplies a best third-place qualifier" : ""}">${label}</label>
        ${teamInputHtml(`teams-list-group-${grp}`, `${key}_team`, teamVal, modelTeam || "Type team…")}
        <input type="number" class="score-input" min="0" max="99" inputmode="numeric" data-key="${key}_gf" value="${state[`${key}_gf`] ?? ""}" title="Goals for">
        <input type="number" class="score-input" min="0" max="99" inputmode="numeric" data-key="${key}_ga" value="${state[`${key}_ga`] ?? ""}" title="Goals against">
      `;
      card.appendChild(field);
    });

    const meta = document.createElement("p");
    meta.className = "model-value";
    meta.style.fontSize = "0.8rem";
    meta.style.color = "var(--muted)";
    const p3 = picks.p_third_qualifies
      ? ` · best 3rd: ${picks.third_qualifier || "?"} (${pct(picks.p_third_qualifies)} to qualify)`
      : "";
    meta.textContent = `Model: ${picks.first || "?"} / ${picks.second || "?"}${p3}`;
    card.appendChild(meta);

    const roster = document.createElement("p");
    roster.className = "group-roster";
    roster.innerHTML = `<strong>All 4 teams:</strong> ${teams.map((t) => escHtml(t)).join(" · ")}`;
    card.appendChild(roster);

    const fourth = document.createElement("p");
    fourth.className = "group-fourth-line";
    fourth.innerHTML = `4th place: <span class="group-fourth">${escHtml(inferredFourthPlace(grp, state))}</span>`;
    card.appendChild(fourth);

    grid.appendChild(card);
  });
}

function getViewerTimezone() {
  const sel = document.getElementById("viewer-timezone");
  const choice = sel?.value || localStorage.getItem(TZ_STORAGE_KEY) || "auto";
  if (choice && choice !== "auto") return choice;
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";
  } catch {
    return "UTC";
  }
}

function formatKickoffInTz(kickoffUtc, timeZone) {
  if (!kickoffUtc) return "—";
  try {
    const d = new Date(kickoffUtc);
    const time = new Intl.DateTimeFormat(undefined, {
      timeZone,
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    }).format(d);
    const date = new Intl.DateTimeFormat(undefined, {
      timeZone,
      month: "short",
      day: "numeric",
    }).format(d);
    return `${time}<br><span class="schedule-sub">${date}</span>`;
  } catch {
    return "—";
  }
}

function venueTimeCell(m) {
  const city = m.host_city || "";
  const local = m.time_local || "";
  const abbr = m.venue_timezone
    ? m.venue_timezone.split("/").pop().replace(/_/g, " ")
    : "";
  return `${escHtml(local)}<br><span class="schedule-sub">${escHtml(city)}${abbr ? ` · ${escHtml(abbr)}` : ""}</span>`;
}

function updateTimezoneLabel() {
  const label = document.getElementById("viewer-tz-label");
  if (!label) return;
  const tz = getViewerTimezone();
  label.textContent = `Your time zone: ${tz}`;
}

function renderSchedule() {
  const container = document.getElementById("schedule-table");
  if (!container || !bundle?.schedule?.length) {
    if (container) container.textContent = "Schedule not loaded — re-run export_ui_data.py.";
    return;
  }

  const viewerTz = getViewerTimezone();
  updateTimezoneLabel();

  const header = `
    <thead>
      <tr>
        <th>#</th><th>Date</th><th>Venue time</th><th>Your time</th><th>Match</th><th>Round</th><th>Stadium</th>
      </tr>
    </thead>`;
  const body = bundle.schedule
    .map((m) => {
      const round =
        m.stage === "group" ? `Group ${m.group}` : m.round || "Knockout";
      const venue = [m.host_city, m.stadium].filter(Boolean).join(" · ");
      const yourTime = formatKickoffInTz(m.kickoff_utc, viewerTz);
      return `<tr>
        <td>${m.match_number}</td>
        <td>${escHtml(m.date || "")}${m.day ? `<br><span class="schedule-sub">${escHtml(m.day)}</span>` : ""}</td>
        <td>${venueTimeCell(m)}</td>
        <td>${yourTime}</td>
        <td>${escHtml(m.team1)} vs ${escHtml(m.team2)}</td>
        <td>${escHtml(round)}</td>
        <td>${escHtml(venue)}</td>
      </tr>`;
    })
    .join("");

  container.innerHTML = `<table class="schedule-table">${header}<tbody>${body}</tbody></table>`;
}

function initTimezoneSelector() {
  const sel = document.getElementById("viewer-timezone");
  if (!sel) return;
  const saved = localStorage.getItem(TZ_STORAGE_KEY);
  if (saved) sel.value = saved;
  sel.addEventListener("change", () => {
    localStorage.setItem(TZ_STORAGE_KEY, sel.value);
    renderSchedule();
  });
}

function renderKnockout(state) {
  document.querySelectorAll(".slots").forEach((container) => {
    const round = container.dataset.round;
    const count = Number(container.dataset.count);
    container.innerHTML = "";
    for (let i = 0; i < count; i += 1) {
      const key = `ko_${round}_${i}`;
      const slot = document.createElement("div");
      slot.className = "ko-slot";
      slot.innerHTML = `
        ${teamInputHtml("teams-list-all", `${key}_t1`, state[`${key}_t1`] || "", "Team 1")}
        <span class="vs">vs</span>
        ${teamInputHtml("teams-list-all", `${key}_t2`, state[`${key}_t2`] || "", "Team 2")}
      `;
      container.appendChild(slot);
    }
  });
}

function bindPersistence() {
  const persist = (ev) => {
    const key = ev.target.dataset.key;
    if (!key) return;
    const state = loadState();
    state[key] = ev.target.value;
    saveState(state);
    if (key.startsWith("group_") && key.endsWith("_team")) {
      updateGroupFourthLabels(state);
    }
  };
  document.body.addEventListener("input", persist);
  document.body.addEventListener("change", persist);
}

function loadModelPicks() {
  const state = loadState();
  Object.entries(bundle.model_picks).forEach(([grp, picks]) => {
    state[`group_${grp}_1st_team`] = picks.first || "";
    state[`group_${grp}_2nd_team`] = picks.second || "";
    state[`group_${grp}_3rd_team`] = picks.third_qualifier || "";
  });
  saveState(state);
  renderGroups(state);
}

function loadStageNotes() {
  const sidebar = document.getElementById("stage-sidebar");
  const container = document.getElementById("stage-notes");
  sidebar.classList.remove("hidden");

  const rows = Object.entries(bundle.stage_probs)
    .filter(([team]) => teamGroupMap[team])
    .sort((a, b) => b[1].model_win_pct - a[1].model_win_pct)
    .slice(0, 24);

  container.innerHTML = `
    <div class="stage-note-row" style="font-weight:600">
      <span>Team</span><span>QF</span><span>SF</span><span>Final</span><span>Win</span>
    </div>
    ${rows
      .map(
        ([team, p]) => `
      <div class="stage-note-row" title="${escHtml(team)} (Group ${escHtml(teamGroupMap[team] || "")})">
        <span class="stage-team-name">${stageTeamLabel(team)}</span>
        <span>${pct(p.p_reach_qf)}</span>
        <span>${pct(p.p_reach_sf)}</span>
        <span>${pct(p.p_reach_final)}</span>
        <span>${pct(p.model_win_pct)}</span>
      </div>`
      )
      .join("")}
  `;
}

function predictMatch() {
  const rawA = document.getElementById("team-1").value;
  const rawB = document.getElementById("team-2").value;
  const out = document.getElementById("calc-results");
  const resA = resolveTeamName(rawA);
  const resB = resolveTeamName(rawB);
  if (resA.error === "empty" || resB.error === "empty") {
    out.textContent = "Type both team names (e.g. Mex), pick from the list, then Predict.";
    return;
  }
  if (resA.error) {
    out.textContent = `Team 1: ${resA.error}`;
    return;
  }
  if (resB.error) {
    out.textContent = `Team 2: ${resB.error}`;
    return;
  }
  const teamA = resA.name;
  const teamB = resB.name;
  document.getElementById("team-1").value = teamA;
  document.getElementById("team-2").value = teamB;
  const lambdas = computeLambdas(teamA, teamB);
  if (!lambdas) {
    out.textContent = "Select two valid teams.";
    return;
  }
  const rho = bundle.metadata.rho ?? -0.03;
  const { pHome, pDraw, pAway } = dixonColesOutcomeProbs(
    lambdas.lambdaHome,
    lambdas.lambdaAway,
    rho
  );
  out.textContent = `${teamA} win ${pct(pHome)} · Draw ${pct(pDraw)} · ${teamB} win ${pct(pAway)}`;
}

function clearSaved() {
  localStorage.removeItem(STORAGE_KEY);
  renderGroups({});
  renderKnockout({});
}

async function init() {
  const resp = await fetch(BUNDLE_URL);
  if (!resp.ok) {
    document.getElementById("last-updated").textContent =
      `Failed to load ${BUNDLE_URL}. Run scripts/export_ui_data.py first.`;
    return;
  }
  bundle = await resp.json();
  bundle.teams.forEach((t) => {
    teamByName[t.name] = t;
  });
  teamGroupMap = buildTeamGroupMap();

  document.getElementById("last-updated").textContent =
    `Model iter ${bundle.metadata.iteration} · cutoff ${bundle.metadata.cutoff} · updated ${bundle.metadata.last_updated}`;

  ensureDatalists();
  const state = loadState();
  renderGroups(state);
  renderKnockout(state);
  initTimezoneSelector();
  renderSchedule();
  bindPersistence();

  document.getElementById("btn-load-model").addEventListener("click", loadModelPicks);
  document.getElementById("btn-load-stage-notes").addEventListener("click", loadStageNotes);
  document.getElementById("btn-predict").addEventListener("click", predictMatch);
  document.getElementById("btn-clear").addEventListener("click", clearSaved);
  document.getElementById("print-blank").addEventListener("change", (ev) => {
    document.body.classList.toggle("print-blank", ev.target.checked);
  });
}

init();
