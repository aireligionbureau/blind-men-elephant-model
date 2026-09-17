const imageStage = document.querySelector("#imageStage");
const storyTrigger = document.querySelector("#storyTrigger");
const questionForm = document.querySelector("#questionForm");
const questionInput = document.querySelector("#questionInput");
const personTray = document.querySelector("#personTray");
const transition = document.querySelector("#transition");
const transitionPeople = document.querySelector("#transitionPeople");
const transitionQuestion = document.querySelector("#transitionQuestion");
const transitionKicker = document.querySelector("#transitionKicker");
const transitionTitle = document.querySelector("#transitionTitle");
const transitionCopy = document.querySelector("#transitionCopy");
const transitionCount = document.querySelector("#transitionCount");
const transitionResume = document.querySelector("#transitionResume");
const puzzlePieces = document.querySelector("#puzzlePieces");
const shadowTraces = document.querySelector("#shadowTraces");
const phaseSteps = [...document.querySelectorAll(".phase-step")];
const runState = document.querySelector("#runState");
const resumeRun = document.querySelector("#resumeRun");
const reportLink = document.querySelector("#reportLink");
const modelSettingsTrigger = document.querySelector("#modelSettingsTrigger");
const setupOverlay = document.querySelector("#setupOverlay");
const setupKicker = document.querySelector("#setupKicker");
const setupForm = document.querySelector("#setupForm");
const apiKeyInput = document.querySelector("#apiKeyInput");
const toggleApiKey = document.querySelector("#toggleApiKey");
const savedKeyHint = document.querySelector("#savedKeyHint");
const providerPreset = document.querySelector("#providerPreset");
const baseUrlField = document.querySelector("#baseUrlField");
const baseUrlInput = document.querySelector("#baseUrlInput");
const modelInput = document.querySelector("#modelInput");
const modelSuggestions = document.querySelector("#modelSuggestions");
const cohortModelInput = document.querySelector("#cohortModelInput");
const thinkingInput = document.querySelector("#thinkingInput");
const advancedSettings = document.querySelector("#advancedSettings");
const setupStatus = document.querySelector("#setupStatus");
const setupCancel = document.querySelector("#setupCancel");
const setupSubmit = document.querySelector("#setupSubmit");

const SVG_NS = "http://www.w3.org/2000/svg";
const PERSON_COUNT = 24;
const PHASE_ORDER = ["forensic", "detective", "contour"];
const ACTIVE_RUN_STATUSES = new Set(["queued", "running", "initialized"]);
const demoParams = new URLSearchParams(window.location.search);
const demoMode = demoParams.get("demo") === "transition";
const forceSetupPreview = demoParams.get("setup") === "1";
const PROVIDER_PRESETS = {
  deepseek: {
    label: "DeepSeek",
    base_url: "https://api.deepseek.com",
    model: "deepseek-v4-pro",
    cohort_model: "deepseek-v4-flash",
    supports_thinking: true,
    suggestions: ["deepseek-v4-pro", "deepseek-v4-flash", "deepseek-flash"],
  },
  openai: {
    label: "OpenAI",
    base_url: "https://api.openai.com/v1",
    model: "gpt-4.1",
    cohort_model: "gpt-4.1-mini",
    supports_thinking: false,
    suggestions: ["gpt-4.1", "gpt-4.1-mini", "gpt-4o"],
  },
};

let isGenerating = false;
let activeRunId = null;
let generatedPeople = 0;
let pollTimer = null;
let pollFailures = 0;
let transitionPending = false;
let transitionStarted = false;
let transitionPhase = "handoff";
let forensicCompleted = 0;
let latestStatus = null;
let reportHandoffTimer = null;
let reportNavigationTimer = null;
let settingsConfigured = false;
let settingsBusy = false;
let activeSettings = null;
let appStateRestored = false;
let previousMainModel = "";

storyTrigger.addEventListener("click", () => {
  if (isGenerating) return;
  if (!settingsConfigured && !demoMode) {
    openSetup({ required: true });
    return;
  }
  imageStage.classList.add("asking");
  window.setTimeout(() => questionInput.focus(), 160);
});

questionForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question || isGenerating) return;

  localStorage.setItem("bme:last-question", question);
  startDigitalPersonGeneration(question);
});

resumeRun.addEventListener("click", resumeActiveRun);
transitionResume.addEventListener("click", resumeActiveRun);
reportLink.addEventListener("click", markReportArrival);
modelSettingsTrigger.addEventListener("click", () => openSetup({ required: false }));
setupCancel.addEventListener("click", closeSetup);
toggleApiKey.addEventListener("click", () => {
  const reveal = apiKeyInput.type === "password";
  apiKeyInput.type = reveal ? "text" : "password";
  toggleApiKey.textContent = reveal ? "隐藏" : "显示";
  toggleApiKey.setAttribute("aria-pressed", reveal ? "true" : "false");
});
setupForm.addEventListener("submit", saveProviderSettings);
providerPreset.addEventListener("change", () => {
  const selected = providerPreset.value;
  const settings = selected === activeSettings?.provider_preset
    ? activeSettings
    : PROVIDER_PRESETS[selected] || {};
  baseUrlInput.value = settings.base_url || "";
  modelInput.value = settings.model || "";
  cohortModelInput.value = settings.cohort_model || settings.model || "";
  thinkingInput.checked = settings.supports_thinking === true;
  apiKeyInput.value = "";
  previousMainModel = modelInput.value;
  updateProviderFields();
});
baseUrlInput.addEventListener("input", updateProviderFields);
modelInput.addEventListener("input", () => {
  if (!cohortModelInput.value || cohortModelInput.value === previousMainModel) {
    cohortModelInput.value = modelInput.value;
  }
  previousMainModel = modelInput.value;
});
document.addEventListener("keydown", (event) => {
  if (
    event.key === "Escape"
    && !setupOverlay.hidden
    && settingsConfigured
    && !settingsBusy
  ) {
    closeSetup();
  }
});

questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.isComposing && !isGenerating) {
    event.preventDefault();
    if (typeof questionForm.requestSubmit === "function") {
      questionForm.requestSubmit();
    } else {
      questionForm.dispatchEvent(new Event("submit", { cancelable: true, bubbles: true }));
    }
    return;
  }

  if (event.key === "Escape" && !isGenerating) {
    imageStage.classList.remove("asking");
    questionInput.blur();
  }
});

async function startDigitalPersonGeneration(question) {
  isGenerating = true;
  imageStage.classList.remove("asking");
  imageStage.classList.add("generating");
  personTray.innerHTML = "";
  runState.textContent = "正在创建本题的数字人群";
  resumeRun.hidden = true;
  questionInput.disabled = true;
  questionInput.value = question;
  generatedPeople = 0;
  resetTransition();

  try {
    const response = await fetch("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!response.ok) {
      const details = await responseErrorDetails(response);
      const error = new Error(details.message);
      error.code = details.code;
      throw error;
    }
    const job = await response.json();
    activeRunId = job.run_id;
    localStorage.setItem("bme:active-run", activeRunId);
    updateFromStatus(job);
    schedulePoll(400);
  } catch (error) {
    showRunError(error.message || "任务没有成功提交");
    if (error.code === "provider_not_configured") {
      settingsConfigured = false;
      openSetup({ required: true });
    }
  }
}

async function bootstrapApp() {
  if (demoMode) {
    settingsConfigured = true;
    questionInput.disabled = false;
    runTransitionDemo();
    return;
  }

  try {
    const response = await fetch("/api/settings", { cache: "no-store" });
    if (!response.ok) throw new Error(await responseError(response));
    const settings = await response.json();
    applySettingsToForm(settings);
    modelSettingsTrigger.hidden = false;
    settingsConfigured = Boolean(settings.configured);
    questionInput.disabled = !settingsConfigured;
    if (!settings.configured || forceSetupPreview) {
      openSetup({ required: !settings.configured });
      if (!settings.configured) return;
    }
    await restoreAppStateOnce();
  } catch (error) {
    applySettingsToForm(defaultSettings());
    openSetup({ required: true });
    setSetupStatus(error.message || "暂时无法读取模型设置。", "error");
  }
}

async function saveProviderSettings(event) {
  event.preventDefault();
  if (settingsBusy) return;
  settingsBusy = true;
  setSetupBusy(true);
  setSetupStatus("正在验证模型接口，不会产生推理费用。", "");

  try {
    const response = await fetch("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        api_key: apiKeyInput.value.trim(),
        provider_preset: providerPreset.value,
        base_url: baseUrlInput.value.trim(),
        model: modelInput.value.trim(),
        cohort_model: cohortModelInput.value.trim(),
        supports_thinking: thinkingInput.checked,
      }),
    });
    if (!response.ok) {
      const details = await responseErrorDetails(response);
      throw new Error(details.message);
    }
    const settings = await response.json();
    applySettingsToForm(settings);
    settingsConfigured = true;
    questionInput.disabled = false;
    modelSettingsTrigger.hidden = false;
    const probe = settings.probe || {};
    const unverified = probe.verified === false || probe.model_available === false;
    setSetupStatus(
      unverified
        ? "设置已保存；此接口未能确认所选模型，请在运行前核对。"
        : "连接成功，已经可以开始提问。",
      unverified ? "warning" : "success",
    );
    window.setTimeout(async () => {
      closeSetup();
      await restoreAppStateOnce();
    }, unverified ? 1800 : 520);
  } catch (error) {
    setSetupStatus(error.message || "模型接口没有通过验证。", "error");
  } finally {
    settingsBusy = false;
    setSetupBusy(false);
  }
}

function applySettingsToForm(settings) {
  activeSettings = settings;
  providerPreset.value = settings.provider_preset || "deepseek";
  baseUrlInput.value = settings.base_url || "";
  modelInput.value = settings.model || "";
  cohortModelInput.value = settings.cohort_model || settings.model || "";
  thinkingInput.checked = settings.supports_thinking !== false;
  apiKeyInput.value = "";
  previousMainModel = modelInput.value;
  updateProviderFields();
  setupCancel.hidden = !settings.configured;
  setupSubmit.textContent = settings.configured ? "验证并保存" : "验证并开始";
  modelSettingsTrigger.textContent = settings.configured
    ? `模型设置 · ${PROVIDER_PRESETS[settings.provider_preset]?.label || "自定义"}`
    : "模型设置";
}

function defaultSettings() {
  return {
    configured: false,
    provider_preset: "deepseek",
    ...PROVIDER_PRESETS.deepseek,
    supports_thinking: true,
    api_key_masked: "",
  };
}

function updateProviderFields() {
  const custom = providerPreset.value === "openai-compatible";
  baseUrlField.hidden = !custom;
  baseUrlInput.required = custom;
  const sameProvider = providerPreset.value === activeSettings?.provider_preset;
  const sameEndpoint = baseUrlInput.value.trim().replace(/\/+$/, "")
    === (activeSettings?.base_url || "").replace(/\/+$/, "");
  const canReuseKey = Boolean(activeSettings?.configured && sameProvider && sameEndpoint);
  apiKeyInput.required = !canReuseKey;
  const keyMask = activeSettings?.api_key_masked || "";
  savedKeyHint.textContent = canReuseKey && keyMask
    ? `已保存密钥 ${keyMask}；留空即可继续使用。`
    : "请填写所选模型服务的 API Key。";
  modelSuggestions.replaceChildren();
  for (const name of PROVIDER_PRESETS[providerPreset.value]?.suggestions || []) {
    const option = document.createElement("option");
    option.value = name;
    modelSuggestions.append(option);
  }
}

function openSetup({ required }) {
  if (!activeSettings) applySettingsToForm(defaultSettings());
  setupKicker.textContent = required ? "首次使用" : "模型设置";
  setupCancel.hidden = required || !settingsConfigured;
  setupOverlay.hidden = false;
  setupOverlay.setAttribute("aria-hidden", "false");
  setupStatus.textContent = "";
  setupStatus.className = "setup-status";
  window.setTimeout(() => apiKeyInput.focus(), 80);
}

function closeSetup() {
  if (!settingsConfigured || settingsBusy) return;
  setupOverlay.hidden = true;
  setupOverlay.setAttribute("aria-hidden", "true");
  apiKeyInput.type = "password";
  toggleApiKey.textContent = "显示";
  toggleApiKey.setAttribute("aria-pressed", "false");
  advancedSettings.open = false;
  modelSettingsTrigger.focus();
}

function setSetupBusy(busy) {
  setupForm.setAttribute("aria-busy", busy ? "true" : "false");
  setupSubmit.disabled = busy;
  setupCancel.disabled = busy;
  toggleApiKey.disabled = busy;
  providerPreset.disabled = busy;
  baseUrlInput.disabled = busy;
  modelInput.disabled = busy;
  cohortModelInput.disabled = busy;
  thinkingInput.disabled = busy;
  setupSubmit.textContent = busy
    ? "正在验证..."
    : activeSettings?.configured
      ? "验证并保存"
      : "验证并开始";
}

function setSetupStatus(message, state) {
  setupStatus.textContent = message;
  setupStatus.className = `setup-status${state ? ` ${state}` : ""}`;
}

async function restoreAppStateOnce() {
  if (appStateRestored) return;
  appStateRestored = true;
  await restoreActiveRun();
}

async function resumeActiveRun() {
  if (!activeRunId) return;
  resumeRun.hidden = true;
  transitionResume.hidden = true;
  isGenerating = true;
  runState.textContent = "正在从保存的检查点继续";
  if (transitionStarted) {
    transitionKicker.textContent = "进度已经恢复";
    transitionTitle.textContent = "从保存的位置继续拼合";
    transitionCopy.textContent = "已经完成的材料不会重跑，模型正在继续未完成的部分。";
  }
  try {
    const response = await fetch(`/api/runs/${encodeURIComponent(activeRunId)}/resume`, {
      method: "POST",
    });
    if (!response.ok) throw new Error(await responseError(response));
    schedulePoll(300);
  } catch (error) {
    showRunError(error.message || "暂时无法继续运行");
  }
}

function ensurePersonIcons(target) {
  const count = Math.min(PERSON_COUNT, Math.max(0, target));
  while (generatedPeople < count) {
    generatedPeople += 1;
    personTray.appendChild(createPersonIcon(generatedPeople));
  }
}

async function pollRun() {
  if (!activeRunId) return;
  try {
    const response = await fetch(`/api/runs/${encodeURIComponent(activeRunId)}`, {
      cache: "no-store",
    });
    if (!response.ok) throw new Error(await responseError(response));
    const status = await response.json();
    pollFailures = 0;

    if (status.status === "unknown") {
      clearInactiveRun("上一次任务记录已经失效，可以输入新的问题");
      return;
    }

    updateFromStatus(status);

    if (status.status === "succeeded" && status.report_ready) {
      isGenerating = false;
      ensurePersonIcons(PERSON_COUNT);
      localStorage.removeItem("bme:active-run");
      reportLink.href = status.report_url;
      if (!transitionStarted) enterPuzzleTransition(status, true);
      setTransitionPhase("complete", status);
      scheduleReportHandoff(status.report_url);
      return;
    }
    if (status.status === "recoverable_failed") {
      showRecoverableRun(status);
      return;
    }
    if (status.status === "failed") {
      showRunError(status.error || "运行没有通过系统检查");
      return;
    }
    schedulePoll(1200);
  } catch (error) {
    pollFailures += 1;
    if (pollFailures <= 8) {
      const message = "连接暂时中断，正在重新获取运行状态";
      runState.textContent = message;
      if (transitionStarted) transitionCount.textContent = message;
      schedulePoll(Math.min(1000 * pollFailures, 6000));
    } else {
      showRunError(error.message || "无法获取运行状态");
    }
  }
}

function schedulePoll(delay) {
  window.clearTimeout(pollTimer);
  pollTimer = window.setTimeout(pollRun, delay);
}

function updateFromStatus(status) {
  latestStatus = status;
  const cohortCompleted = Number(status.cohort?.completed || 0);
  if (cohortCompleted > 0) ensurePersonIcons(cohortCompleted);
  const completed = Number(status.digital_people?.completed || 0);
  if (completed > 0) ensurePersonIcons(completed);
  updateHomeStatus(status);

  if (cohortCompleted >= PERSON_COUNT && !transitionStarted && !transitionPending) {
    transitionPending = true;
    window.setTimeout(() => {
      transitionPending = false;
      if (!transitionStarted) enterPuzzleTransition(latestStatus || status);
    }, 720);
  }

  if (transitionStarted) updatePuzzleTransition(status);
}

function updateHomeStatus(status) {
  const completed = Number(status.digital_people?.completed || 0);
  if (status.current_stage === "streaming_front") {
    const searched = Number(status.retrieval?.completed || 0);
    const thought = Number(status.digital_people?.completed || 0);
    const diagnosed = Number(status.semantic_verdicts?.completed || 0);
    runState.textContent = `搜索 ${searched}/${PERSON_COUNT} · 思考 ${thought}/${PERSON_COUNT} · 诊断 ${diagnosed}/${PERSON_COUNT}`;
    return;
  }
  if (status.current_stage === "retrieval") {
    const searched = Number(status.retrieval?.completed || 0);
    const total = Number(status.retrieval?.total || PERSON_COUNT);
    const failed = Number(status.retrieval?.failed || 0);
    const suffix = failed > 0 ? `，${failed} 人正在换备用来源` : "";
    runState.textContent = `数字人带着过滤器搜索 ${searched}/${total}${suffix}`;
    return;
  }
  if (status.current_stage === "digital_people" && completed > 0) {
    runState.textContent = `数字人正在形成完整思考链 ${completed}/${PERSON_COUNT}`;
    return;
  }
  runState.textContent = `${stageName(status.current_stage)} · ${Number(status.completed_stage_count || 0)}/9`;
}

function enterPuzzleTransition(status, instant = false) {
  if (transitionStarted) return;
  transitionStarted = true;
  transitionQuestion.textContent = status?.question || questionInput.value.trim();
  buildPuzzle();
  buildTransitionPeople();
  document.body.classList.add("puzzle-running");
  transition.classList.add("show", "phase-handoff");
  transition.setAttribute("aria-hidden", "false");

  const positionDelay = instant ? 20 : 80;
  window.setTimeout(positionTransitionPeople, positionDelay);
  window.setTimeout(() => {
    const current = latestStatus || status || {};
    setTransitionPhase(deriveTransitionPhase(current), current);
  }, instant ? 40 : 1220);
}

function updatePuzzleTransition(status) {
  const phase = deriveTransitionPhase(status);
  setTransitionPhase(phase, status);

  let diagnosed = Number(status.semantic_verdicts?.completed || 0);
  const forensicStatus = status.analysis_layers?.forensic?.status;
  if (forensicStatus === "succeeded" || PHASE_ORDER.indexOf(phase) > 0) {
    diagnosed = PERSON_COUNT;
  }
  setForensicProgress(diagnosed);

  if (phase === "forensic") {
    const searched = Number(status.retrieval?.completed || 0);
    const thought = Number(status.digital_people?.completed || 0);
    transitionCount.textContent = `已经搜索 ${searched}/${PERSON_COUNT} · 完成思考 ${thought}/${PERSON_COUNT} · 看清思考链 ${diagnosed}/${PERSON_COUNT}`;
  }
}

function deriveTransitionPhase(status) {
  if (status?.status === "succeeded" && status?.report_ready) return "complete";

  const layers = status?.analysis_layers || {};
  const contourStatus = layers.contour?.status;
  const detectiveStatus = layers.detective?.status;
  if (["running", "succeeded"].includes(contourStatus)) return "contour";
  if (["running", "succeeded"].includes(detectiveStatus)) return "detective";

  const current = status?.current_stage;
  if (current === "finalize") return "contour";
  if (current === "relational_synthesis") {
    const checkpoint = status?.synthesis || {};
    return checkpoint.detective_status === "succeeded" ? "contour" : "detective";
  }
  if (current === "shadow_puzzle") return "detective";
  return "forensic";
}

function setTransitionPhase(phase, status = {}) {
  if (!transitionStarted) return;
  if (!phase) phase = "forensic";
  transitionPhase = phase;
  transition.classList.remove(
    "phase-handoff",
    "phase-forensic",
    "phase-detective",
    "phase-contour",
    "phase-complete",
  );
  transition.classList.add(`phase-${phase}`);

  const messages = {
    handoff: {
      kicker: "24 种认知位置已经到场",
      title: "每个人正在交出自己的碎片",
      copy: "他们从不同位置出发，也带着各自的信息过滤器。",
      count: "",
    },
    forensic: {
      kicker: "法医层正在工作",
      title: "正在逐个检查：他们究竟在哪里看偏了",
      copy: "沿着信息获取、证据选择、核心假设、推理路径和结论，完整检查每个人的思考链。",
      count: `已经看清 ${forensicCompleted}/${PERSON_COUNT} 个人的完整思考链`,
    },
    detective: {
      kicker: "侦探层正在工作",
      title: "正在让彼此的碎片发生关系",
      copy: "互相支持的材料靠近，彼此冲突的材料试拼后分开，共同没看见的地方逐渐显出缺口。",
      count: "零散材料正在形成局部拼图",
    },
    contour: {
      kicker: "轮廓层正在工作",
      title: "正在从局部关系反演整体",
      copy: "可靠部分成为锚点，错误材料暴露边界，无法确认的地方被如实保留。",
      count: "轮廓正在显现，但不会被强行填满",
    },
    complete: {
      kicker: "当前真相轮廓已经显现",
      title: "它不完整，却已经无法被误认成别的东西",
      copy: "它有缺口、有磨损，仍有碎片散落在外，但它已经不再是柱子、蛇、扇子或绳子。",
      count: "分析完成，正在展开结果",
    },
  };
  const message = messages[phase] || messages.forensic;
  transitionKicker.textContent = message.kicker;
  transitionTitle.textContent = message.title;
  transitionCopy.textContent = message.copy;
  transitionCount.textContent = message.count;
  updatePhaseTrack(phase);

  if (PHASE_ORDER.indexOf(phase) > 0 || phase === "complete") {
    setForensicProgress(PERSON_COUNT);
  }
  if (phase === "complete") {
    reportLink.hidden = !(status?.report_url || reportLink.getAttribute("href") !== "#");
  }
}

function updatePhaseTrack(phase) {
  const activeIndex = phase === "complete" ? PHASE_ORDER.length : PHASE_ORDER.indexOf(phase);
  phaseSteps.forEach((step, index) => {
    step.classList.toggle("active", index === activeIndex);
    step.classList.toggle("done", index < activeIndex || phase === "complete");
  });
}

function setForensicProgress(value) {
  const next = Math.min(PERSON_COUNT, Math.max(forensicCompleted, Number(value || 0)));
  forensicCompleted = next;
  [...transitionPeople.children].forEach((person, index) => {
    person.classList.toggle("offered", index < next);
  });
  [...puzzlePieces.children].forEach((piece) => {
    const owner = Number(piece.dataset.person || 0);
    piece.classList.toggle("revealed", owner < next);
  });
  if (transitionPhase === "forensic") {
    transitionCount.textContent = `已经看清 ${next}/${PERSON_COUNT} 个人的完整思考链`;
  }
}

function buildTransitionPeople() {
  transitionPeople.innerHTML = "";
  ensurePersonIcons(PERSON_COUNT);
  const sourcePeople = [...personTray.querySelectorAll(".person-icon")];
  sourcePeople.slice(0, PERSON_COUNT).forEach((source, index) => {
    const rect = source.getBoundingClientRect();
    const clone = document.createElement("span");
    clone.className = `transition-person person-variant-${(index % 4) + 1}`;
    clone.dataset.index = String(index);
    clone.innerHTML = source.innerHTML;
    clone.style.left = `${rect.left + rect.width / 2}px`;
    clone.style.top = `${rect.top + rect.height / 2}px`;
    transitionPeople.appendChild(clone);
  });
}

function positionTransitionPeople() {
  [...transitionPeople.children].forEach((person, index) => {
    const target = perimeterPosition(index);
    person.style.left = `${(target.x / 1200) * 100}%`;
    person.style.top = `${(target.y / 720) * 100}%`;
  });
}

function buildPuzzle() {
  if (puzzlePieces.childElementCount > 0) return;
  const columns = 11;
  const rows = 7;
  const xStart = 175;
  const yStart = 170;
  const width = 875;
  const height = 475;
  const cellWidth = width / columns;
  const cellHeight = height / rows;
  const horizontalEdges = Array.from({ length: rows - 1 }, (_, row) =>
    Array.from({ length: columns }, (_, column) => deterministicSign(row, column, 17)),
  );
  const verticalEdges = Array.from({ length: rows }, (_, row) =>
    Array.from({ length: columns - 1 }, (_, column) => deterministicSign(row, column, 41)),
  );

  let visibleIndex = 0;
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const x = xStart + column * cellWidth;
      const y = yStart + row * cellHeight;
      if (!cellTouchesElephant(x, y, cellWidth, cellHeight)) continue;

      const edges = {
        top: row === 0 ? 0 : -horizontalEdges[row - 1][column],
        right: column === columns - 1 ? 0 : verticalEdges[row][column],
        bottom: row === rows - 1 ? 0 : horizontalEdges[row][column],
        left: column === 0 ? 0 : -verticalEdges[row][column - 1],
      };
      const pathData = jigsawPath(x, y, cellWidth, cellHeight, edges);
      const piece = document.createElementNS(SVG_NS, "g");
      const path = document.createElementNS(SVG_NS, "path");
      const role = pieceRole(visibleIndex);
      const owner = visibleIndex % PERSON_COUNT;
      const centerX = x + cellWidth / 2;
      const centerY = y + cellHeight / 2;
      const scatter = scatterTransform(owner, visibleIndex, centerX, centerY);
      const island = islandTransform(visibleIndex, centerX, centerY);
      const final = finalTransform(role, owner, visibleIndex, centerX, centerY);

      piece.setAttribute("clip-path", "url(#elephantClip)");
      piece.setAttribute("class", `puzzle-piece piece-${role}`);
      piece.dataset.person = String(owner);
      piece.dataset.role = role;
      setTransformVariables(piece, "scatter", scatter);
      setTransformVariables(piece, "island", island);
      setTransformVariables(piece, "final", final);
      piece.style.setProperty("--pulse-delay", `${(visibleIndex % 9) * -0.37}s`);
      path.setAttribute("d", pathData);
      piece.appendChild(path);
      puzzlePieces.appendChild(piece);

      if (visibleIndex % 3 === 0) {
        const trace = document.createElementNS(SVG_NS, "path");
        trace.setAttribute("class", "shadow-trace");
        trace.setAttribute("clip-path", "url(#elephantClip)");
        trace.setAttribute("d", pathData);
        trace.style.setProperty("--trace-x", `${island.x + jitter(visibleIndex, 5, 18)}px`);
        trace.style.setProperty("--trace-y", `${island.y + jitter(visibleIndex, 9, 14)}px`);
        trace.style.setProperty("--trace-r", `${island.r + jitter(visibleIndex, 3, 4)}deg`);
        shadowTraces.appendChild(trace);
      }
      visibleIndex += 1;
    }
  }
}

function jigsawPath(x, y, width, height, edges) {
  const tab = Math.min(width, height) * .2;
  const x38 = x + width * .38;
  const x50 = x + width * .5;
  const x62 = x + width * .62;
  const y38 = y + height * .38;
  const y50 = y + height * .5;
  const y62 = y + height * .62;
  const right = x + width;
  const bottom = y + height;
  const p = (value) => Number(value.toFixed(2));
  return [
    `M${p(x)} ${p(y)}`,
    `L${p(x38)} ${p(y)}`,
    `C${p(x38)} ${p(y - edges.top * tab * .18)} ${p(x + width * .42)} ${p(y - edges.top * tab)} ${p(x50)} ${p(y - edges.top * tab)}`,
    `C${p(x + width * .58)} ${p(y - edges.top * tab)} ${p(x62)} ${p(y - edges.top * tab * .18)} ${p(x62)} ${p(y)}`,
    `L${p(right)} ${p(y)}`,
    `L${p(right)} ${p(y38)}`,
    `C${p(right + edges.right * tab * .18)} ${p(y38)} ${p(right + edges.right * tab)} ${p(y + height * .42)} ${p(right + edges.right * tab)} ${p(y50)}`,
    `C${p(right + edges.right * tab)} ${p(y + height * .58)} ${p(right + edges.right * tab * .18)} ${p(y62)} ${p(right)} ${p(y62)}`,
    `L${p(right)} ${p(bottom)}`,
    `L${p(x62)} ${p(bottom)}`,
    `C${p(x62)} ${p(bottom + edges.bottom * tab * .18)} ${p(x + width * .58)} ${p(bottom + edges.bottom * tab)} ${p(x50)} ${p(bottom + edges.bottom * tab)}`,
    `C${p(x + width * .42)} ${p(bottom + edges.bottom * tab)} ${p(x38)} ${p(bottom + edges.bottom * tab * .18)} ${p(x38)} ${p(bottom)}`,
    `L${p(x)} ${p(bottom)}`,
    `L${p(x)} ${p(y62)}`,
    `C${p(x - edges.left * tab * .18)} ${p(y62)} ${p(x - edges.left * tab)} ${p(y + height * .58)} ${p(x - edges.left * tab)} ${p(y50)}`,
    `C${p(x - edges.left * tab)} ${p(y + height * .42)} ${p(x - edges.left * tab * .18)} ${p(y38)} ${p(x)} ${p(y38)}`,
    `L${p(x)} ${p(y)}Z`,
  ].join("");
}

function cellTouchesElephant(x, y, width, height) {
  const samples = [
    [x + width * .15, y + height * .15],
    [x + width * .85, y + height * .15],
    [x + width * .5, y + height * .5],
    [x + width * .15, y + height * .85],
    [x + width * .85, y + height * .85],
  ];
  return samples.some(([sampleX, sampleY]) => insideElephant(sampleX, sampleY));
}

function insideElephant(x, y) {
  const ellipse = (cx, cy, rx, ry) => (((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2) <= 1;
  const inBody = ellipse(565, 360, 292, 165);
  const inHead = ellipse(872, 360, 112, 112);
  const inTrunk = x >= 900 && x <= 1025 && y >= 375 && y <= 635;
  const inLegs = y >= 445 && y <= 625 && x >= 325 && x <= 825;
  const inTail = x >= 175 && x <= 315 && y >= 290 && y <= 420;
  return inBody || inHead || inTrunk || inLegs || inTail;
}

function pieceRole(index) {
  if ([8, 22, 37, 53].includes(index)) return "unresolved";
  if ([5, 16, 31, 45, 57].includes(index)) return "unseen";
  if (index % 4 === 2 || index % 9 === 3) return "distorted";
  return "seen";
}

function scatterTransform(owner, index, centerX, centerY) {
  const anchor = perimeterPosition(owner);
  return {
    x: anchor.x + jitter(index, 11, 55) - centerX,
    y: anchor.y + jitter(index, 23, 42) - centerY,
    r: jitter(index, 31, 64),
  };
}

function islandTransform(index, centerX, centerY) {
  let cluster = { x: -145, y: -58, r: -5 };
  if (centerX >= 820) {
    cluster = { x: 145, y: -12, r: 7 };
  } else if (centerY >= 445) {
    cluster = { x: -10, y: 112, r: -3 };
  } else if (centerX >= 555) {
    cluster = { x: 44, y: -112, r: 4 };
  } else if (centerX < 310) {
    cluster = { x: -128, y: 24, r: -8 };
  }
  return {
    x: cluster.x + jitter(index, 107, 5),
    y: cluster.y + jitter(index, 109, 5),
    r: cluster.r + jitter(index, 113, 1.8),
  };
}

function finalTransform(role, owner, index, centerX, centerY) {
  if (role === "unresolved") {
    const anchor = perimeterPosition((owner + 7) % PERSON_COUNT);
    return {
      x: anchor.x + jitter(index, 47, 34) - centerX,
      y: anchor.y + jitter(index, 61, 30) - centerY,
      r: jitter(index, 71, 38),
    };
  }
  if (role === "distorted") {
    return {
      x: jitter(index, 83, 7),
      y: jitter(index, 97, 6),
      r: jitter(index, 101, 3.2),
    };
  }
  return { x: 0, y: 0, r: 0 };
}

function setTransformVariables(node, prefix, values) {
  node.style.setProperty(`--${prefix}-x`, `${values.x}px`);
  node.style.setProperty(`--${prefix}-y`, `${values.y}px`);
  node.style.setProperty(`--${prefix}-r`, `${values.r}deg`);
}

function perimeterPosition(index) {
  const side = Math.floor(index / 6);
  const offset = index % 6;
  const progress = (offset + 1) / 7;
  if (side === 0) return { x: 125 + progress * 950, y: 112 };
  if (side === 1) return { x: 1110, y: 105 + progress * 510 };
  if (side === 2) return { x: 1075 - progress * 950, y: 548 };
  return { x: 90, y: 615 - progress * 500 };
}

function deterministicSign(row, column, salt) {
  return ((row * 37 + column * 53 + salt) % 2) === 0 ? 1 : -1;
}

function jitter(index, salt, range) {
  const value = Math.sin((index + 1) * (salt + 3) * 12.9898) * 43758.5453;
  const normalized = value - Math.floor(value);
  return Number(((normalized * 2 - 1) * range).toFixed(2));
}

function scheduleReportHandoff(url) {
  window.clearTimeout(reportHandoffTimer);
  window.clearTimeout(reportNavigationTimer);
  reportLink.href = url;
  reportLink.hidden = false;
  if (demoMode || !url || url === "#") return;
  markReportArrival();
  reportHandoffTimer = window.setTimeout(() => {
    transition.classList.add("report-handoff");
  }, 2200);
  reportNavigationTimer = window.setTimeout(() => {
    window.location.assign(url);
  }, 3600);
}

function markReportArrival() {
  if (!reportLink.href || reportLink.getAttribute("href") === "#") return;
  try {
    window.sessionStorage.setItem("bme:puzzle-handoff", "1");
  } catch (_error) {
    // The report remains reachable when storage is unavailable.
  }
}

function showRecoverableRun(status) {
  isGenerating = false;
  questionInput.disabled = false;
  const message = `运行暂时停在${stageName(status.current_stage)}，进度已经保存`;
  runState.textContent = message;
  resumeRun.hidden = false;
  if (transitionStarted) {
    transitionKicker.textContent = "拼图已经安全停住";
    transitionTitle.textContent = "已经完成的部分都还在";
    transitionCopy.textContent = message;
    transitionCount.textContent = "继续后会从保存的位置接上";
    transitionResume.hidden = false;
  }
  // Keep a quiet watch on the saved run. A restarted service may resume it
  // automatically, and the page should follow without asking the user to refresh.
  schedulePoll(12000);
}

function stageName(stage) {
  const labels = {
    preflight: "检查运行条件",
    cohort: "生成数字人群",
    retrieval: "带着过滤器搜索",
    digital_people: "数字人独立思考",
    diagnosis: "逐人审视思考链",
    semantic_verdicts: "点明认知阴影",
    streaming_front: "搜索、思考与逐人诊断并行推进",
    shadow_puzzle: "整理阴影材料",
    relational_synthesis: "连接材料并拼出轮廓",
    finalize: "生成结果页面",
  };
  return labels[stage] || "准备运行";
}

function showRunError(message) {
  window.clearTimeout(pollTimer);
  isGenerating = false;
  questionInput.disabled = false;
  runState.textContent = message;
  if (activeRunId) resumeRun.hidden = false;
  if (transitionStarted) {
    transitionKicker.textContent = "拼图暂时没有继续移动";
    transitionTitle.textContent = "进度仍然保存在原处";
    transitionCopy.textContent = message;
    transitionCount.textContent = "可以从保存位置继续，不会丢失已经完成的分析";
    transitionResume.hidden = !activeRunId;
    return;
  }
  questionInput.focus();
}

function clearInactiveRun(message) {
  window.clearTimeout(pollTimer);
  localStorage.removeItem("bme:active-run");
  activeRunId = null;
  isGenerating = false;
  pollFailures = 0;
  questionInput.disabled = false;
  imageStage.classList.remove("generating");
  resumeRun.hidden = true;
  personTray.innerHTML = "";
  generatedPeople = 0;
  if (transitionStarted || transitionPending) resetTransition();
  runState.textContent = message;
}

async function responseError(response) {
  return (await responseErrorDetails(response)).message;
}

async function responseErrorDetails(response) {
  try {
    const payload = await response.json();
    return {
      message: payload.error || `请求失败 (${response.status})`,
      code: payload.code || "request_failed",
    };
  } catch (_error) {
    return {
      message: `请求失败 (${response.status})`,
      code: "request_failed",
    };
  }
}

function createPersonIcon(index) {
  const icon = document.createElement("span");
  icon.className = `person-icon person-variant-${(index % 4) + 1}`;
  icon.style.animationDelay = `${Math.min(index * .012, .22)}s`;
  icon.setAttribute("role", "img");
  icon.setAttribute("aria-label", `AI 数字人 ${index}`);
  icon.innerHTML = `
    <svg viewBox="0 0 48 58" aria-hidden="true" focusable="false">
      <path class="person-shadow" d="M11 53c5.8 2.4 18.6 2.2 26-.5" />
      <circle class="person-head" cx="24" cy="10" r="6.2" />
      <path class="person-blindfold" d="M17.6 9.5c4.3 1.5 8.6 1.5 12.9-.2" />
      <path class="person-neck" d="M23.4 16.4c.2 2.2-.4 3.7-1.7 4.7" />
      <path class="person-body" d="M18.5 20.5c-4.7 5.2-6.2 16.4-4 27.5 5.8 1.9 13.7 2 20.1-.3 1.4-11.4-.8-21.4-5.1-27.2-3.4 1.4-7.1 1.4-11 0Z" />
      <path class="person-fold person-fold-a" d="M21.5 22.5c-2.8 6.8-3.5 15.4-2.3 25.3" />
      <path class="person-fold person-fold-b" d="M28.7 22.8c2.1 7 2.1 15 .5 24.3" />
      <path class="person-arm person-arm-left" d="M18.2 25.5c-4.1 3-6.8 6.4-8.7 10.4" />
      <path class="person-arm person-arm-right" d="M30.6 25.1c4.7 2 7.5 5.1 9.4 9.8" />
      <path class="person-cane" d="M39.4 34.2 43 55.4" />
      <path class="person-foot person-foot-left" d="M17.8 48.2c-1 2.6-2.7 4.4-5.3 5.2" />
      <path class="person-foot person-foot-right" d="M30.8 47.7c1.2 2.7 3.2 4.5 6 5.3" />
    </svg>
  `;
  return icon;
}

function resetTransition() {
  window.clearTimeout(reportHandoffTimer);
  window.clearTimeout(reportNavigationTimer);
  transitionPending = false;
  transitionStarted = false;
  transitionPhase = "handoff";
  forensicCompleted = 0;
  latestStatus = null;
  transition.className = "transition";
  transition.setAttribute("aria-hidden", "true");
  transitionPeople.innerHTML = "";
  puzzlePieces.innerHTML = "";
  shadowTraces.innerHTML = "";
  reportLink.hidden = true;
  reportLink.href = "#";
  transitionResume.hidden = true;
  document.body.classList.remove("puzzle-running");
}

async function restoreActiveRun() {
  const savedRunId = localStorage.getItem("bme:active-run");
  if (!savedRunId) return;

  questionInput.value = localStorage.getItem("bme:last-question") || "";
  runState.textContent = "正在恢复上一次运行状态";

  try {
    const response = await fetch(`/api/runs/${encodeURIComponent(savedRunId)}`, {
      cache: "no-store",
    });
    if (!response.ok) throw new Error(await responseError(response));
    const status = await response.json();

    if (status.status === "unknown") {
      clearInactiveRun("上一次任务记录已经失效，可以输入新的问题");
      return;
    }

    activeRunId = savedRunId;

    if (status.status === "succeeded") {
      clearInactiveRun(
        status.report_ready
          ? "上一次分析已经完成，可以输入新的问题"
          : "上一次运行已经结束，可以输入新的问题",
      );
      return;
    }

    imageStage.classList.add("generating");
    if (status.status === "recoverable_failed") {
      updateFromStatus(status);
      showRecoverableRun(status);
      return;
    }
    if (status.status === "failed") {
      showRunError(status.error || "上一次运行没有通过系统检查");
      return;
    }
    if (!ACTIVE_RUN_STATUSES.has(status.status)) {
      clearInactiveRun("上一次运行已经结束，可以输入新的问题");
      return;
    }

    isGenerating = true;
    questionInput.disabled = true;
    updateFromStatus(status);
    schedulePoll(100);
  } catch (error) {
    activeRunId = savedRunId;
    showRunError(error.message || "暂时无法读取上一次运行状态");
  }
}

function runTransitionDemo() {
  const phase = demoParams.get("phase") || "auto";
  const question = demoParams.get("question") || "美股 AI 泡沫会不会很快破灭？";
  const recordingMode = demoParams.get("record") === "1";
  const demoReportUrl = demoParams.get("report") || "";
  const requestedDelay = Number(demoParams.get("delay") || 380);
  const openingDelay = Math.min(2400, Math.max(120, requestedDelay));
  questionInput.value = question;
  imageStage.classList.add("generating");
  ensurePersonIcons(PERSON_COUNT);
  const baseStatus = {
    question,
    status: "running",
    current_stage: "streaming_front",
    cohort: { completed: PERSON_COUNT, total: PERSON_COUNT },
    retrieval: { completed: 16, total: PERSON_COUNT },
    digital_people: { completed: 11, total: PERSON_COUNT },
    semantic_verdicts: { completed: 7, total: PERSON_COUNT },
    analysis_layers: {
      forensic: { status: "running" },
      detective: { status: "pending" },
      contour: { status: "pending" },
    },
  };
  latestStatus = baseStatus;
  window.setTimeout(() => {
    enterPuzzleTransition(baseStatus, true);
    if (phase !== "auto") {
      window.setTimeout(() => {
        const targetPhase = phase === "report" ? "complete" : phase;
        const progress = targetPhase === "forensic" ? 9 : PERSON_COUNT;
        setForensicProgress(progress);
        setTransitionPhase(targetPhase, targetPhase === "complete" ? { report_url: "#" } : baseStatus);
        if (phase === "report") {
          window.setTimeout(() => transition.classList.add("report-handoff"), 900);
        }
      }, 90);
      return;
    }
    setForensicProgress(4);
    let count = 4;
    const revealTimer = window.setInterval(() => {
      count = Math.min(PERSON_COUNT, count + 2);
      setForensicProgress(count);
      if (count === PERSON_COUNT) window.clearInterval(revealTimer);
    }, 180);
    window.setTimeout(() => setTransitionPhase("detective", baseStatus), 2600);
    window.setTimeout(() => setTransitionPhase("contour", baseStatus), 5200);
    window.setTimeout(() => {
      const completedStatus = { report_url: demoReportUrl || "#" };
      if (demoReportUrl) reportLink.href = demoReportUrl;
      setTransitionPhase("complete", completedStatus);
    }, 8000);
    if (recordingMode) {
      window.setTimeout(() => {
        if (demoReportUrl) markReportArrival();
        transition.classList.add("report-handoff");
      }, 10200);
      if (demoReportUrl) {
        window.setTimeout(() => window.location.assign(demoReportUrl), 11600);
      }
    }
  }, openingDelay);
}

bootstrapApp();
