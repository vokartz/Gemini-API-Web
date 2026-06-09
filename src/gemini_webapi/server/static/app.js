// Ortak yardımcılar ve düzen (header/footer) — tüm sayfalar bunu kullanır.
"use strict";

const $ = (id) => document.getElementById(id);
const esc = (value) =>
  String(value ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );

function fmt(value) {
  if (!value) return "-";
  try { return new Date(value).toLocaleString("tr-TR"); }
  catch (e) { return String(value); }
}

function fmtDuration(seconds) {
  const v = Math.max(0, Number(seconds || 0));
  if (!v) return "yakında";
  const h = Math.floor(v / 3600);
  const m = Math.ceil((v % 3600) / 60);
  return h > 0 ? `${h}sa ${m ? m + "dk" : ""}`.trim() : `${m}dk`;
}

function getApiKey() { return localStorage.getItem("gemini-api-key") || ""; }
function authHeaders(extra = {}) {
  const headers = { ...extra };
  const key = getApiKey();
  if (key) headers["Authorization"] = `Bearer ${key}`;
  return headers;
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    ...options,
    headers: authHeaders({ "Content-Type": "application/json", ...(options.headers || {}) }),
  });
  let data = null;
  try { data = await res.json(); } catch (e) {}
  if (!res.ok) {
    const detail = (data && (data.detail || (data.error && data.error.message))) || `HTTP ${res.status}`;
    throw new Error(detail);
  }
  return data;
}

let toastTimer = null;
function toast(message, bad = false) {
  let el = $("toast");
  if (!el) {
    el = document.createElement("div");
    el.id = "toast";
    el.className = "toast";
    document.body.appendChild(el);
  }
  el.textContent = message;
  el.className = `toast show ${bad ? "bad" : ""}`.trim();
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.className = "toast"; }, 3200);
}

// ---------- Layout (sidebar + login) ----------
const ICONS = {
  generate: '<svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="2"/><path d="M21 15l-5-5L5 21"/></svg>',
  accounts: '<svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>',
  gems: '<svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 3h12l4 6-10 13L2 9z"/><path d="M11 3 8 9l4 13 4-13-3-6"/><path d="M2 9h20"/></svg>',
  api: '<svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>',
  history: '<svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v5h5"/><path d="M3.05 13A9 9 0 1 0 6 5.3L3 8"/><path d="M12 7v5l4 2"/></svg>',
};
const NAV = [
  { href: "/", label: "Görsel Üretimi", icon: "generate" },
  { href: "/history.html", label: "Üretim Geçmişi", icon: "history" },
  { href: "/accounts.html", label: "Hesaplar & Kota", icon: "accounts" },
  { href: "/gems.html", label: "Gems", icon: "gems" },
  { href: "/api.html", label: "API & Ayarlar", icon: "api" },
];

function renderLayout() {
  const path = window.location.pathname;
  const navHtml = NAV.map((n) => {
    const active = n.href === path || (n.href === "/" && (path === "/index.html" || path === "/"));
    return `<a href="${n.href}" class="${active ? "active" : ""}">${ICONS[n.icon]}<span class="label">${esc(n.label)}</span></a>`;
  }).join("");

  // Mevcut içeriği (.content > main) bir sarmalayıcıya taşı. Script düğümleri body'de kalır.
  const content = document.createElement("div");
  content.className = "content";
  Array.from(document.body.children).forEach((node) => {
    if (node.tagName !== "SCRIPT") content.appendChild(node);
  });

  const sidebar = document.createElement("aside");
  sidebar.className = "sidebar";
  sidebar.innerHTML = `
    <div class="brand">
      <img src="/static/logo.svg" alt="logo">
      <div class="name">Studio<small>Gemini Görsel</small></div>
    </div>
    <nav class="side-nav">${navHtml}</nav>
    <div class="side-foot">
      <span class="health"><span class="dot" id="healthDot"></span><span id="healthText">Bağlanıyor…</span></span>
      <button id="logoutBtn" class="small ghost" style="display:none">Çıkış yap</button>
    </div>`;

  document.body.appendChild(sidebar);
  document.body.appendChild(content);

  $("logoutBtn").onclick = async () => {
    await fetch("/v1/admin/logout", { method: "POST" });
    location.reload();
  };

  buildLoginModal();
}

function buildLoginModal() {
  const backdrop = document.createElement("div");
  backdrop.className = "modal-backdrop";
  backdrop.id = "loginBackdrop";
  backdrop.innerHTML = `
    <div class="modal">
      <div class="modal-head">Yönetici Girişi</div>
      <div class="modal-body">
        <div class="hint">Bu panel şifre korumalı. Devam etmek için yönetici şifresini girin.</div>
        <label>Şifre<input id="loginPassword" type="password" placeholder="Yönetici şifresi"></label>
        <button id="loginBtn" class="primary">Giriş Yap</button>
      </div>
    </div>`;
  document.body.appendChild(backdrop);
  $("loginBtn").onclick = doLogin;
  $("loginPassword").addEventListener("keydown", (e) => { if (e.key === "Enter") doLogin(); });
}

async function doLogin() {
  try {
    await api("/v1/admin/login", { method: "POST", body: JSON.stringify({ password: $("loginPassword").value }) });
    location.reload();
  } catch (e) { toast(e.message, true); }
}

async function checkAdmin() {
  try {
    const res = await fetch("/v1/admin/status");
    const data = await res.json();
    const locked = data.enabled && !data.authenticated;
    $("loginBackdrop").classList.toggle("open", locked);
    $("logoutBtn").style.display = data.enabled && data.authenticated ? "" : "none";
    return !locked;
  } catch (e) { return true; }
}

async function refreshHealth() {
  try {
    const status = await api("/v1/status");
    $("healthDot").classList.add("ok");
    $("healthText").textContent = `Çevrimiçi · Hesap #${status.current_account_id ?? "-"}`;
    return status;
  } catch (e) {
    $("healthDot").classList.remove("ok");
    $("healthText").textContent = "Çevrimdışı";
    return null;
  }
}

// Her sayfa bunu çağırır: layout + admin kontrolü + sayfa init.
async function bootstrap(initFn) {
  renderLayout();
  const ok = await checkAdmin();
  await refreshHealth();
  if (ok && typeof initFn === "function") {
    try { await initFn(); }
    catch (e) { toast(e.message, true); }
  }
}

// ---------- Gem seçici (üretim + gems sayfası paylaşır) ----------
async function loadGems() {
  const data = await api("/v1/custom-gems");
  return data.gems || [];
}

// ---------- Image zoom modal (tüm sayfalar) ----------
function ensureZoomModal() {
  let bd = $("zoomBackdrop");
  if (bd) return bd;
  bd = document.createElement("div");
  bd.id = "zoomBackdrop";
  bd.className = "modal-backdrop";
  bd.innerHTML = `
    <div class="modal xl">
      <div class="modal-head"><span id="zoomTitle">Görsel</span><button id="zoomClose">Kapat ✕</button></div>
      <div class="modal-img"><img id="zoomImg" alt="görsel"></div>
      <div class="modal-foot">
        <a id="zoomDownload" class="btn" download target="_blank">İndir</a>
        <button class="primary" id="zoomCloseBtn">Tamam</button>
      </div>
    </div>`;
  document.body.appendChild(bd);
  const close = () => bd.classList.remove("open");
  bd.addEventListener("click", (e) => { if (e.target === bd) close(); });
  $("zoomClose").onclick = close;
  $("zoomCloseBtn").onclick = close;
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") close(); });
  return bd;
}
function openZoom(url, title) {
  const bd = ensureZoomModal();
  $("zoomImg").src = url;
  $("zoomTitle").textContent = title || "Görsel";
  $("zoomDownload").href = url;
  bd.classList.add("open");
}

// ---------- Generic confirm modal ----------
function confirmDialog(message, { title = "Onay", danger = true } = {}) {
  return new Promise((resolve) => {
    const bd = document.createElement("div");
    bd.className = "modal-backdrop open";
    bd.innerHTML = `
      <div class="modal">
        <div class="modal-head">${esc(title)}</div>
        <div class="modal-body"><div>${esc(message)}</div></div>
        <div class="modal-foot">
          <button data-act="no">Vazgeç</button>
          <button class="${danger ? "danger" : "primary"}" data-act="yes">Onayla</button>
        </div>
      </div>`;
    document.body.appendChild(bd);
    const done = (v) => { bd.remove(); resolve(v); };
    bd.addEventListener("click", (e) => { if (e.target === bd) done(false); });
    bd.querySelector('[data-act="no"]').onclick = () => done(false);
    bd.querySelector('[data-act="yes"]').onclick = () => done(true);
  });
}

// ---------- File upload (referans görseller) ----------
async function uploadFile(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/v1/gemini/files", { method: "POST", headers: authHeaders(), body: form });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `Yükleme hatası (${res.status})`);
  return data.file;
}

function fmtBytes(n) {
  const v = Number(n || 0);
  if (v < 1024) return `${v} B`;
  if (v < 1024 * 1024) return `${(v / 1024).toFixed(1)} KB`;
  return `${(v / 1024 / 1024).toFixed(1)} MB`;
}

function fmtSeconds(ms) {
  const v = Math.max(0, Number(ms || 0)) / 1000;
  return `${v.toFixed(1)}s`;
}

// Asenkron buton işlemleri sırasında spinner gösterir ve butonu devre dışı bırakır.
async function withLoading(btn, fn) {
  if (!btn) return fn();
  const original = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span> ${esc(btn.textContent.trim())}`;
  try {
    return await fn();
  } finally {
    btn.disabled = false;
    btn.innerHTML = original;
  }
}
