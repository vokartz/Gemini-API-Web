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

// ---------- Layout (header + footer + login) ----------
const NAV = [
  { href: "/", label: "Görsel Üretimi" },
  { href: "/accounts.html", label: "Hesaplar & Kota" },
  { href: "/gems.html", label: "Gems" },
  { href: "/api.html", label: "API" },
];

function renderLayout() {
  const path = window.location.pathname;
  const navHtml = NAV.map((n) => {
    const active = n.href === path || (n.href === "/" && path === "/index.html");
    return `<a href="${n.href}" class="${active ? "active" : ""}">${esc(n.label)}</a>`;
  }).join("");

  const header = document.createElement("header");
  header.className = "site-header";
  header.innerHTML = `
    <div class="brand"><span class="logo">G</span> Gemini Panel</div>
    <nav class="site-nav">${navHtml}</nav>
    <div class="header-right">
      <span class="health"><span class="dot" id="healthDot"></span><span id="healthText">Bağlanıyor…</span></span>
      <button id="logoutBtn" class="small ghost" style="display:none">Çıkış</button>
    </div>`;
  document.body.prepend(header);

  const footer = document.createElement("footer");
  footer.className = "site-footer";
  footer.innerHTML = `<span>Gemini API Web · çok hesaplı görsel üretimi</span><span>${esc(window.location.host)}</span>`;
  document.body.appendChild(footer);

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
