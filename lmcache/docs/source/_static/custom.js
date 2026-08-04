/**
 * Add the RunLLM assistant widget script once per page.
 *
 * @returns {void}
 */
function addRunLlmWidget() {
  if (document.getElementById("runllm-widget-script")) {
    return;
  }

  var script = document.createElement("script");
  script.type = "module";
  script.id = "runllm-widget-script";

  script.src = "https://widget.runllm.com";

  script.setAttribute("version", "stable");
  script.setAttribute("crossorigin", "true");
  script.setAttribute("runllm-keyboard-shortcut", "Mod+j");
  script.setAttribute("runllm-name", "LMCache Assistant");
  script.setAttribute("runllm-position", "BOTTOM_RIGHT");
  script.setAttribute("runllm-assistant-id", "1185");

  script.async = true;
  document.head.appendChild(script);
}

/**
 * Return true when the current page is rendered under the Chinese docs prefix.
 *
 * @returns {boolean} Whether the current path is a Chinese documentation page.
 */
function isChineseDocsPage() {
  return window.location.pathname.split("/").includes("zh_CN");
}

/**
 * Build the target language URL for the current page.
 *
 * @param {"en" | "zh_CN"} language The target language.
 * @returns {string} The target URL.
 */
function buildLanguageUrl(language) {
  var pathParts = window.location.pathname.split("/");
  var zhIndex = pathParts.indexOf("zh_CN");

  if (language === "zh_CN" && zhIndex === -1) {
    pathParts.splice(1, 0, "zh_CN");
  } else if (language === "en" && zhIndex !== -1) {
    pathParts.splice(zhIndex, 1);
  }

  var nextPath = pathParts.join("/") || "/";
  return nextPath + window.location.search + window.location.hash;
}

/**
 * Build the home URL for a target language.
 *
 * @param {"en" | "zh_CN"} language The target language.
 * @returns {string} The target language home URL.
 */
function buildLanguageHomeUrl(language) {
  return language === "zh_CN" ? "/zh_CN/" : "/";
}

/**
 * Fall back to the target language home page if the equivalent page is absent.
 *
 * @param {HTMLAnchorElement} link The language link to validate.
 * @param {"en" | "zh_CN"} language The target language.
 * @returns {void}
 */
function fallbackMissingLanguagePage(link, language) {
  window
    .fetch(link.href, { method: "HEAD" })
    .then(function (response) {
      if (!response.ok) {
        link.href = buildLanguageHomeUrl(language);
      }
    })
    .catch(function () {
      link.href = buildLanguageHomeUrl(language);
    });
}

/**
 * Add a compact language switcher to the docs header.
 *
 * @returns {void}
 */
function addLanguageSwitcher() {
  if (document.querySelector(".lmcache-language-switcher")) {
    return;
  }

  var switcher = document.createElement("div");
  var chineseLink = document.createElement("a");
  var divider = document.createElement("span");
  var englishLink = document.createElement("a");
  var isChinesePage = isChineseDocsPage();

  switcher.className = "lmcache-language-switcher";
  switcher.setAttribute("aria-label", "Documentation language");

  chineseLink.href = buildLanguageUrl("zh_CN");
  chineseLink.textContent = "中文";
  chineseLink.setAttribute("aria-label", "Switch to Chinese");

  divider.className = "lmcache-language-switcher__divider";
  divider.textContent = "|";

  englishLink.href = buildLanguageUrl("en");
  englishLink.textContent = "Eng";
  englishLink.setAttribute("aria-label", "Switch to English");

  if (isChinesePage) {
    chineseLink.setAttribute("aria-current", "page");
  } else {
    englishLink.setAttribute("aria-current", "page");
  }

  switcher.appendChild(chineseLink);
  switcher.appendChild(divider);
  switcher.appendChild(englishLink);

  // Place the switcher in the top nav bar with the other icons.
  // If the nav bar isn't there, show it as a floating button instead.
  var navbar = findDocsNavbar();
  if (navbar) {
    navbar.appendChild(switcher);
  } else {
    switcher.classList.add("lmcache-language-switcher--fallback");
    document.body.appendChild(switcher);
  }

  fallbackMissingLanguagePage(chineseLink, "zh_CN");
  fallbackMissingLanguagePage(englishLink, "en");
}

/**
 * Locate the top nav bar that holds the GitHub / profile / theme-toggle
 * icons. Prefers the structural `header nav` selector; falls back to
 * the GitHub link's parent if the theme markup differs.
 *
 * @returns {HTMLElement | null} The nav bar element, or null if not found.
 */
function findDocsNavbar() {
  var navbar = document.querySelector("header nav");
  if (navbar) {
    return navbar;
  }
  var githubLink = document.querySelector('a[title="Visit GitHub"]');
  return githubLink ? githubLink.parentElement : null;
}

/**
 * Initialize docs widgets after the DOM is ready.
 *
 * @returns {void}
 */
function initializeDocsWidgets() {
  addLanguageSwitcher();
  addRunLlmWidget();
}

document.addEventListener("DOMContentLoaded", initializeDocsWidgets);

/**
 * Default-expand the top-level sidebar sections so visitors can skim the full
 * information architecture at a glance. Sections remain collapsible -- this
 * only changes the initial Alpine ``expanded`` state of each ``.toctree-l1``
 * (the theme starts them collapsed unless they are the current page).
 *
 * @returns {void}
 */
// Top-level sections that should start collapsed (secondary / reference /
// deprecated material) rather than expanded-by-default.
const COLLAPSED_BY_DEFAULT = [
  "legacy/index",
  "developer_guide/index",
  "non_kv_cache/index",
  "community/index",
  "kv_cache_optimizations/index",
];

let didInitialNavExpand = false;

function expandTopLevelNavSections() {
  // Run the default expansion exactly once. Without this guard it fires on
  // both `alpine:initialized` and the DOMContentLoaded fallback, and if the
  // user collapses a section between the two calls the second one re-opens it.
  if (didInitialNavExpand) {
    return;
  }
  // Need Alpine ready to set the reactive `expanded` state; bail without
  // marking done so a later trigger can retry.
  if (!(window.Alpine && window.Alpine.$data)) {
    return;
  }
  let expandedAny = false;
  document.querySelectorAll("nav .toctree-l1").forEach((li) => {
    const link = li.querySelector(":scope > a");
    const href = link ? link.getAttribute("href") || "" : "";
    if (COLLAPSED_BY_DEFAULT.some((p) => href.includes(p))) {
      return;
    }
    try {
      const data = window.Alpine.$data(li);
      if (data && "expanded" in data) {
        data.expanded = true;
        expandedAny = true;
      }
    } catch (e) {
      /* nav item not yet managed by Alpine */
    }
  });
  if (expandedAny) {
    didInitialNavExpand = true;
  }
}

// Run both on Alpine init and as a fallback, to cover either load ordering.
document.addEventListener("alpine:initialized", expandTopLevelNavSections);
document.addEventListener("DOMContentLoaded", () =>
  setTimeout(expandTopLevelNavSections, 0),
);

/**
 * Preserve the left sidebar's scroll position across page navigations. The
 * sidebar (``#left-sidebar``) is an independently scrollable container, so a
 * normal page load re-renders it at the top -- making the nav "jump" every
 * time you click a link. We stash its scrollTop in sessionStorage before
 * leaving and restore it on the next page.
 *
 * @returns {void}
 */
const SIDEBAR_SCROLL_KEY = "lmcacheSidebarScroll";

function saveSidebarScroll() {
  const sidebar = document.getElementById("left-sidebar");
  if (sidebar) {
    sessionStorage.setItem(SIDEBAR_SCROLL_KEY, String(sidebar.scrollTop));
  }
}

function restoreSidebarScroll() {
  const sidebar = document.getElementById("left-sidebar");
  const saved = sessionStorage.getItem(SIDEBAR_SCROLL_KEY);
  if (!sidebar || saved === null) {
    return;
  }
  const target = parseInt(saved, 10) || 0;
  // Sections expand asynchronously (Alpine reveals the child <ul> via x-show),
  // so the sidebar starts short and scrollTop gets clamped near the top.
  // Re-apply the target across several animation frames until it sticks.
  let tries = 0;
  const apply = () => {
    sidebar.scrollTop = target;
    tries += 1;
    if (tries < 20 && Math.abs(sidebar.scrollTop - target) > 1) {
      requestAnimationFrame(apply);
    }
  };
  requestAnimationFrame(apply);
}

// Save the position whenever a sidebar link is clicked (captures it exactly
// at click time) and as a fallback right before the page is hidden.
document.addEventListener("DOMContentLoaded", () => {
  const sidebar = document.getElementById("left-sidebar");
  if (sidebar) {
    sidebar.addEventListener("click", (e) => {
      if (e.target.closest("a")) {
        saveSidebarScroll();
      }
    });
  }
});
window.addEventListener("pagehide", saveSidebarScroll);
// Restore on initial load and on bfcache restore (pageshow).
document.addEventListener("DOMContentLoaded", restoreSidebarScroll);
window.addEventListener("pageshow", restoreSidebarScroll);
