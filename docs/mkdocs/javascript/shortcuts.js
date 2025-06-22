const isMac = window.navigator.userAgent.toLowerCase().includes("mac")
const shortcut = isMac ? "⌘+k" : "ctrl+k";
document.documentElement.style.setProperty('--md-search-extension-shortcut', shortcut);

keyboard$.subscribe(key => {
  const query = document.querySelector('[data-md-component="search-query"]')
  if (key.mode === 'global' && key.type === 'k' && (key.ctrlKey || key.metaKey)) {
    key.claim();
    query.focus();
    query.select();
  }
})
