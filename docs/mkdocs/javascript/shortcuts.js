const isMac = window.navigator.userAgent.toLowerCase().includes("mac")
const shortcut = isMac ? "⌘+k" : "ctrl+k";
document.documentElement.style.setProperty('--md-search-extension-shortcut', shortcut);

document.addEventListener('keydown', function (event) {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
    event.preventDefault();
    const query = document.querySelector('[data-md-component="search-query"]');
    if (query) {
      query.focus();
      query.select();
    }
  }
});
