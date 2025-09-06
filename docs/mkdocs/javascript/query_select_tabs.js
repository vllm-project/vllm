// If user directly navigates to a grouped tab
// sync the tab state by clicking the corresponding label
function syncGroupedTabs() {
  // Only run on installation pages with grouped tabs
  const currentPath = window.location.pathname;
  if (!currentPath.endsWith('installation/gpu.html') && 
    !currentPath.endsWith('installation/cpu.html')) {
    return;
  }

  const anchor = window.location.hash;

  if (anchor) {
    const label = document.querySelector(`a[href="${anchor}"]`);

    if (label) {
      label.click();
    }
  }
}

// Run once page has fully loaded
window.addEventListener('load', syncGroupedTabs);
