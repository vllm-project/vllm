// If user directly navigates to a grouped tab
// sync the tab state by clicking the corresponding label
window.addEventListener("load", function () {
  // Only run if there's an anchor in the URL
  const anchor = window.location.hash;
  if (!anchor) {
    return;
  }

  // Only if there's a tabbed-labels div
  const labelsDiv = document.querySelector("div.tabbed-set");
  if (!labelsDiv) {
    return;
  }

  // Strip any _1, _2, etc. suffixes from the anchor
  const baseAnchor = anchor.replace(/_\d+$/, '');
  // Find and click the matching label
  for (const label of labelsDiv.querySelectorAll("a")) {
    if (label.getAttribute("href") === baseAnchor) {
      label.click();
      // Update URL to use the base anchor (remove suffix)
      if (baseAnchor !== anchor) {
        history.replaceState(null, '', baseAnchor);
      }
      return;
    }
  }

});