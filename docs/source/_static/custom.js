// Add RunLLM widget
document.addEventListener("DOMContentLoaded", function () {
    var script = document.createElement("script");
    script.type = "module";
    script.id = "runllm-widget-script"
  
    script.src = "https://widget.runllm.com";
  
    script.setAttribute("version", "stable");
    script.setAttribute("runllm-keyboard-shortcut", "Mod+j"); // cmd-j or ctrl-j to open the widget.
    script.setAttribute("runllm-name", "vLLM");
    script.setAttribute("runllm-position", "BOTTOM_RIGHT");
    script.setAttribute("runllm-position-y", "20%");
    script.setAttribute("runllm-position-x", "3%");
    script.setAttribute("runllm-assistant-id", "207");
  
    script.async = true;
    document.head.appendChild(script);
  });

// Update URL search params when tab is clicked
  document.addEventListener("DOMContentLoaded", function () {
    const tabs = document.querySelectorAll(".sd-tab-label");

    function updateURL(tab) {
      const syncGroup = tab.getAttribute("data-sync-group");
      const syncId = tab.getAttribute("data-sync-id");
      if (syncGroup && syncId) {
          const url = new URL(window.location);
          url.searchParams.set(syncGroup, syncId);
          window.history.replaceState(null, "", url);
      }
    }

    tabs.forEach(tab => {
        tab.addEventListener("click", () => updateURL(tab));
    });
});
